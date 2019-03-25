from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.backends import cudnn
import torchnet as tnt
from data.datasets import ApneaDataset1x512, ApneaDataset2x500
from models import MSResNet, ApneaNet5x1, ApneaNet6


from utils import calc_weight, plot_result

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Apnea Detection Networks')
    # Model options
    parser.add_argument('--train_data_path', default='data/2x50/train-06.mat', type=str)
    parser.add_argument('--test_data_path', default='data/2x50/test-06.mat', type=str)
    parser.add_argument('--nthread', default=4, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    # Training options
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--weight_decay', default=1e-1, type=float)
    parser.add_argument('--restarts', default='[2,4,8,16,32,64,128]', type=json.loads,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save', default='./logs', type=str,
                        help='save parameters and logs in this folder')
    return parser.parse_args()


def create_dataset(mat_file):
    # data_set = ApneaDataset1x512(mat_file)
    data_set = ApneaDataset2x500(mat_file)
    return data_set


def main():
    args = parse_args()
    print('parsed options:', vars(args))

    have_cuda = torch.cuda.is_available()

    def cast(x):
        return x.cuda() if have_cuda else x

    num_classes = args.num_classes

    def create_iterator(mode):
        if mode:
            dataset = create_dataset(args.train_data_path)
        else:
            dataset = create_dataset(args.test_data_path)
        weight = calc_weight(dataset.Y, num_classes)
        return DataLoader(dataset, args.batch_size, shuffle=mode,
                          num_workers=args.nthread, pin_memory=torch.cuda.is_available()), weight

    train_loader, loss_weight = create_iterator(True)
    test_loader, test_loss_weight = create_iterator(False)

    # model = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=num_classes)
    model = ApneaNet5x1(args.dropout)
    # model = ApneaNet6(args.dropout)
    model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())

    # optimizer = SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=loss_weight).cuda()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    def log(data, is_best=False):
        if not Path(args.save).exists():
            Path(args.save).mkdir()
        save_path = Path(args.save) / 'checkpoint.pth.tar'
        if is_best:
            save_path += '.best'
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': data['epoch'],
                    }, save_path)
        z = {**vars(args), **data}
        with open(Path(args.save) / 'log.txt', 'a') as f:
            f.write(json.dumps(z) + '\n')
        print(z)

    def train():
        model.train()
        meter_loss = tnt.meter.AverageValueMeter()
        class_acc = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = tnt.meter.ConfusionMeter(2)
        train_iterator = tqdm(train_loader, dynamic_ncols=True)
        running_acc = 0.0
        for x, y in train_iterator:
            optimizer.zero_grad()
            outputs = model(cast(x))
            loss = criterion(outputs, cast(y))
            loss.backward()
            optimizer.step()
            meter_loss.add(loss.item())
            train_iterator.set_postfix(loss=loss.item())
            class_acc.add(outputs.data.cpu(), y.cpu())
            confusion_matrix.add(outputs.data.cpu(), y.cpu())
            corrects = (torch.max(outputs, 1)[1].view(y.size()).data == cast(y).data).sum()
            accuracy = 100.0 * corrects / args.batch_size
            running_acc += accuracy.item()
        manual_accuracy = running_acc / len(train_iterator)

        return meter_loss.mean, class_acc.value()[0], confusion_matrix.value(), manual_accuracy

    def test():
        model.eval()
        meter_loss = tnt.meter.AverageValueMeter()
        class_acc = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = tnt.meter.ConfusionMeter(2)
        test_iterator = tqdm(test_loader, dynamic_ncols=True)
        running_acc = 0.0
        for x, y in test_iterator:
            optimizer.zero_grad()
            outputs = model(cast(x))
            loss = criterion(outputs, cast(y))
            meter_loss.add(loss.item())
            class_acc.add(outputs.data.cpu(), y.cpu())
            confusion_matrix.add(outputs.data.cpu(), y.cpu())
            corrects = (torch.max(outputs, 1)[1].view(y.size()).data == cast(y).data).sum()
            accuracy = 100.0 * corrects / args.batch_size
            running_acc += accuracy.item()
        manual_accuracy = running_acc / len(test_iterator)
        return meter_loss.mean, class_acc.value()[0], confusion_matrix.value(), manual_accuracy

    # 训练开始
    total_train_loss = np.zeros([args.epochs, 1])
    total_test_loss = np.zeros([args.epochs, 1])
    total_train_acc = np.zeros([args.epochs, 1])
    total_test_acc = np.zeros([args.epochs, 1])
    total_train_ma_acc = np.zeros([args.epochs, 1])
    total_test_ma_acc = np.zeros([args.epochs, 1])

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        # if epoch in args.restarts:
        #     scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=args.lr_min)
        train_loss, train_acc, train_cm, train_ma_acc = train()
        test_loss, test_acc, test_cm, test_ma_acc = test()
        log_data = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch": epoch,
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "lr": scheduler.get_lr(),
        }
        total_train_loss[epoch] = train_loss
        total_test_loss[epoch] = test_loss
        total_train_acc[epoch] = train_acc
        total_test_acc[epoch] = test_acc
        total_train_ma_acc[epoch] = train_ma_acc
        total_test_ma_acc[epoch] = test_ma_acc

        log(log_data)
        print('==> id: %s (%d/%d), train_acc: \33[91m%.2f\033[0m, test_acc: \33[91m%.2f\033[0m,'
              'train_manual_acc: %.3f, test_manual_acc: %.3f,train_cm: %s, test_cm: %s' %
              (args.save, epoch, args.epochs, train_ma_acc, test_ma_acc, train_acc, test_acc, str(train_cm), str(test_cm)))

    # show time
    plot_result(total_train_loss, total_test_loss, total_train_acc, total_test_acc, total_train_ma_acc, total_test_ma_acc)
    # plot_result(total_train_loss, total_test_loss, total_train_acc, total_test_acc)


if __name__ == '__main__':
    main()
