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


from utils import calc_weight, calc_performance, plot_result

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Apnea Detection Networks')
    # Model options
    parser.add_argument('--train_data_path', default='data/2x50/train.mat', type=str)
    parser.add_argument('--test_data_path', default='data/2x50/test.mat', type=str)
    parser.add_argument('--nthread', default=4, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    # Training options
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save', default='./logs', type=str,
                        help='save parameters and logs in this folder')
    return parser.parse_args()


def create_dataset(mat_file):
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
    # model = ApneaNet5x1(args.dropout)
    model = ApneaNet6(args.dropout)
    model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())

    # optimizer = SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=loss_weight).cuda()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)

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
        if is_best:
            save_path = Path(args.save) / 'checkpoint_TestAcc{:.3f}.pth.tar'.format(data['test_acc'])
            log_path = Path(args.save) / 'log_best.txt'
        else:
            save_path = Path(args.save) / 'checkpoint.pth.tar'
            log_path = Path(args.save) / 'log.txt'

        # 保存模型
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': data['epoch'],
                    }, save_path)
        # 记录日志
        z = {**vars(args), **data}
        with open(log_path, 'a') as f:
            f.write(json.dumps(z) + '\n')
        print(z)

    def log_acc(acc):
        if not Path(args.save).exists():
            Path(args.save).mkdir()
        log_path = Path(args.save) / 'log_acc.txt'
        with open(log_path, 'a') as f:
            f.write(str(acc) + '\n')

    def train():
        model.train()
        meter_loss = tnt.meter.AverageValueMeter()
        confusion_matrix = tnt.meter.ConfusionMeter(2)
        train_iterator = tqdm(train_loader, dynamic_ncols=True)

        for x, y in train_iterator:
            optimizer.zero_grad()
            outputs = model(cast(x))
            loss = criterion(outputs, cast(y))
            loss.backward()
            optimizer.step()
            meter_loss.add(loss.item())
            train_iterator.set_postfix(loss=loss.item())
            confusion_matrix.add(outputs.data.cpu(), y.cpu())

        cm = calc_performance(confusion_matrix.value())
        return meter_loss.mean, cm['acc'], cm

    def test():
        model.eval()
        meter_loss = tnt.meter.AverageValueMeter()
        confusion_matrix = tnt.meter.ConfusionMeter(2)
        test_iterator = tqdm(test_loader, dynamic_ncols=True)
        for x, y in test_iterator:
            optimizer.zero_grad()
            outputs = model(cast(x))
            loss = criterion(outputs, cast(y))
            meter_loss.add(loss.item())
            confusion_matrix.add(outputs.data.cpu(), y.cpu())
        cm = calc_performance(confusion_matrix.value())
        return meter_loss.mean, cm['acc'], cm

    # 训练开始
    total_train_loss = np.zeros([args.epochs, 1])
    total_test_loss = np.zeros([args.epochs, 1])
    total_train_acc = np.zeros([args.epochs, 1])
    total_test_acc = np.zeros([args.epochs, 1])

    best_acc = 0
    best_log_data = {}

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        train_loss, train_acc, train_cm = train()
        test_loss, test_acc, test_cm = test()

        total_train_loss[epoch] = train_loss
        total_test_loss[epoch] = test_loss
        total_train_acc[epoch] = train_acc
        total_test_acc[epoch] = test_acc

        log_data = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "train_cm": str(train_cm),
            "test_cm": str(test_cm),
            "epoch": epoch,
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "lr": scheduler.get_lr(),
        }

        # 记录最好测试准确率和log信息
        if test_acc > best_acc:
            best_acc = test_acc
            best_log_data = log_data

        log(log_data)   # 保存每轮结果

        print('==> id: %s (%d/%d), train_acc: \33[91m%.2f\033[0m, test_acc: \33[91m%.2f\033[0m,' %
              (args.save, epoch, args.epochs, train_acc, test_acc))

    # 保存最佳结果
    print('best test acc: {}'.format(best_log_data['test_cm']))
    log(best_log_data, is_best=True)
    log_acc(total_test_acc)

    # show time
    plot_result(total_train_loss, total_test_loss, total_train_acc, total_test_acc)


if __name__ == '__main__':
    main()
