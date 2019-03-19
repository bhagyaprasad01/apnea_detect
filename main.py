import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from csv_dataloader import get_apnea_dataloader
from model.apneanet6 import get_apnea_model
from torch.autograd import Variable
import time
import util


def train(args):
    # for plot
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    # hyperparameter
    batch_size = args['batch_size']
    lr = args['lr']
    # weight_decay = args['weight_decay']
    num_epochs = args['num_epochs']

    # get model
    model = get_apnea_model()
    # assuming you have a GPU
    model.cuda()
    # get dataset loader
    train_loader, test_loader = get_apnea_dataloader(batch_size)
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, verbose=1, patience=3)
    # log
    best_acc = 0
    log_interval = 1
    test_interval = 1

    # start training
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_acc = 0.0
        running_loss = 0.0
        for steps, (inputs, labels, type_id) in enumerate(train_loader):
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            type_id = Variable(type_id.cuda())
            optimizer.zero_grad()
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # BackPropagation
            loss.backward()
            optimizer.step()
            # ========================= Log ======================
            if steps % log_interval == 0:
                corrects = (torch.max(outputs, 1)[1].view(labels.size()).data == labels.data).sum()
                accuracy = 100.0 * corrects / batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps + 1,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch_size))
            running_loss += loss.item()
            running_acc += accuracy.item()
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(running_acc / len(train_loader))
        scheduler.step(running_loss / len(train_loader))
        # if epoch == 19:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
        # if epoch == 39:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
        print("\n[epoch {} - loss:{:.6f} acc{:.3f}".format(epoch + 1,
                                                           running_loss / len(train_loader),
                                                           running_acc / len(train_loader)))
        print('Evaluating {}'.format(epoch + 1))
        running_test_acc, running_test_loss = test(model, test_loader)
        test_acc.append(running_test_acc)
        test_loss.append(running_test_loss)
        if running_test_acc > best_acc:
            best_acc = running_test_acc
    print('Finished Training, using {} seconds'.format(round(time.time()-start_time)))
    print('Best accuracy in evaluation set: {}'.format(best_acc))
    util.show_acc(train_acc, test_acc)
    util.show_loss(train_loss, test_loss)


def test(model, test_loader):
    model.eval()
    corrects, avg_loss = 0, 0
    for steps, (inputs, labels, type_id) in enumerate(test_loader):
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        type_id = Variable(type_id.cuda())

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        avg_loss += loss.item()
        corrects += (torch.max(outputs, 1)
                     [1].view(labels.size()).data == labels.data).sum()

    size = len(test_loader.dataset)
    avg_loss /= len(test_loader)
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy, avg_loss


def main():
    args = {
        # hyperparameter
        'batch_size': 256,
        'lr': 0.001,  # learning rate, best acc 71%
        'weight_decay': 0.9,
        'num_epochs': 50
    }
    # start to train
    train(args)


if __name__ == '__main__':
    main()
