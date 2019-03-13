import torch
import torch.nn as nn
import torch.optim as optim
from csv_dataloader import get_apnea_dataloader
from apneanet import get_apnea_model
from torch.autograd import Variable
import matplotlib.pyplot as plt


def train(model, train_loader, test_loader, batch_size):
    model.train()

    # for plot
    acc_list = []

    start_epoch = 0
    total_epoch = 100
    total_loss = 0

    # TODO: use some advanced parameters
    criterion = nn.CrossEntropyLoss()

    # TODO: try other optimizer
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.5)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lr = 0.001     # init
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(start_epoch, total_epoch):
        print('\n-----> epoch %d ' % epoch)
        running_loss = 0.0
        print("len(train_loader) % batch_size :", round(len(train_loader) / batch_size))
        len_of_batch = round(len(train_loader) / batch_size)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % batch_size == 0:
                total_loss += running_loss / batch_size
                print(" [%d, %5d] loss : %.3f, lr:%f" % (epoch + 1, i + 1, running_loss / batch_size, lr))
                running_loss = 0.0
        if epoch % 20 == 0:
            lr /= 10
        print('Testing')
        test_acc = test(model, test_loader)
        acc_list.append(test_acc)

    print('Finished Training')
    total_loss = total_loss / len_of_batch
    print("[%d epoch loss :%.3f" % (epoch + 1, total_loss))
    total_loss = 0
    plt.title('SGD Optimizer, batch size=64, sr=50')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(acc_list)
    plt.show()


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    # with torch.no_grad():
    for data in loader:
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print('Accuracy of the network on the %d test signals: %d %%' % (
            total, test_acc))
    return test_acc


def main():
    # get my model
    model = get_apnea_model()

    # assuming you have a GPU
    model.cuda()

    # get dataset loader
    batch_size = 32
    train_loader, test_loader = get_apnea_dataloader(batch_size)

    # start to train
    # test in every epoch
    train(model, train_loader, test_loader, batch_size)


if __name__ == '__main__':
    main()
