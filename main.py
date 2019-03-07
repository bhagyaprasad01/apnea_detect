import torch
import torch.nn as nn
import torch.optim as optim
from csv_dataloader import get_apnea_dataloader
from apneanet import get_apnea_model
from torch.autograd import Variable


def train(model, loader, batch_size, cuda_avail):
    model.train()

    start_epoch = 0
    total_epoch = 2
    total_loss = 0

    # TODO: use some advanced parameters
    criterion = nn.CrossEntropyLoss()

    # TODO: try other optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.9)

    for epoch in range(start_epoch, total_epoch):
        print('\n-----> epoch %d ' % epoch)
        running_loss = 0.0
        print("len(train_loader) % batch_size :", round(len(loader) / batch_size))
        len_of_batch = round(len(loader) / batch_size)

        for i, (inputs, labels) in enumerate(loader):
            if cuda_avail:
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
                print(" [%d, %5d] loss : %.3f" % (epoch + 1, i + 1, running_loss / batch_size))
                running_loss = 0.0

    print('Finished Training')
    total_loss = total_loss / len_of_batch
    print("[%d epoch loss :%.3f" % (epoch + 1, total_loss))
    total_loss = 0


def test(model, loader, cuda_avail):
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

    print('Accuracy of the network on the %d test signals: %d %%' % (
            total, 100 * correct / total))


def main():
    # get my awesome model
    model = get_apnea_model()

    # Check if gpu support is available
    # TODO: Use GPU model could have some more elegant implement ways
    cuda_avail = torch.cuda.is_available()

    if cuda_avail:
        model.cuda()

    # get dataset loader
    batch_size = 32
    train_loader, test_loader = get_apnea_dataloader(batch_size)

    # start to train
    train(model, train_loader, batch_size, cuda_avail)

    # start to test
    test(model, test_loader, cuda_avail)


if __name__ == '__main__':
    main()
