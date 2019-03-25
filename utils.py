import torch
import matplotlib.pyplot as plt


def calc_weight(label, num_classes):
    avg_number = label.shape[0] / num_classes
    weight = torch.full((num_classes, ), 1)
    for i, w in enumerate(weight):
        total_number = (label == i).type(torch.int).sum()
        weight[i] = avg_number / total_number.type(torch.float)
    return weight


# def plot_result(train_loss, test_loss, train_acc, test_acc):
def plot_result(train_loss, test_loss, train_acc, test_acc, train_ma_acc, test_ma_acc):
    plt.subplots()
    plt.plot(train_acc, label='train_acc')
    plt.plot(test_acc, label='test_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.subplots()
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.subplots()
    plt.plot(train_ma_acc, label='train_ma_acc')
    plt.plot(test_ma_acc, label='test_ma_acc')
    plt.xlabel('epoch')
    plt.ylabel('manual accuracy')
    plt.legend()
    plt.show()
