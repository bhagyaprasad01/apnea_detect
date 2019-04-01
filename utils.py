import torch
import matplotlib.pyplot as plt


def calc_weight(label, num_classes):
    avg_number = label.shape[0] / num_classes
    weight = torch.full((num_classes, ), 1)
    for i, w in enumerate(weight):
        total_number = (label == i).type(torch.int).sum()
        weight[i] = avg_number / total_number.type(torch.float)
    return weight


def calc_performance(cm):
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    acc = 100.0 * ((TP + TN) / (TP + TN + FP + FN))
    pre = 100.0 * (TP / (TP + FP))
    recall = 100.0 * (TP / (TP+ FN))
    f1 = 2 * recall * pre / (recall + pre)
    ret = {
        'cm': cm,
        'acc': acc,
        'pre': pre,
        'recall': recall,
        'f1': f1
    }
    return ret


def plot_result(train_loss, test_loss, train_acc, test_acc):
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
