import matplotlib.pyplot as plt


def show_acc(train_acc, test_acc):
    plt.subplots()
    plt.plot(train_acc, label='train_acc')
    plt.plot(test_acc, label='test_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def show_loss(train_loss, test_loss):
    plt.subplots()
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def evaluation_result(pred, labels, tp, tn, fn, fp):
    tp += ((pred == 1) & (labels.data == 1)).sum()
    tn += ((pred == 0) & (labels.data == 0)).sum()
    fn += ((pred == 0) & (labels.data == 1)).sum()
    fp += ((pred == 1) & (labels.data == 0)).sum()
    return
