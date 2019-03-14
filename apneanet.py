import torch.nn as nn


# TODO: add BN between every convolution layers
class ApneaNet(nn.Module):

    def __init__(self, ):
        super(ApneaNet, self).__init__()
        self.bn = nn.BatchNorm1d(2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 20, 50),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 20, 50),
            nn.BatchNorm1d(20),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(20, 24, 30),
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(24, 24, 30),
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(24, 24, 10),
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(24, 12, 10),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc = nn.Linear(192, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Input   Size
        conv1   2*2000
        conv2   20*1951
        conv3   20*1902
        conv4   24*1873
        conv5   24*1844
        conv6   24*1835
        fc      12*16
        """
        x= self.bn(x)
        x = self.conv1(x)
        # in:20*1951
        # out:20*1902
        # TODO: add conv2 input to output
        out = self.conv2(x)
        out.sum(x.narrow(1, 0, 1901))
        x = out
        print('conv2 output:', x.shape)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 192)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def get_apnea_model():
    model = ApneaNet()
    return model
