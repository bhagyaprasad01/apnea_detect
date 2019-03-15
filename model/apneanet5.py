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
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 24, 30),
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(24, 24, 30),
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(24, 24, 10),
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(24, 12, 10),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc = nn.Linear(192, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x= self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 192)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def get_apnea_model():
    model = ApneaNet()
    return model
