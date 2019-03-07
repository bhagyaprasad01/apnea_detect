import torch.nn as nn


# TODO: add BN between every convolution layers
class ApneaNet(nn.Module):

    def __init__(self, ):
        super(ApneaNet, self).__init__()
        self.bn = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 20, 50),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 20, 50),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(20, 24, 30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(24, 24, 30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.25)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(24, 24, 10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(24, 12, 10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )
        self.fc1 = nn.Linear(192, 192)
        self.fc2 = nn.Linear(192, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 192)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def get_apnea_model():
    model = ApneaNet()
    return model
