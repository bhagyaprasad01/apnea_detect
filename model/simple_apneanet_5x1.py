import torch.nn as nn


class SimpleApneaNet(nn.Module):

    def __init__(self, ):
        super(SimpleApneaNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 30, 5, stride=5),
            # nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(30, 30, 5, stride=5),
            # nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(30, 30, 5, stride=5),
            # nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.25)
        )
        self.fc = nn.Linear(60, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 60)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class SimpleApneaResNet(nn.Module):

    def __init__(self, ):
        super(SimpleApneaResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 30, 5, stride=5),
            # nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(30, 30, 5, stride=5),
            # nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(30, 30, 5, stride=5),
            # nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.5)
        )
        self.fc = nn.Linear(60, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_200 = self.conv1(x)
        x_200 += x.narrow(2, 0, 200)
        x_20 = self.conv2(x_200)
        x_20 += x_200.narrow(2, 0, 20)
        x_2 = self.conv3(x_20)
        x_2 += x_20.narrow(2, 0, 2)
        x = x_2.view(-1, 60)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def get_apnea_model(resnet=False):
    if resnet:
        model = SimpleApneaResNet()
    else:
        model = SimpleApneaNet()
    return model
