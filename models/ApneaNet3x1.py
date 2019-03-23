import torch.nn as nn


class ApneaNet5x1(nn.Module):

    def __init__(self, ):
        super(SimpleApneaNet3x1, self).__init__()
        self.bn = nn.BatchNorm1d(5)
        self.conv1 = nn.Sequential(
            nn.Conv1d(5, 32, 3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 32, 3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.5)
        )
        self.fc = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class SimpleApneaResNet3x1(nn.Module):

    def __init__(self, ):
        super(SimpleApneaResNet3x1, self).__init__()
        self.bn = nn.BatchNorm1d(5)
        self.conv1 = nn.Sequential(
            nn.Conv1d(5, 32, 3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 32, 3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.5)
        )
        self.fc = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def get_apnea_model(resnet=False):
    if resnet:
        model = SimpleApneaResNet3x1()
    else:
        model = SimpleApneaNet3x1()
    return model
