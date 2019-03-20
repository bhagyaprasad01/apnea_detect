import torch.nn as nn


class SimpleApneaNet(nn.Module):

    def __init__(self, ):
        super(SimpleApneaNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 30, 3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(30, 30, 3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(30, 30, 3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(p=0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(30, 30, 3, stride=3),
            nn.ReLU(),
        )
        self.fc = nn.Linear(90, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 90)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def get_apnea_model():
    model = SimpleApneaNet()
    return model
