import torch.nn as nn


class ApneaNet6(nn.Module):
    """
    input size: [2*2000]
    """

    def __init__(self, dropout):
        super(ApneaNet6, self).__init__()
        self.bn = nn.BatchNorm1d(2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 20, 50),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 20, 50),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(20, 24, 30),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(24, 24, 30),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(24, 24, 10),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout)
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(24, 12, 10),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout)
        )
        self.fc = nn.Linear(192, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 192)
        x = self.fc(x)
        x = self.softmax(x)
        return x

