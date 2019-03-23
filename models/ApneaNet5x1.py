import torch.nn as nn


class ApneaNet5x1(nn.Module):
    """
    input size: [2 * 2000]
    """

    def __init__(self, dropout=0):
        super(ApneaNet5x1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 30, 5, stride=5),
            # nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(30, 30, 5, stride=5),
            # nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(30, 30, 5, stride=5),
            # nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=dropout)
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
