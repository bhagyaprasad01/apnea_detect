import torch.nn as nn


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
        conv1   [64, 2, 2000]
        conv2   [64, 20, 975]
        conv3   [64, 20, 926]
        conv4   [64, 24, 897]
        conv5   [64, 24, 868]
        conv6   [64, 24, 859]
        """
        x = self.bn(x)
        conv1_output = self.conv1(x)    # conv1_output = conv2_input = [64, 20, 975]

        # residual net
        conv2_output = self.conv2(conv1_output)
        conv2_input = conv1_output.narrow(2, 0, 926)  # tmp [64, 2, 975]
        conv2_output += conv2_input

        conv3_output = self.conv3(conv2_output)   # in:[64,20,926]
        # conv3_input = conv2_output.narrow(2, 0, 897)
        # print('conv3_input size: ', conv3_input.shape)
        # tmp = torch.zeros(64, 4, 897).cuda()
        # print('tmp size:', tmp.shape)
        # conv3_output += torch.cat((conv3_input, tmp), 1)
        # print('conv3_output size:', conv3_output.shape)

        conv4_output = self.conv4(conv3_output)   # in:[64,24,897]
        conv4_input = conv3_output.narrow(2, 0, 868)
        conv4_output += conv4_input

        conv5_output = self.conv5(conv4_output)   # in:[64,24,868]
        conv5_input = conv4_output.narrow(2, 0, 859)
        conv5_output += conv5_input

        x = self.conv6(conv5_output)   # in:[64:24,859]
        # conv5_input = conv5_output
        x = x.view(-1, 192)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def get_apnea_model():
    model = ApneaNet()
    return model
