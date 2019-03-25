import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import scipy.signal as signal
from models import ApneaNet5x1


batch_size = 256

mat_data = sio.loadmat('data/2x50/test-07')
test_data = mat_data['data']
test_label = mat_data['label']

num_test_instances = len(test_data)

test_data = signal.resample_poly(test_data, 200, 50)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
test_data = test_data.view(num_test_instances, 2, 2000)
test_label = torch.squeeze(test_label)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model_dict = torch.load('logs/checkpoint.pth.tar')
model_dict = model_dict['state_dict']
model = ApneaNet5x1()
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()

correct_test = 0
for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())
        # labelsV = labelsV.view(-1)

    predict_label = model(samplesV)
    correct_test += (torch.max(predict_label, 1)[1].view(labelsV.size()).data == labelsV.data).sum()
    # prediction = predict_label[0].data.max(1)[1]
    # prediction = torch.max(outputs, 1)[1].view(y.size()).data
    # correct_test += prediction.eq(labelsV.data.long()).sum()

    # if i == 0:
    #     batch_prediction= prediction
    #     batch_featuremap = predict_label[1].data
    #     fault_prediction = batch_prediction
    #     featuremap = batch_featuremap
    #
    # elif i > 0:
    #     batch_prediction = prediction
    #     batch_featuremap = predict_label[1].data
    #
    #     fault_prediction = np.concatenate((fault_prediction, batch_prediction), axis=0)
    #     featuremap = np.concatenate((featuremap, batch_featuremap), axis=0)

print("Test accuracy:", (100 * float(correct_test) / num_test_instances))

