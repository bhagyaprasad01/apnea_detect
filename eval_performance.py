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
from models import ApneaNet5x1, ApneaNet6
from data.datasets import ApneaTestDataset


batch_size = 256

date = '20180609'
mat_file = 'data/one_day/{}.mat'.format(date)
test_dataset = ApneaTestDataset(mat_file)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print('data len {}'.format(len(test_data_loader.dataset)))

model_dict = torch.load('logs/checkpoint_TestAcc98.215.pth.tar')
model_dict = model_dict['state_dict']
model = ApneaNet6(dropout=0)
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()

res = []
for i, samples in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())

    predict_label = model(samplesV)
    res.extend(torch.max(predict_label, 1)[1].data.tolist())

list_file = open('{}-res.txt'.format(date), 'w')
for label in res:
    list_file.write(str(label))
    list_file.write('\n')
list_file.close()
