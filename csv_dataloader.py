import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data

import numpy as np
import pandas as pd


"""
dataset is raw signal from Piezoelectric Ceramics. 
every sample is 2channel with 10s  200hz
"""


# TODO： BUG:skip 1st sample when load dataset
class CSVDataset(data.Dataset):

    def __init__(self, path, transforms=None):
        # load csv from folder
        all_files = glob.glob(os.path.join(path, "*.csv"))
        df_from_each_file = (pd.read_csv(f, header=None) for f in all_files)
        df = pd.concat(df_from_each_file, ignore_index=True)
        data_np = np.array(df)
        sample_num = len(data_np)
        data_tensor = torch.from_numpy(data_np)
        sig = data_tensor[:, 0:4000]
        sig = sig.view(sample_num, -1, 2000)
        labels = data_tensor[:, 4000:4001]
        self.X = sig.float()
        self.Y = torch.squeeze(labels.long())
        print(self.Y.shape)
        self.transforms = transforms

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.X)


def get_apnea_dataloader(batch_size):
    trained_dataset = CSVDataset('data/big/train/tmp')
    test_dataset = CSVDataset('data/big/test/tmp')

    train_loader = DataLoader(trained_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("len of train_loader : {}, len of data set {}".format(len(train_loader), len(train_loader.dataset)))
    print("len of test_loader : {}, len of data set {}".format(len(test_loader), len(test_loader.dataset)))
    return test_loader, test_loader
