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


class CSVDataset(data.Dataset):

    def __init__(self, data_tensor, transforms=None):
        sig = data_tensor[:, 0:4000]
        sig = sig.view(-1, 2, 2000)
        labels = data_tensor[:, 4000:4001]
        type_id = data_tensor[:, 4001:4002]
        self.X = sig.float()
        self.Y = torch.squeeze(labels.long())
        self.Z = torch.squeeze(type_id.long())
        print('x:{}, y:{}, z:{}'.format(self.X.shape, self.Y.shape, self.Z.shape))
        self.transforms = transforms

    def __getitem__(self, idx):
        x, y, z = self.X[idx], self.Y[idx], self.Z[idx]
        if self.transforms:
            x = self.transforms(x)
        return x, y, z

    def __len__(self):
        return len(self.X)


def get_tensor(path):
    """
    get train set tensor and test set tensor
    :param path: root path of csv files
    :return:
    """
    # load csv from folder
    pos_files = glob.glob(os.path.join(path, "osa-201807*.csv"))
    neg_files = os.path.join(path, "normal.csv")
    df_from_pos_files = (pd.read_csv(f, header=None) for f in pos_files)
    df_pos = pd.concat(df_from_pos_files, ignore_index=True)
    data_pos_np = np.array(df_pos)
    data_neg_np = np.array(pd.read_csv(neg_files, header=None))
    positive_num = len(data_pos_np)
    negative_num = len(data_neg_np)
    sample_num = min(positive_num, negative_num)
    train_num = round(sample_num * 0.8)
    pos_tensor = torch.from_numpy(data_pos_np)
    neg_tensor = torch.from_numpy(data_neg_np)
    train_pos_tensor = pos_tensor[0:train_num, 0:4002]
    train_neg_tesnor = neg_tensor[0:train_num, 0:4002]
    train_tensor = torch.cat((train_pos_tensor, train_neg_tesnor), 0)
    test_pos_tensor = pos_tensor[train_num:sample_num, 0:4002]
    test_neg_tensor = neg_tensor[train_num:sample_num, 0:4002]
    test_tensor = torch.cat((test_pos_tensor, test_neg_tensor), 0)
    return train_tensor, test_tensor


def get_apnea_dataloader(batch_size):
    train_tensor, test_tensor = get_tensor('D:/project/yymedic/alg/tools/alg_sim/samples/slide/osa')
    train_dataset = CSVDataset(train_tensor)
    test_dataset = CSVDataset(test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("len of train_loader : {}, len of data set {}".format(len(train_loader), len(train_loader.dataset)))
    print("len of test_loader : {}, len of data set {}".format(len(test_loader), len(test_loader.dataset)))
    return train_loader, test_loader

#
# def main():
#     train_loader, test_loader = get_apnea_dataloader(256)
#
#
# main()
# get_tensor('D:/project/yymedic/alg/tools/alg_sim/samples/slide/osa')



