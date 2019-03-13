import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import pandas as pd
import scipy.signal as signal


"""
dataset is raw signal from Piezoelectric Ceramics. 
every sample is 10s with 50hz
"""


# TODO： BUG:skip 1st sample when load dataset
class CSVDataset(data.Dataset):

    def __init__(self,path, transforms=None):
        df = pd.read_csv(path)
        x, y = list(), list()
        for index, row in df.iterrows():
            # from 50hz resample to 200hz
            sig = row[:-1]
            sig = signal.resample_poly(sig, 200, 50)
            # add one dim
            sig = torch.Tensor(sig)
            sig = sig.unsqueeze(0)
            x.append(sig)
            y.append(int(row[-1]))
        self.X = torch.stack(x)
        self.Y = torch.LongTensor(y)
        self.transforms = transforms

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.X)


def get_apnea_dataloader(batch_size):
    trained_dataset = CSVDataset('data/train.csv')
    test_dataset = CSVDataset('data/test.csv')

    train_loader = DataLoader(trained_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("len of train_loader : ", len(train_loader))
    print("len of test_loader : ", len(test_loader))

    return train_loader, test_loader
