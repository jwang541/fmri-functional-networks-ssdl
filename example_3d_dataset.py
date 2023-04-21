import os
import os.path

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset



class Example3dDataset(Dataset):
    def __init__(self, n_subjects=30, eps=1e-8):
        self.fns = torch.randn(20, 32 ** 3)
        self.tcs = torch.randn(n_subjects, 120, 20)
        self.data = torch.reshape(torch.matmul(self.tcs, self.fns), (n_subjects, 120, 32, 32, 32))
        self.eps = eps

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        with torch.no_grad():
            X = self.data[idx]
            var, mu = torch.var_mean(X, dim=(0,))
            X = (X - mu) / torch.sqrt(var + self.eps)
            return X


if __name__ == '__main__':
    trainset = Example3dDataset(n_subjects=30)
    testset = Example3dDataset(n_subjects=10)
