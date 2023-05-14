import os
import os.path

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class SimulatedDataset(Dataset):
    def __init__(self, dir, train=True, eps=1e-8, print_params=False):
        self.dir = dir
        self.data_dir = os.path.join(self.dir, 'data')
        if train:
            self.filenames_file = os.path.join(self.dir, 'train.txt')
        else:
            self.filenames_file = os.path.join(self.dir, 'test.txt')
        self.filenames = []
        with open(self.filenames_file, 'r') as f:
            for line in f:
                if len(line.strip()) != 0:
                    self.filenames.append(line.strip())
        self.params_file = os.path.join(self.dir, 'params.mat')
        params = scipy.io.loadmat(self.params_file)

        self.n_components = params['sP'][0][0][1][0][0]
        self.fmri_size = params['sP'][0][0][2][0][0]
        self.n_time_points = params['sP'][0][0][3][0][0]
        self.eps = eps

        if print_params:
            print('# subjects:', len(self.filenames))
            print('# components:', self.n_components)
            print('fmri size:', self.fmri_size)
            print('# time points:', self.n_time_points)
            print('point shape:', self.__getitem__(0)[0].shape)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        mat_path = os.path.join(self.data_dir, self.filenames[idx])
        mat = scipy.io.loadmat(mat_path)
        data = np.array(mat['D'])

        with torch.no_grad():
            torch_data = torch.from_numpy(data)
            X = torch.reshape(torch_data, (self.n_time_points, self.fmri_size, self.fmri_size, 1))

            mask = torch.greater(X, 200.0)[0, :, :, :]
            std, mu = torch.std_mean(
                torch.masked_select(
                    torch.reshape(X, (X.shape[0], -1)),
                    torch.reshape(mask, (-1,))
                ))
            X = (X - mu) / std * mask
            return X, mask
