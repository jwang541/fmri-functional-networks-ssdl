import os
import os.path

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class SimulatedFMRIDataset(Dataset):
    def __init__(self, dir, print_params=False):
        self.dir = dir
        self.data_dir = os.path.join(self.dir, 'data')
        self.len = 0
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.mat'):
                self.len += 1
        params = scipy.io.loadmat('./data/simulated-fmri/params.mat')
        self.n_subjects = params['sP'][0][0][0][0][0]
        self.n_components = params['sP'][0][0][1][0][0]
        self.fmri_size = params['sP'][0][0][2][0][0]
        self.n_time_points = params['sP'][0][0][3][0][0]

        if print_params:
            print('# subjects:', self.n_subjects)
            print('# components:', self.n_components)
            print('fmri size:', self.fmri_size)
            print('# time points:', self.n_time_points)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mat_path = os.path.join(self.data_dir, str(idx))
        mat = scipy.io.loadmat(mat_path)
        data = np.array(mat['D'])
        torch_data = torch.from_numpy(data)
        reshaped_data = torch.reshape(torch_data, (self.n_time_points, self.fmri_size, self.fmri_size, 1))
        return reshaped_data


if __name__ == '__main__':
    dataset = SimulatedFMRIDataset('./data/simulated-fmri', print_params=True)
    print('sample point shape: ', dataset.__getitem__(0).shape)
