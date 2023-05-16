import os
import os.path

import scipy
import nibabel as nib
import torch
from torch.utils.data import Dataset


class SimulatedDatasetNII(Dataset):
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

        self.mask_file = os.path.join(self.dir, 'mask.nii')
        mask_img = nib.load(self.mask_file)
        mask_dat = mask_img.get_fdata()
        self.mask = torch.greater(torch.from_numpy(mask_dat), 0.01)

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
        data_file = os.path.join(self.data_dir, self.filenames[idx])
        nifti_img = nib.load(data_file)
        nifti_dat = nifti_img.get_fdata()

        with torch.no_grad():
            torch_dat = torch.from_numpy(nifti_dat)

            # TODO: currently the dataset reverses the spatial dimensions
            X = torch.permute(torch_dat, (3, 0, 1, 2))

            # std, mu = torch.std_mean(
            #     torch.stack(
            #         [torch.masked_select(
            #             torch.reshape(X[k], (-1,)),
            #             torch.reshape(self.mask, (-1,))
            #         ) for k in range(X.shape[0])]))

            std, mu = torch.std_mean(X, dim=0)

            X = (X - mu) / std * self.mask
            return X, self.mask
