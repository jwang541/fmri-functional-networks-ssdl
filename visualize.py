import matplotlib.pyplot as plt
import torch

from config import *
from model import BaseModel, AttentionModel
from datasets import SimulatedDataset, SimulatedDatasetNII
from loss import time_courses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    with torch.no_grad():
        config = eval_config()

        testset = SimulatedDatasetNII('data/ssdl_fn_sim_data', train=False, print_params=False)

        if config.model_type == 'base':
            model = BaseModel(k_networks=config.n_functional_networks,
                              c_features=config.n_time_invariant_features)
        elif config.model_type == 'se':
            model = AttentionModel(k_networks=config.n_functional_networks,
                                   c_features=config.n_time_invariant_features)
        else:
            raise Exception('config.model_type should be \'base\' or \'se\'')
        model.load_state_dict(torch.load(config.weights_file))
        model = model.to(device)
        model.eval()

        # visualize fmri datasets
        mri, mask = testset.__getitem__(13)
        mri = mri.float().to(device)
        mask = mask.bool().to(device)
        fns = (mask * model((mask * mri)[None]))[0, :, :, :, 0]

        X = torch.reshape(mri, (mri.shape[0], -1))
        V = torch.reshape(fns, (fns.shape[0], -1))
        flattened_mask = torch.reshape(mask, (-1,))

        X_nz = torch.stack([
            torch.masked_select(X[k], flattened_mask)
            for k in range(X.shape[0])
        ])
        V_nz = torch.stack([
            torch.masked_select(V[k], flattened_mask)
            for k in range(V.shape[0])
        ])

        U_nz = time_courses(X=X_nz, V=V_nz)
        X_approx_nz = torch.mm(U_nz, V_nz)

        X_approx = torch.zeros(X.shape).to(device)
        for k in range(X_approx.shape[0]):
            X_approx[k][flattened_mask] = X_approx_nz[k]
        X_approx = X_approx.cpu()

        V_learned = torch.zeros(V.shape).to(device)
        for k in range(V.shape[0]):
            V_learned[k][flattened_mask] = V_nz[k]
        V_learned = V_learned.cpu()

        # visualize fmri datasets
        mri_data = (mri * mask)[:, :, :, 0].cpu()
        rows, columns = 4, 5
        fig = plt.figure(figsize=(10, 10))
        for i in range(rows * columns):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(mri_data[i])
            plt.xticks([])
            plt.yticks([])


        # visualize learned FNs of a single subject
        fns_learned_data = torch.reshape(V_learned, fns.shape)
        rows, columns = 4, 5
        fig = plt.figure(figsize=(10, 10))
        for i in range(fns_learned_data.shape[0]):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(fns_learned_data[i])
            plt.xticks([])
            plt.yticks([])



        # visualize reconstructed mri datasets
        mri_approx_data = torch.reshape(X_approx, mri_data.shape)
        rows, columns = 2, 5
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(mri_data[0])
        plt.xticks([])
        plt.yticks([])
        fig.add_subplot(rows, columns, 2)
        plt.imshow(mri_data[20])
        plt.xticks([])
        plt.yticks([])
        fig.add_subplot(rows, columns, 3)
        plt.imshow(mri_data[40])
        plt.xticks([])
        plt.yticks([])
        fig.add_subplot(rows, columns, 4)
        plt.imshow(mri_data[60])
        plt.xticks([])
        plt.yticks([])
        fig.add_subplot(rows, columns, 5)
        plt.imshow(mri_data[80])
        plt.xticks([])
        plt.yticks([])
        fig.add_subplot(rows, columns, 6)
        plt.imshow(mri_approx_data[0])
        plt.xticks([])
        plt.yticks([])
        fig.add_subplot(rows, columns, 7)
        plt.imshow(mri_approx_data[20])
        plt.xticks([])
        plt.yticks([])
        fig.add_subplot(rows, columns, 8)
        plt.imshow(mri_approx_data[40])
        plt.xticks([])
        plt.yticks([])
        fig.add_subplot(rows, columns, 9)
        plt.imshow(mri_approx_data[60])
        plt.xticks([])
        plt.yticks([])
        fig.add_subplot(rows, columns, 10)
        plt.imshow(mri_approx_data[80])
        plt.xticks([])
        plt.yticks([])

        plt.show()
