import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.linalg as linalg

import math

from model import Model
from simulated_dataset import SimulatedFMRIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

def time_courses(X, V):
    return torch.mm(
        torch.mm(X, V.t()),
        torch.pinverse(torch.mm(V, V.t()))
    )


def finetune_loss(mri, fns, trade_off=10.0, eps=1.0e-5):
    assert (len(mri.shape) == 5)
    assert (len(fns.shape) == 5)
    assert (fns.shape[0] == mri.shape[0])
    assert (fns.shape[2] == mri.shape[2])
    assert (fns.shape[3] == mri.shape[3])
    assert (fns.shape[4] == mri.shape[4])

    loss = 0.0
    for i in range(mri.shape[0]):
        X = torch.reshape(mri[i], (mri.shape[1], -1))
        V = torch.reshape(fns[i], (fns.shape[1], -1))
        mask = torch.amax(torch.greater(X, 0.0), dim=0)

        X = torch.stack([
            torch.masked_select(X[k], mask)
            for k in range(X.shape[0])
        ])

        V = torch.stack([
            torch.masked_select(V[k], mask)
            for k in range(V.shape[0])
        ])

        lstsq = torch.linalg.lstsq(V.t(), X.t())
        U = lstsq.solution.t()

        X_approx = torch.mm(U, V)

        var, mu = torch.var_mean(X)

        hoyer = torch.sum(torch.sum(torch.abs(V), dim=1) / (torch.sqrt(torch.sum(torch.square(V), dim=1) + eps) + eps))

        data_fitting = torch.square(X - X_approx)
        data_fitting = data_fitting / (var + eps)
        data_fitting = torch.sum(data_fitting)

        loss = loss + data_fitting + trade_off * hoyer
    return loss



def pretrain_loss(mri, fns, eps=1e-8):
    TC = torch.empty(size=(
        mri.shape[0],
        mri.shape[1],
        0
    )).to(device)
    # replace for loops with a big einsum
    for k in range(fns.shape[1]):
        spatial_mass = torch.sum(fns[:, k], dim=(1, 2, 3))
        spatial_density = fns[:, k, :, :, :] / (spatial_mass + eps)
        TC_k = torch.einsum('ntxyz, nxyz -> nt', mri, spatial_density)
        TC = torch.cat((TC, TC_k[:, :, None]), dim=2)
    X_recon = torch.empty(size=(
        mri.shape[0],
        0,
        mri.shape[2],
        mri.shape[3],
        mri.shape[4]
    )).to(device)
    for t in range(mri.shape[1]):
        TC_t = TC[:, t, :]
        X_recon_t = torch.einsum('nk, nkxyz -> nxyz', TC_t, fns)
        X_recon = torch.cat((X_recon, X_recon_t[:, None, :, :, :]), dim=1)
    recon_error = torch.square(X_recon - mri)
    recon_loss = torch.sum(recon_error)

    return recon_loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        print()

    model = Model().to(device)

    trainloader = torch.utils.data.DataLoader(
        SimulatedFMRIDataset('data/simtb1', print_params=False),
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)

    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            X = data.float().to(device)

            optimizer.zero_grad()
            Y = model(X)
            #loss = loss_finetune(mri=X, fns=Y)

            loss = finetune_loss(mri=X, fns=Y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    torch.save(model.state_dict(), 'models/weights.pth')
