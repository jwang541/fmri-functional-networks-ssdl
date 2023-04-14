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
torch.backends.cudnn.enabled = False

def time_courses(X, V):
    # lstsq = torch.linalg.lstsq(V.t(), X.t())
    # return lstsq.solution.t()
    return torch.mm(
        torch.mm(X, V.t()),
        torch.pinverse(torch.mm(V, V.t()))
    )


def loss_fn(mri, fns, indices, trade_off=10.0, eps=1.0e-5):
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

        print('errors', loss.item(), data_fitting.item(), trade_off * hoyer.item())
    return loss


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

    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            X = data.float().to(device)
            I = torch.greater(X, 200.0)[:, 60, :, :, :]

            optimizer.zero_grad()
            Y = model(X, I)

            loss = loss_fn(mri=X, fns=Y, indices=I)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0



    torch.save(model.state_dict(), 'models/weights.pth')
