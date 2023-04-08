import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

from model import Model
from simulated_dataset import SimulatedFMRIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(actual, expected, trade_off=10):
    assert (len(actual.shape) == 5)
    assert (len(expected.shape) == 5)
    assert (actual.shape[0] == expected.shape[0])
    assert (actual.shape[2] == expected.shape[2])
    assert (actual.shape[3] == expected.shape[3])
    assert (actual.shape[4] == expected.shape[4])

    loss = torch.zeros(actual.shape[0]).to(device)
    for i in range(actual.shape[0]):
        X = torch.reshape(expected[i], (expected.shape[1], -1))
        V = torch.reshape(actual[i], (actual.shape[1], -1))
        U = torch.mm(
            torch.mm(
                X,
                torch.transpose(V, 0, 1)
            ),
            torch.pinverse(
                torch.mm(
                    V,
                    torch.transpose(V, 0, 1)
                )
            )
        )

        X_approx = torch.mm(U, V)

        hoyer = 0.0
        for k in range(actual.shape[1]):
            a = torch.sum(torch.abs(V[k, :]))
            b = torch.sqrt(torch.sum(torch.square(V[k, :])))
            hoyer = hoyer + torch.divide(a, b)

        '''print('X', X)
        print('X approx', X_approx)
        print('V', V)
        print('U', U)
        print(hoyer)'''

        loss[i] = torch.square(torch.norm(X - X_approx)) + trade_off * hoyer

    #print(loss)
    return torch.sum(loss)


if __name__ == '__main__':

    '''for i in range(10):
        A = torch.randn(12, 17, 128, 128, 1).to(device)
        E = torch.randn(12, 120, 128, 128, 1).to(device)
        print(loss_fn(A, E))'''

    model = Model().to(device)

    trainloader = torch.utils.data.DataLoader(
        SimulatedFMRIDataset('./data/simulated-fmri', print_params=True),
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)

    for epoch in range(300):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #print(i)

            X = data.float().to(device)
            optimizer.zero_grad()
            Y = model(X)

            loss = loss_fn(actual=Y, expected=X)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    torch.save(model.state_dict(), './models/weights_0.pth')
