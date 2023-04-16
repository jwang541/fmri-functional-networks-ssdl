import torch

from model import Model
from simulated_dataset import SimulatedFMRIDataset

from train import finetune_loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    with torch.no_grad():
        model = Model()
        model.load_state_dict(torch.load('models/weights_100.pth'))
        model = model.to(device)

        testset = SimulatedFMRIDataset('data/simtb2', print_params=True)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=True,
            num_workers=4
        )

        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            X = data.float().to(device)
            I = torch.greater(X, 200.0)
            Y = model(X)

            loss = finetune_loss(mri=X, fns=Y, indices=I)

            print(i, loss.item(), X.shape)

            running_loss += loss.item()

        print(running_loss / 100)

        data = testset.__getitem__(0)




        #M_0 = Model().to(device)
        #M_0.load_state_dict(torch.load('models/weights_0.pth'))
        #M_1 = Model().to(device)
        #M_1.load_state_dict(torch.load('models/weights_1.pth'))
        M_100 = Model().to(device)
        M_100.load_state_dict(torch.load('models/weights_100.pth'))
        #M_200 = Model().to(device)
        #M_200.load_state_dict(torch.load('models/weights_200.pth'))

        MRI = testset.__getitem__(0).float().to(device)
        I = torch.greater(MRI, 200.0)

        print(torch.sum(I.long()))
        print(I.shape)

        print('-----------------------')
        #print('M_0', loss_fn(mri=MRI[None], fns=M_0(MRI[None])))
        #print('M_1', loss_fn(mri=MRI[None], fns=M_1(MRI[None])))
        print('M_100', finetune_loss(mri=MRI[None], fns=M_100(MRI[None]), indices=I[None]).item())
        #print('M_200', loss_fn(mri=MRI[None], fns=M_200(MRI[None])))
        print('-----------------------')


