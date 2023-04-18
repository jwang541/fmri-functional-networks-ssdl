import torch

from config import *
from model import Model
from simulated_dataset import SimulatedFMRIDataset
from loss import finetune_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    with torch.no_grad():
        config = eval_config()

        model = Model()
        model.load_state_dict(torch.load('out/lr0.0001_k17_c16_sp10.0_preFalse_finetune/weights_300.pth'))
        model = model.to(device)
        model.eval()

        testset = SimulatedFMRIDataset('data/simtb2', print_params=False)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )

        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            X = data.float().to(device)
            Y = model(X)
            loss = finetune_loss(mri=X, fns=Y, trade_off=config.sparse_trade_off)
            running_loss += loss.item()

        print(running_loss / len(testset))
