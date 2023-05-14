import torch

from config import *
from model import BaseModel, AttentionModel
from datasets import SimulatedDataset, SimulatedDatasetNII
from loss import finetune_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    with torch.no_grad():
        config = eval_config()

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

        testset = SimulatedDatasetNII('data/ssdl_fn_sim_data', train=False, print_params=False)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )

        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            X, mask = data
            X = X.to(device).float()
            mask = mask.to(device).bool()

            X = mask * X
            Y = mask * model(X)

            loss = finetune_loss(mri=X, fns=Y, mask=mask, trade_off=config.sparse_trade_off)
            running_loss += loss.item()

        print(running_loss / len(testset))
