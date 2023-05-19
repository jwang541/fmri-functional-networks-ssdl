import os
import torch
import torch.nn as nn

from config import *
from model import BaseModel, AttentionModel
from loss import finetune_loss
from datasets import SimulatedDataset, SimulatedDatasetNII

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    config = finetune_config()
    print('Configuration: ', config.mode)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    trainloader = torch.utils.data.DataLoader(
        SimulatedDatasetNII('data/ssdl_fn_sim_data', train=True, print_params=False),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        SimulatedDatasetNII('data/ssdl_fn_sim_data', train=False, print_params=False),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    if config.model_type == 'base':
        model = BaseModel(k_networks=config.n_functional_networks,
                          c_features=config.n_time_invariant_features)
    elif config.model_type == 'se':
        model = AttentionModel(k_networks=config.n_functional_networks,
                               c_features=config.n_time_invariant_features)
    else:
        raise Exception('config.model_type should be \'base\' or \'se\'')

    if config.use_pretrained:
        model.load_state_dict(torch.load(config.pretrained_weights_file))
        print('Finetuning with: ', config.pretrained_weights_file)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    for epoch in range(config.n_epochs):
        if epoch % config.checkpoint_interval == 0:
            torch.save(model.state_dict(), config.output_dir
                       + 'weights_{}.pt'.format(epoch))

        model.train()
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            X, mask = data
            X = X.float().to(device)
            mask = mask.bool().to(device)

            optimizer.zero_grad()
            X = mask * X
            Y = mask * model(X)

            loss = finetune_loss(mri=X, fns=Y, mask=mask, trade_off=config.sparse_trade_off)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                X, mask = data
                X = X.float().to(device)
                mask = mask.bool().to(device)

                X = X * mask
                Y = model(X) * mask

                loss = finetune_loss(mri=X, fns=Y, mask=mask, trade_off=config.sparse_trade_off)
                test_loss += loss.item()

        print(f'[{epoch + 1}]\t\ttrain loss: {train_loss / len(trainloader.dataset):.3f}'
              f'\t\ttest loss: {test_loss / len(testloader.dataset):.3f}')

    torch.save(model.state_dict(), config.output_dir
               + 'weights_{}.pt'.format(config.n_epochs))
