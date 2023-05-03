import os
import torch
import torch.nn as nn

from config import *
from model import BaseModel
from loss import finetune_loss, pretrain_loss
from simulated_dataset import SimulatedFMRIDataset
from example_3d_dataset import Example3dDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    # config = pretrain_config()
    config = finetune_config()
    print('Configuration: ', config.mode)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    trainloader = torch.utils.data.DataLoader(
        SimulatedFMRIDataset('data/simtb1', print_params=False),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    # trainloader = torch.utils.data.DataLoader(
    #     Example3dDataset(n_subjects=100),
    #     batch_size=config.batch_size,
    #     shuffle=True,
    #     num_workers=4
    # )
    len_dataset = len(trainloader.dataset)

    model = BaseModel(k_networks=config.n_functional_networks,
                      c_features=config.n_time_invariant_features)
    if config.mode == 'finetune' and config.use_pretrained:
        model.load_state_dict(torch.load(config.pretrained_weights_file))
        print('Pretraining with: ', config.pretrained_weights_file)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    for epoch in range(config.n_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            X = data.float().to(device)

            optimizer.zero_grad()
            Y = model(X)

            if config.mode == 'pretrain':
                loss = pretrain_loss(mri=X, fns=Y)
            elif config.mode == 'finetune':
                loss = finetune_loss(mri=X, fns=Y, trade_off=config.sparse_trade_off)
            else:
                raise Exception('config.mode should be \'pretrain\' or \'finetune\'')

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            if i % len_dataset == len_dataset - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len_dataset:.3f}')
                running_loss = 0.0

        if epoch % config.checkpoint_interval == 0:
            torch.save(model.state_dict(), config.output_dir
                       + 'weights_{}.pth'.format(epoch))

    torch.save(model.state_dict(), config.output_dir
               + 'weights_{}.pth'.format(config.n_epochs))
