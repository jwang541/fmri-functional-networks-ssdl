import re


class Config:
    pass


def finetune_config():
    config = Config()
    config.mode = 'finetune'

    # training parameters
    config.n_epochs = 300
    config.batch_size = 1
    config.lr = 1e-4
    config.sparse_trade_off = 10.0
    config.use_pretrained = False

    # model parameters
    config.model_type = 'se'                        # must be 'base' or 'se'
    config.n_time_invariant_features = 16
    config.n_functional_networks = 17

    # IO parameters
    config.output_dir = './out/' + '{}_lr{}_k{}_c{}_sp{}_pre{}_finetune/' \
        .format(config.model_type,
                config.lr,
                config.n_functional_networks,
                config.n_time_invariant_features,
                config.sparse_trade_off,
                config.use_pretrained)
    config.pretrained_weights_file = './out/lr0.0001_k17_c16_pretrain/weights_300.pth'
    config.checkpoint_interval = 10

    return config


def pretrain_config():
    config = Config()
    config.mode = 'pretrain'

    # training parameters
    config.n_epochs = 300
    config.batch_size = 1
    config.lr = 1e-4

    # model parameters
    config.model_type = 'base'                      # must be 'base' or 'se'                                           
    config.n_time_invariant_features = 16
    config.n_functional_networks = 17

    # IO parameters
    config.output_dir = './out/' + '{}_lr{}_k{}_c{}_pretrain/' \
        .format(config.model_type,
                config.lr,
                config.n_functional_networks,
                config.n_time_invariant_features)
    config.checkpoint_interval = 10

    return config


def eval_config():
    config = Config()
    config.mode = 'eval'

    # testing parameters
    config.batch_size = 1

    # IO parameters
    config.weights_file = './out/se_lr0.0001_k17_c16_sp10.0_preFalse_finetune/weights_300.pth'

    # extract model parameters from file name
    pattern = r'([\w-]+)_lr([\d\.]+)_k(\d+)_c(\d+)_sp([\d\.]+)_pre(False|True)_finetune'
    match = re.search(pattern, config.weights_file)
    if match:
        config.model_type = match.group(1)
        config.n_functional_networks = int(match.group(3))
        config.n_time_invariant_features = int(match.group(4))
        config.sparse_trade_off = float(match.group(5))
    else:
        raise Exception('could not parse parameters from file name ' + config.weights_file)

    return config
