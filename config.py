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
    config.n_time_invariant_features = 16
    config.n_functional_networks = 17

    # IO parameters
    config.output_dir = './out/' + 'lr{}_k{}_c{}_sp{}_pre{}_finetune/'.format(config.lr,
                                                                              config.n_functional_networks,
                                                                              config.n_time_invariant_features,
                                                                              config.sparse_trade_off,
                                                                              config.use_pretrained)
    config.pretrained_weights_file = None
    config.checkpoint_interval = 10

    return config


def pretrain_config():
    config = Config()
    config.mode = 'pretrain'

    # training parameters
    config.n_epochs = 30
    config.batch_size = 1
    config.lr = 1e-4

    # model parameters
    config.n_time_invariant_features = 16
    config.n_functional_networks = 17

    # IO parameters
    config.output_dir = './out/' + 'lr{}_k{}_c{}_pretrain/'.format(config.lr,
                                                                   config.n_functional_networks,
                                                                   config.n_time_invariant_features)
    config.checkpoint_interval = 10

    return config


def eval_config():
    config = Config()
    config.mode = 'eval'

    # testing parameters
    config.batch_size = 1
    config.sparse_trade_off = 10.0

    # model parameters
    config.n_time_invariant_features = 16
    config.n_functional_networks = 17

    # IO parameters
    config.weights_file = './models/w100.pth'

    return config
