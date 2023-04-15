class Config:
    pass


def train_config():
    config = Config()

    config.batch_size = 1           # batch size
    config.im_size = [56, 72, 56]   # image size of input fMRI data after cropping
    config.im_s = [2, 0, 2]         # start location used to crop image (for saving GPU memory)
    config.im_e = [58, 72, 58]      # end location used to crop image
    config.num_t = 120              # number of time points of fMRI data used during training to compute loss function
                                    # in each iteration. can be changed according to available GPU memory size,
                                    # larger num_t consumes more memory
    config.t_sample_num = 20        # number of time points of fMRI data used during training to compute time-invariant
                                    # feature representation in each iteration. can be changed according to available
                                    # GPU memory size, larger t_sample_num (<=num_t) consumes more memory
    config.mask_val = 0
    config.lr = 1e-4                # learning rate
    config.iteration = 30000        # number of training iterations
    config.step_iter = 30000        # parameter to change the learning rate during training, not used currently
    config.mk = 17                  # number of FNs to be identified

    config.im_preproc = 'vn'        # intensity normalization for input fMRI
    config.smooth_conv = False      # spatially smoothing FNs or not

    # list (.txt) file of the training fMRI images, each row corresponds to the preprocessed fMRI data of one subject
    config.nii_lst = r"/cbica/home/lihon/comp_space/bbl_pnc_resting/hcp_sm_data/hcp_sm6_t400_tra.txt"
    # grey matter (cerebral cortex) mask image
    config.im_mask = r'/cbica/home/lihon/comp_space/bbl_pnc_resting/rnn_autoencoder/scripts/mask_thr0p5_wmparc.2_cc_3mm.nii.gz'

    config.pretrain = False         # set True for pretraining the network
                                    # set False for finetuning the network, a pretrained model is
                                    # required and its path should be set below

    if not config.pretrain:
        config.sparse_lambda = 10   # trade-off parameter for sparsity regularization in loss function
                                    # path to pretrained model
        config.ckpt_dir_finetune = '/cbica/home/lihon/comp_space/bbl_pnc_resting/hcp_res/ckpt_lr0.0001_mk17_sp0.0001_pri0_wSm_False_1pt_3mm_sm6t400_v2_vn_mask0p5_relu'
        config.ckpt_stamp = '29001' # ckpt_stamp for the pretrained model

    return config


def test_config():
    config = Config()

    return config
