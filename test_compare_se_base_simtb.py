import numpy as np
import scipy
import scipy.stats
import torch

from datasets import SimulatedDatasetNII
from loss import finetune_loss
from model import BaseModel, AttentionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Compare the base and scSE attention models on the .nii formatted simtb dataset
if __name__ == '__main__':
    with torch.no_grad():
        base_model = BaseModel(k_networks=20, c_features=20)
        base_model.load_state_dict(
            torch.load('./weights/base_lr0.0001_k20_c20_sp0.1_preFalse_finetune/weights_300.pt'))
        base_model = base_model.float().to(device)
        base_model.eval()

        se_model = AttentionModel(k_networks=20, c_features=20)
        se_model.load_state_dict(
            torch.load('./weights/se_lr0.0001_k20_c20_sp0.1_preFalse_finetune/weights_300.pt'))
        se_model = se_model.float().to(device)
        se_model.eval()

        testset = SimulatedDatasetNII('data/ssdl_fn_sim_data', train=False, print_params=False)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=True,
            num_workers=4
        )

        # Test 1: compare data reconstruction loss on testing dataset
        base_loss, se_loss = 0.0, 0.0
        for i, data in enumerate(testloader, 0):
            X, mask = data
            X = X.to(device).float()
            mask = mask.to(device).bool()

            X = mask * X
            Y_base = mask * base_model(X)
            Y_se = mask * se_model(X)

            base_loss += finetune_loss(mri=X, fns=Y_base, mask=mask, trade_off=0.0).item()
            se_loss += finetune_loss(mri=X, fns=Y_se, mask=mask, trade_off=0.0).item()

        print('- Average data reconstruction loss -')
        print('Base model: ', base_loss / len(testset))
        print('scSE model: ', se_loss / len(testset))
        print()

        # Test 2: compare to simtb ground truth FNs (all testset subjects)
        print('- Ground truth spatial correlation -')
        base_correlations = np.zeros(shape=(20, 20))
        se_correlations = np.zeros(shape=(20, 20))

        # calculate correlations between learned FNs and ground truth FNs,
        # summed over all test subject data
        base_outputs = []
        se_outputs = []
        for i in range(len(testset)):
            # unpack rsfMRI and mask data
            X, mask = testset.__getitem__(i)
            X, mask = X[None], mask[None]
            X = X.float().to(device)
            mask = mask.bool().to(device)
            X = X * mask

            # estimate FNs using base and scSE models
            Y_base = (base_model(X) * mask)[0]
            Y_se = (se_model(X) * mask)[0]
            mask = mask[0]

            # TODO: currently the dataset reverses the spatial dimensions
            Y_base = Y_base.transpose(1, 2)
            Y_se = Y_se.transpose(1, 2)

            # flatten mask outputs across spatial dimensions
            Y_base_mat = torch.reshape(Y_base, (Y_base.shape[0], -1))
            Y_se_mat = torch.reshape(Y_se, (Y_se.shape[0], -1))
            mask_mat = torch.reshape(mask, (-1,))

            # apply mask to spatially flattened outputs
            Y_base_mat_masked = torch.stack([
                torch.masked_select(Y_base_mat[k], mask_mat)
                for k in range(Y_base_mat.shape[0])
            ]).cpu().numpy()
            Y_se_mat_masked = torch.stack([
                torch.masked_select(Y_se_mat[k], mask_mat)
                for k in range(Y_se_mat.shape[0])
            ]).cpu().numpy()

            # store outputs
            base_outputs.append(Y_base_mat_masked)
            se_outputs.append(Y_se_mat_masked)

            # load ground truth FNs
            sim_filename = './data/ssdl_fn_sim_data/data/sim_subject_{0:0=3d}_SIM.mat'.format(i + 81)
            sim_data = scipy.io.loadmat(sim_filename)
            mask_mat_np = mask_mat.cpu().numpy()
            gt_fns_mat_masked = np.stack([
                np.extract(mask_mat_np, sim_data['SM'][k])
                for k in range(sim_data['SM'].shape[0])
            ])

            # update correlation matrix between ground truth and { base and se outputs }
            for a in range(gt_fns_mat_masked.shape[0]):
                for b in range(Y_base_mat_masked.shape[0]):
                    r, _ = scipy.stats.pearsonr(gt_fns_mat_masked[a], Y_base_mat_masked[b])
                    base_correlations[a, b] += r
                for b in range(Y_se_mat_masked.shape[0]):
                    r, _ = scipy.stats.pearsonr(gt_fns_mat_masked[a], Y_se_mat_masked[b])
                    se_correlations[a, b] += r

        # perform linear sum assignment over summed correlations to get group atlas
        base_row_ind, base_col_ind = scipy.optimize.linear_sum_assignment(-1.0 * base_correlations)
        se_row_ind, se_col_ind = scipy.optimize.linear_sum_assignment(-1.0 * se_correlations)

        # calculate and output correlations between learned FNs and group atlas
        for i in range(20):
            # load ground truth FNs
            sim_filename = './data/ssdl_fn_sim_data/data/sim_subject_{0:0=3d}_SIM.mat'.format(i + 81)
            sim_data = scipy.io.loadmat(sim_filename)
            mask_mat_np = mask_mat.cpu().numpy()
            gt_fns_mat_masked = np.stack([
                np.extract(mask_mat_np, sim_data['SM'][k])
                for k in range(sim_data['SM'].shape[0])
            ])

            # get outputs corresponding to the current subject
            base_output = base_outputs[i]
            se_output = se_outputs[i]

            print("Subject {}".format(i + 81))
            sum_base_correlations = 0
            sum_se_correlations = 0
            for j in range(20):
                # calculate correlation between output and current ground truth FN
                base_correlation, _ = scipy.stats.pearsonr(gt_fns_mat_masked[j], base_output[base_col_ind[j]])
                se_correlation, _ = scipy.stats.pearsonr(gt_fns_mat_masked[j], se_output[se_col_ind[j]])

                sum_base_correlations += base_correlation
                sum_se_correlations += se_correlation

                # print correlations and average correlation
                print('FN {}\t\tBASE {} {:.5f}\t\tSE {} {:.5f}'.format(
                    j,
                    base_col_ind[j],
                    base_correlation,
                    se_col_ind[j],
                    se_correlation
                ))
            print('Average\t\tBASE {:.5f}\t\tSE {:.5f}'.format(
                sum_base_correlations / 20.0,
                sum_se_correlations / 20.0
            ))
            print()

        # TODO Test 3: compare to simtb ground truth tcs
        
