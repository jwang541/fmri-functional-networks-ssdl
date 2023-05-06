import torch


def time_courses(X, V):
    return torch.mm(
        torch.mm(X, V.t()),
        torch.pinverse(torch.mm(V, V.t()))
    )


def finetune_loss(mri, fns, mask, trade_off=10.0, eps=1e-8):
    assert (len(mri.shape) == 5)
    assert (len(fns.shape) == 5)
    assert (fns.shape[0] == mri.shape[0])
    assert (fns.shape[2] == mri.shape[2])
    assert (fns.shape[3] == mri.shape[3])
    assert (fns.shape[4] == mri.shape[4])

    loss = 0.0
    for i in range(mri.shape[0]):
        X = torch.reshape(mri[i], (mri.shape[1], -1))
        V = torch.reshape(fns[i], (fns.shape[1], -1))
        M = torch.reshape(mask[i], (-1,))
        # mask = torch.amax(torch.greater(X, 0.0), dim=0)

        X = torch.stack([
            torch.masked_select(X[k], M)
            for k in range(X.shape[0])
        ])

        V = torch.stack([
            torch.masked_select(V[k], M)
            for k in range(V.shape[0])
        ])

        U = time_courses(X, V)

        X_approx = torch.mm(U, V)

        var, mu = torch.var_mean(X)

        hoyer = torch.sum(torch.sum(torch.abs(V), dim=1) / (torch.sqrt(torch.sum(torch.square(V), dim=1) + eps) + eps))

        data_fitting = torch.square(X - X_approx)
        data_fitting = data_fitting / (var + eps)
        data_fitting = torch.sum(data_fitting)

        loss = loss + data_fitting + trade_off * hoyer

    return loss


def pretrain_loss(mri, fns, eps=1e-8):
    spatial_mass = torch.sum(fns, dim=(2, 3, 4))
    spatial_density = torch.einsum('nkxyz, nk -> nkxyz', fns, 1.0 / (spatial_mass + eps))
    TC = torch.einsum('ntxyz, nkxyz -> ntk', mri, spatial_density)
    X_recon = torch.einsum('ntk, nkxyz -> ntxyz', TC, fns)

    recon_error = torch.square(X_recon - mri)
    recon_loss = torch.sum(recon_error)
    return recon_loss
