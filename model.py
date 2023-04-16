import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class FeatureExtractor(nn.Module):
    def __init__(self, c_features, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.conv = nn.Conv3d(1, c_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(c_features, affine=True)

        nn.init.trunc_normal_(self.conv.weight, std=0.01, a=-0.02, b=0.02)

    def forward(self, X):
        # Temporally separated convolution
        X = F.leaky_relu(torch.stack([
            self.conv(X[:, i, None, :, :, :])
            for i in range(X.shape[1])
        ], 0))

        # Average pooling along temporal axis
        X = torch.mean(X, 0)
        X = self.norm(X)

        return X


class UNet(nn.Module):
    def __init__(self, c_features, eps=1e-9):
        super().__init__()
        self.eps = eps

        self.conv1 = nn.Conv3d(c_features, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv5 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv6 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv7 = nn.Conv3d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)

        nn.init.trunc_normal_(self.conv1.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv2.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv3.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv4.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.deconv5.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.deconv6.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.deconv7.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv8.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.conv9.weight, std=0.01, a=-0.02, b=0.02)

        self.norm1 = nn.InstanceNorm3d(16, affine=True)
        self.norm2 = nn.InstanceNorm3d(32, affine=True)
        self.norm3 = nn.InstanceNorm3d(32, affine=True)
        self.norm4 = nn.InstanceNorm3d(32, affine=True)
        self.norm5 = nn.InstanceNorm3d(32, affine=True)
        self.norm6 = nn.InstanceNorm3d(32, affine=True)
        self.norm7 = nn.InstanceNorm3d(16, affine=True)
        self.norm8 = nn.InstanceNorm3d(16, affine=True)
        self.norm9 = nn.InstanceNorm3d(16, affine=True)

    def forward(self, X):
        X1 = self.norm1(F.leaky_relu(self.conv1(X), negative_slope=0.01))
        X2 = self.norm2(F.leaky_relu(self.conv2(X1), negative_slope=0.01))
        X3 = self.norm3(F.leaky_relu(self.conv3(X2), negative_slope=0.01))
        X4 = self.norm4(F.leaky_relu(self.conv4(X3), negative_slope=0.01))
        X5 = self.norm5(F.leaky_relu(self.deconv5(X4), negative_slope=0.01))
        X5 = torch.cat([F.interpolate(X5, X3.shape[2:5]), X3], 1)
        X6 = self.norm6(F.leaky_relu(self.deconv6(X5), negative_slope=0.01))
        X6 = torch.cat([F.interpolate(X6, X2.shape[2:5]), X2], 1)
        X7 = self.norm7(F.leaky_relu(self.deconv7(X6), negative_slope=0.01))
        X7 = torch.cat([F.interpolate(X7, X1.shape[2:5]), X1], 1)
        X8 = self.norm8(F.leaky_relu(self.conv8(X7), negative_slope=0.01))
        X9 = self.norm9(F.leaky_relu(self.conv9(X8), negative_slope=0.01))
        return X9


class Output(nn.Module):
    def __init__(self, k_networks, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.conv = nn.Conv3d(16, k_networks, kernel_size=3, stride=1, padding=1)
        nn.init.trunc_normal_(self.conv.weight, std=0.01, a=-0.02, b=0.02)

    def forward(self, X):
        X = F.relu(self.conv(X))
        component_max = torch.amax(X, dim=(2, 3, 4))
        X = torch.einsum('nkxyz, nk -> nkxyz',
                         X,
                         1.0 / (component_max + self.eps))
        return X


# maps a N x T x D x H x W tensor --> N x K x D x H x W tensor
class Model(nn.Module):
    def __init__(self, c_features=16, k_networks=17, eps=1e-9):
        super().__init__()

        self.eps = eps

        self.feature_extractor = FeatureExtractor(c_features)
        self.unet = UNet(c_features)
        self.output = Output(k_networks)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.unet(x)
        x = self.output(x)

        return x
