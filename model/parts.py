import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


# Implements the scSE (spatial and channel Squeeze-and-Excitation) block proposed in "Recalibrating Fully
# Convolutional Networks with Spatial and Channel ‘Squeeze & Excitation’ Blocks" (Roy et al.) for an input
# with 3 spatial dimensions.
class SCSqueezeExcitation3D(nn.Module):
    def __init__(self, in_channels, channel_neurons):
        super().__init__()
        self.in_channels = in_channels

        self.channel_se = nn.Sequential(
            nn.Linear(in_channels, channel_neurons, bias=True),
            nn.ReLU(),
            nn.Linear(channel_neurons, in_channels, bias=True),
            nn.Sigmoid()
        )
        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, X):
        Z_channel = torch.mean(X, dim=(2, 3, 4))
        S_channel = self.channel_se(Z_channel)
        S_spatial = self.spatial_se(X)

        X_channel = torch.einsum('ncxyz, nc -> ncxyz', X, S_channel)
        X_spatial = torch.mul(X, S_spatial)

        return torch.max(X_channel, X_spatial)
