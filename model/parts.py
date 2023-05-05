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

        self.channel_fc1 = nn.Linear(in_channels, channel_neurons)
        self.channel_fc2 = nn.Linear(channel_neurons, in_channels)

        self.spatial_conv = nn.Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, X):
        assert (X.shape[1] == self.in_channels)

        Z_channel = torch.mean(X, (2, 3, 4))
        S_channel = F.sigmoid(self.channel_fc2(F.relu(self.channel_fc1(Z_channel))))
        S_spatial = F.sigmoid(self.spatial_conv(X))

        S_channel = S_channel[:, :, None, None, None].repeat(1, 1, X.shape[2], X.shape[3], X.shape[4])
        S_spatial = S_spatial.repeat(1, X.shape[1], 1, 1, 1)

        S_combine = torch.maximum(S_channel, S_spatial)
        return X * S_combine
