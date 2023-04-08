import torch
import torch.nn as nn
import torch.nn.functional as F


# maps a N x T x D x H x W tensor --> N x K x D x H x W tensor
class Model(nn.Module):
    def __init__(self, c_features=16, k_networks=17, eps_normalization=1e-9):
        super().__init__()

        self.eps_normalization = eps_normalization

        # batch normalization
        self.batch_normalization = nn.BatchNorm3d(1)

        # time-invariant representation learning
        self.representation_conv1 = nn.Conv3d(1, c_features, kernel_size=3, stride=1, padding=1)

        # functional network learning
        self.functional_bn1 = nn.BatchNorm3d(c_features)
        self.functional_conv1 = nn.Conv3d(c_features, 16, kernel_size=3, stride=1, padding=1)
        self.functional_bn2 = nn.BatchNorm3d(16)
        self.functional_conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.functional_bn3 = nn.BatchNorm3d(32)
        self.functional_conv3 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
        self.functional_bn4 = nn.BatchNorm3d(32)
        self.functional_conv4 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)

        self.functional_bn5 = nn.BatchNorm3d(32)
        self.functional_deconv1 = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.functional_bn6 = nn.BatchNorm3d(64)
        self.functional_deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.functional_bn7 = nn.BatchNorm3d(64)
        self.functional_deconv3 = nn.ConvTranspose3d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.functional_bn8 = nn.BatchNorm3d(32)
        self.functional_conv5 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.functional_bn9 = nn.BatchNorm3d(16)
        self.functional_conv6 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)

        self.functional_bn10 = nn.BatchNorm3d(16)
        self.functional_conv7 = nn.Conv3d(16, k_networks, kernel_size=3, stride=1, padding=1)

        '''self.functional_bn1 = nn.Identity()
        self.functional_bn2 = nn.Identity()
        self.functional_bn3 = nn.Identity()
        self.functional_bn4 = nn.Identity()
        self.functional_bn5 = nn.Identity()
        self.functional_bn6 = nn.Identity()
        self.functional_bn7 = nn.Identity()
        self.functional_bn8 = nn.Identity()
        self.functional_bn9 = nn.Identity()
        self.functional_bn10 = nn.Identity()'''

    def forward(self, x):
        # time-invariant representation learning module
        x = torch.stack([
            self.representation_conv1(x[:, None, i]) for i in range(x.shape[0])
        ], 0)
        x = torch.mean(x, 0)

        # functional network learning module
        x0 = self.functional_bn1(x)
        x1 = self.functional_bn2(F.leaky_relu(self.functional_conv1(x0)))
        x2 = self.functional_bn3(F.leaky_relu(self.functional_conv2(x1)))
        x3 = self.functional_bn4(F.leaky_relu(self.functional_conv3(x2)))
        x = self.functional_bn5(F.leaky_relu(self.functional_conv4(x3)))

        x = self.functional_bn6(F.leaky_relu(torch.cat(
            (self.functional_deconv1(x, x3.shape), x3), 1)))
        x = self.functional_bn7(F.leaky_relu(torch.cat(
            (self.functional_deconv2(x, x2.shape), x2), 1)))
        x = self.functional_bn8(F.leaky_relu(torch.cat(
            (self.functional_deconv3(x, x1.shape), x1), 1)))

        x = self.functional_bn9(F.leaky_relu(self.functional_conv5(x)))
        x = self.functional_bn10(F.leaky_relu(self.functional_conv6(x)))

        x = F.relu(self.functional_conv7(x))

        # rescale so maximum of each sample is 1
        component_max = torch.amax(x, dim=(2, 3, 4))

        x = torch.stack([
            torch.stack([
                x[i, j] / (component_max[i, j] + self.eps_normalization)
                for j in range(component_max.shape[1])
            ]) for i in range(component_max.shape[0])
        ])

        x = torch.clamp_max(x, 1.0)

        return x


if __name__ == '__main__':
    # batch size 12, 120 time points, 128 x 128 x 1 image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''I = torch.randn(12, 120, 128, 128, 1).to(device)
    m = Model().to(device)
    out = m(I)
    print(out.shape)
    out[0, 0, 0, 0, 0].backward()'''

    with torch.no_grad():
        for i in range(100):
            I = torch.randn(1, 120, 128, 128, 1).to(device)

            m = Model().to(device)
            out = m(I)

            print(out.shape)
            print(out)
