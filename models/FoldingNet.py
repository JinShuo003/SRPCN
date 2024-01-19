import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class PCN_encoder(nn.Module):
    def __init__(self, input_num):
        super(PCN_encoder, self).__init__()
        self.i_num = input_num
        self.conv1 = torch.nn.Conv1d(3, 128, 1)  # [in_channels, out_channels, kernel_size]
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(512, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 512, 1)

    def forward(self, x):
        """
         input x is a batch of partial point clouds whose shapes are
          [batch_size, 3(channels), partial_point_cloud_number]
        """
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature_1 = torch.max(x, dim=2, keepdim=True)[0].repeat(1, 1, self.i_num)
        x = torch.cat([x, global_feature_1], dim=1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        feature = torch.max(x, dim=2, keepdim=True)[0]
        return feature


class Folding_layer(nn.Module):
    """
    The folding operation of FoldingNet
    """
    def __init__(self, in_channel: int, out_channels: list):
        super(Folding_layer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        x = torch.cat([grids, codewords], dim=1)
        x = self.layers(x)

        return x


class FoldingNet_decoder(nn.Module):
    """
    Decoder Module of FoldingNet
    """
    def __init__(self, in_channel=512):
        super(FoldingNet_decoder, self).__init__()

        xx = np.linspace(-0.3, 0.3, 46, dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, 46, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)  # (2, 46, 46)

        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 46, 46) -> (2, 46 * 46)

        self.m = self.grid.shape[1]

        self.fold1 = Folding_layer(in_channel + 2, [512, 512, 3])
        self.fold2 = Folding_layer(in_channel + 3, [512, 512, 3])

    def forward(self, x):
        """
        x: (B, C)
        """
        batch_size = x.shape[0]

        grid = self.grid.to(x.device)  # (2, 45 * 45)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)

        x = x.repeat(1, 1, self.m)  # (B, 512, 45 * 45)

        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)

        return recon2


class FoldingNet(nn.Module):
    def __init__(self, input_num):
        super(FoldingNet, self).__init__()
        self.encoder = PCN_encoder(input_num)
        self.decoder = FoldingNet_decoder()

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2).contiguous()
        feat = self.encoder(x)
        pcd = self.decoder(feat).transpose(2, 1).contiguous()
        return pcd
