import torch
from torch import nn
import torch.nn.functional as F


class mlp(nn.Module):
    def __init__(self, in_num, out_num):
        super(mlp, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_num, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, out_num, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class final_mlp(nn.Module):
    def __init__(self, in_num, out_num):
        super(final_mlp, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_num, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, out_num, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


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


class TopNet_decoder(nn.Module):
    def __init__(self, arch=[4, 8, 8, 8]):
        super(TopNet_decoder, self).__init__()
        self.arch = arch
        self.level_num = len(arch)
        self.level_1 = nn.ModuleList()
        self.level_2 = nn.ModuleList()
        self.level_3 = nn.ModuleList()
        self.level_4 = nn.ModuleList()
        # construct tree
        for _ in range(arch[0]):
            self.level_1.append(mlp(512, 8))
        for _ in range(arch[1]):
            self.level_2.append(mlp(512 + 8, 8))
        for _ in range(arch[2]):
            self.level_3.append(mlp(512 + 8, 8))
        for _ in range(arch[3]):
            self.level_4.append(final_mlp(512 + 8, 3))

    def forward(self, x):
        features_1 = []
        for net in self.level_1:
            features_1.append(torch.cat([net(x), x], dim=1))
        features_2 = []
        for feat in features_1:
            for net in self.level_2:
                features_2.append(torch.cat([net(feat), x], dim=1))
        features_3 = []
        for feat in features_2:
            for net in self.level_3:
                features_3.append(torch.cat([net(feat), x], dim=1))
        features_4 = []
        for feat in features_3:
            for net in self.level_4:
                features_4.append(net(feat))
        pcd = torch.cat(features_4, dim=2)
        return pcd


class TopNet(nn.Module):
    def __init__(self, input_num):
        super(TopNet, self).__init__()
        self.encoder = PCN_encoder(input_num)
        self.decoder = TopNet_decoder()

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2).contiguous()
        feat = self.encoder(x)
        pcd = self.decoder(feat)
        return pcd.transpose(1, 2).contiguous()
