from torch import nn
import torch
import torch.nn.functional as F


# -------------------------------------Encoder-----------------------------------
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
        global_feature_2 = torch.max(x, dim=2, keepdim=True)[0]  # [batch_size, 1024, 1]
        return global_feature_2


# class Feature_transfrom(nn.Module):
#     def __init__(self):
#         super(Feature_transfrom, self).__init__()
#         self.conv1 = torch.nn.Conv1d(1536, 1024, 1)  # [in_channels, out_channels, kernel_size]
#         self.conv2 = torch.nn.Conv1d(1024, 512, 1)
#         self.conv3 = torch.nn.Conv1d(2048, 1024, 1)
#         self.conv4 = torch.nn.Conv1d(1024, 512, 1)
#
#     def forward(self, feature):
#         """
#          input feature is a batch of feature whose shapes are
#           [batch_size, dim_input, 1]
#          after the network, the shapes are
#           [batch, dim_output, 1]
#         """
#         x = F.relu(self.conv1(feature))
#         x = F.relu(self.conv2(x))
#         x = torch.cat([x, feature], dim=1)
#         x = F.relu(self.conv3(x))
#         global_feature = self.conv4(x)
#         return global_feature


class Feature_transfrom(nn.Module):
    def __init__(self):
        super(Feature_transfrom, self).__init__()
        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)  # [in_channels, out_channels, kernel_size]
        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3 = torch.nn.Conv1d(1536, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1024, 512, 1)

    def forward(self, feature):
        """
         input feature is a batch of feature whose shapes are
          [batch_size, dim_input, 1]
         after the network, the shapes are
          [batch, dim_output, 1]
        """
        x = F.relu(self.conv1(feature))
        x = F.relu(self.conv2(x))
        x = torch.cat([x, feature], dim=1)
        x = F.relu(self.conv3(x))
        global_feature = self.conv4(x)
        return global_feature


# -------------------------------------Decoder-----------------------------------
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


class TopNet_decoder(nn.Module):
    def __init__(self, arch=[4, 4, 8, 8]):
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
        pc = torch.cat(features_4, dim=2)
        return pc


# -------------------------------------Completion Net-----------------------------------
# class PCDCompletionNet(nn.Module):
#     def __init__(self, pcd_point_num):
#         super().__init__()
#         # encoder
#         self.encoder_pcd1 = PCN_encoder(pcd_point_num)
#         self.encoder_pcd2 = PCN_encoder(pcd_point_num)
#         self.encoder_IBS = PCN_encoder(pcd_point_num)
#         # feature transform
#         self.feature_transform = Feature_transfrom()
#         # decoder
#         self.decoder_path1_pcd1 = TopNet_decoder()
#         self.decoder_path1_pcd2 = TopNet_decoder()
#         self.decoder_path2_pcd1 = TopNet_decoder()
#         self.decoder_path2_pcd2 = TopNet_decoder()
#
#     def forward(self, pcd1_partial, pcd2_partial, IBS):
#         # exchange dim1 and dim2
#         pcd1_partial = pcd1_partial.permute(0, 2, 1)
#         pcd2_partial = pcd2_partial.permute(0, 2, 1)
#         IBS = IBS.permute(0, 2, 1)
#
#         # get the feature of pcd1, pcd2, IBS
#         feature_pcd1 = self.encoder_pcd1(pcd1_partial)
#         feature_pcd2 = self.encoder_pcd2(pcd2_partial)
#         feature_IBS = self.encoder_IBS(IBS)
#
#         # concat the global feature
#         feature_global = torch.cat((feature_pcd1, feature_pcd2, feature_IBS), dim=1)
#
#         # transform feature
#         feature_global = self.feature_transform(feature_global)
#
#         # decode
#         pcd1_path1 = self.decoder_path1_pcd1(feature_global).permute(0, 2, 1)
#         pcd2_path1 = self.decoder_path1_pcd2(feature_global).permute(0, 2, 1)
#         pcd1_path2 = self.decoder_path2_pcd1(feature_global).permute(0, 2, 1)
#         pcd2_path2 = self.decoder_path2_pcd2(feature_global).permute(0, 2, 1)
#         return pcd1_path1, pcd2_path1, pcd1_path2, pcd2_path2


class PCDCompletionNet(nn.Module):
    def __init__(self, pcd_point_num):
        super().__init__()
        # encoder
        self.encoder_pcd = PCN_encoder(pcd_point_num)
        self.encoder_IBS = PCN_encoder(pcd_point_num)
        # feature transform
        self.feature_transform = Feature_transfrom()
        # decoder
        self.decoder = TopNet_decoder()

    def forward(self, pcd_partial, IBS):
        # exchange dim1 and dim2
        pcd_partial = pcd_partial.permute(0, 2, 1)
        IBS = IBS.permute(0, 2, 1)

        # get the feature of pcd1, pcd2, IBS
        feature_pcd = self.encoder_pcd(pcd_partial)
        feature_IBS = self.encoder_IBS(IBS)

        # concat the global feature
        feature_global = torch.cat((feature_pcd, feature_IBS), dim=1)

        # transform feature
        feature_global = self.feature_transform(feature_global)

        # decode
        pcd_out = self.decoder_path1_pcd1(feature_global).permute(0, 2, 1)

        return pcd_out
