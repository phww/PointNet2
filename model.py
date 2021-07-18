#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/7/16 上午10:45
# @Author : PH
# @Version：V 0.1
# @File : model.py
# @desc :
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util_module import sample_and_group


class PointNet(nn.Module):
    """PointNet提取分组过后的点云的特征"""

    def __init__(self, in_channel, mlp_channels):
        super(PointNet, self).__init__()
        self.mlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        for channel in mlp_channels:
            self.mlps.append(nn.Conv2d(in_channels=in_channel,
                                       out_channels=channel,
                                       kernel_size=(1, 1),
                                       stride=(1, 1)))
            self.bns.append(nn.BatchNorm2d(channel))
            in_channel = channel

    def forward(self, group_features):
        """
        Args:
            group_features: 分组过后的点云的特征(包括位置信息） shape"B, npoints, group_size, D"

        Returns:
            abstract_features: pointNet提取的特征 shape"B, mpl_channels[-1], group_size"
        """
        group_features = group_features.permute(0, 3, 1, 2).to(dtype=torch.float)  # B, D, npoints, group_size
        for i, mlp in enumerate(self.mlps):
            bn = self.bns[i]
            group_features = F.relu(bn(mlp(group_features)))  # B, mpl_channels[-1], npoints, group_size
        # max pooling
        abstract_features = torch.max(group_features, dim=-1)[0]  # B, mpl_channels[-1], group_size
        abstract_features = abstract_features.permute(0, 2, 1)
        return abstract_features


class PointNetSetAbstraction(nn.Module):
    """sample + group + pointNet"""

    def __init__(self, npoints, group_size, r, in_channel, mlp_channels, group_all=False, rand_seed=None):
        super(PointNetSetAbstraction, self).__init__()
        self.npoints = npoints
        self.group_size = group_size
        self.r = r
        # PointNet
        self.point_net = PointNet(in_channel, mlp_channels)
        self.group_all = group_all
        self.rand_seed = rand_seed

    def forward(self, xyz, features):
        """
        Args:
            xyz: 点云的坐标信息 shape"B, N', 3"
            features: 点云的特征信息（包括坐标） shape"B, N, C" or None

        Returns:
            new_xyz: FPS采样得到的采样点的坐标信息（球心）shape"B, npoints, 3"
            abstract_features: 分组提取的点云特征 shape"B, npoints, mlp_channels[-1]"
        """
        # B, npoints, 3 / B, npoints, group_size 3
        new_xyz, group_features = sample_and_group(xyz, features, self.npoints, self.group_size,
                                                   self.r, self.group_all, self.rand_seed)
        abstract_features = self.point_net(group_features)
        return new_xyz, abstract_features


class PointNetSetAbstractionMsg(nn.Module):
    """Multi Scale Group"""

    def __init__(self, npoints, group_size_list, r_list, in_channel, mlp_channels_list,
                 set_rand_seed=True):
        """
        相当于以同样的采样点为中心（球心），但是以不同的球半径、组内（球内）点云数量、
        mlp结构分别进行球内点云的特征提取。并将不同尺度的特征堆叠起来
        Args:
            npoints: 采样点数
            group_size_list: 不同的组内（球内）点云数量
            r_list: 不同的半径
            in_channel: 上一层提取的特征的通道数+3（坐标），第一层为0 + 3 = 3
            mlp_channels_list: 不同的mlp结构
            set_rand_seed: 设置随机数种子以保证每次FPS产生的采样点是一样的！！！
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoints = npoints
        self.group_size_list = group_size_list
        self.r_list = r_list
        self.inchannel = in_channel
        self.sa_list = nn.ModuleList()
        # 设置随机数种子
        rand_seed = None
        if set_rand_seed:
            rand_seed = torch.random.seed()
        # 以同样的采样点为中心，进行多尺度的特征提取
        for i, mlp_channels in enumerate(mlp_channels_list):
            self.sa_list.append(PointNetSetAbstraction(group_size=group_size_list[i],
                                                       npoints=npoints,
                                                       r=r_list[i],
                                                       in_channel=in_channel + 3,
                                                       mlp_channels=mlp_channels,
                                                       rand_seed=rand_seed))

    def forward(self, xyz, features):
        """
        Args:
            xyz: 点云坐标信息 shape"B, N', 3"
            features: 点云的特征信息（包括坐标）shape"B, N', C" or None

        Returns:
            new_xyz: 采样点的坐标信息 shape"B, npoints, 3"
            abstract_features_msg: 采样点多尺度的特征信息"B, npoints, sum(mlp_channels)"
        """
        abstract_features_msg = []
        for sa in self.sa_list:
            new_xyz, abstract_features = sa(xyz, features)
            # print(new_xyz[0, 0])
            abstract_features_msg.append(abstract_features)
        abstract_features_msg = torch.cat(abstract_features_msg, dim=-1)
        return new_xyz, abstract_features_msg


class PointNet2Cls(nn.Module):
    def __init__(self):
        super(PointNet2Cls, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [16, 32, 128], [0.1, 0.2, 0.4], 0,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [32, 64, 128], [0.2, 0.4, 0.8], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa_all = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.cls_layer = nn.Sequential(nn.Linear(1024, 512),
                                       nn.BatchNorm1d(512),
                                       nn.Dropout(),
                                       nn.Linear(512, 256),
                                       nn.BatchNorm1d(256),
                                       nn.Dropout(),
                                       nn.Linear(256, 128),
                                       nn.BatchNorm1d(128),
                                       nn.Dropout(),
                                       nn.Linear(128, 40),
                                       )

    def forward(self, pc):
        xyz, features = self.sa1(pc, None)
        xyz, features = self.sa2(xyz, features)
        xyz, features = self.sa_all(xyz, features)
        features = self.cls_layer(features.squeeze(dim=1))
        features = F.log_softmax(features, dim=-1)
        return features


if __name__ == '__main__':
    from torchsummary import summary
    pc = torch.randn((10, 500, 3))
    # 测试sa
    sa = PointNetSetAbstraction(100, 10, 1, 3, (128, 64, 64))
    sa2 = PointNetSetAbstraction(10, 5, 1, 67, (128, 64, 64))
    new_xyz, abstract_features = sa(pc, None)
    print(new_xyz.shape)
    print(abstract_features.shape)
    new_xyz, abstract_features = sa2(new_xyz, abstract_features)
    print(new_xyz.shape)
    print(abstract_features.shape)

    # 测试sa-msg
    sa_msg = PointNetSetAbstractionMsg(200, (10, 5, 3), (1, 2, 3), in_channel=0,
                                       mlp_channels_list=[[128, 64, 64], [128, 64, 32], [64, 32, 32]])
    new_xyz, abstract_features_msg = sa_msg(pc, None)
    print(new_xyz.shape)
    print(abstract_features_msg.shape)
    sa_msg2 = PointNetSetAbstractionMsg(100, (10, 5, 3), (1, 2, 3), in_channel=128,
                                        mlp_channels_list=[[128, 64, 64], [128, 64, 32], [64, 32, 32]])
    new_xyz, abstract_features_msg = sa_msg2(new_xyz, abstract_features_msg)
    print(new_xyz.shape)
    print(abstract_features_msg.shape)

    sa_all = PointNetSetAbstraction(None, None, None, 128 + 3, (128, 64, 64), group_all=True)
    new_xyz, abstract_features_msg = sa_all(new_xyz, abstract_features_msg)
    print(new_xyz.shape)
    print(abstract_features_msg.shape)

    # 测试cls
    cls = PointNet2Cls().to('cuda')
    summary(cls, input_size=(2000, 3), batch_size=32, device='cuda')

