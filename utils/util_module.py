#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/7/14 下午3:30
# @Author : PH
# @Version：V 0.1
# @File : util_module.py
# @desc :
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch


def farthest_points_sampling(pc, npoints, random_seed=None):
    """
    最远点采样
    Args:
        pc: 点云 shape"B, N, 3"
        npoints: 采样点数

    Returns:
        centroids: 采样点的下标 shape"B, npoints"
    """
    if random_seed is not None:
        torch.random.manual_seed(random_seed)
    B, N, _ = pc.shape
    device = pc.device
    # 初始化保存数据的变量
    centroids = torch.zeros(B, npoints, dtype=torch.long).to(device)
    min_dists = torch.ones(B, N).to(device, dtype=torch.float) * 1e6
    # 随机选取第一个点
    farthest_idx = torch.randint(N, (B,), dtype=torch.long).to(device)  # B,
    batch_idx = torch.arange(B, dtype=torch.long).to(device)
    # FPS
    for i in range(npoints):
        centroids[:, i] = farthest_idx
        farthest_point = pc[batch_idx, farthest_idx]  # B, 3
        dists = torch.sum((pc - farthest_point.unsqueeze(dim=1)) ** 2, dim=-1).to(dtype=torch.float)  # B, N
        # 先取候选点集中到采样点集每个点中最近的点
        mask = dists < min_dists
        min_dists[mask] = dists[mask]
        # 再选择这些最近点中距离最大的点，该点就是距离采样点集最远的点
        farthest_idx = torch.max(min_dists, dim=-1)[1]
    return centroids


def ids2points(ids, pc):
    """
    将下标转换为对应的点云，或点云组
    Args:
        ids: shape "B, D1, D2, D3, ..., Dn"
        pc: shape "B, N, C"

    Returns:
        output: shape "B, D1, D2, ..., Dn, 3"
    """
    B, N, _ = pc.shape
    device = pc.device
    shape = ids.shape[1:]
    # batch_idx shape "B, D1, D2, ..., Dn", 首先生成形状为B，的原始batch_idx
    batch_idx = torch.arange(B, dtype=torch.long).to(device)
    # 然后扩展维度到B，1，1，1..,1
    for i in range(1, len(shape) + 1):
        batch_idx = batch_idx.unsqueeze(dim=i)
    # 最后后在维度上repeat 对应的D1, D2, ..., Dn
    batch_idx = batch_idx.repeat(1, *shape)
    output = pc[batch_idx, ids]
    return output


def square_distance(src, dst):
    """
    使用以下公式计算3维的欧氏距离
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2 =
    sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1)  # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # xm*xm + ym*ym + zm*zm
    return dist


def ball_quire(pc, centroids, group_size, r):
    """
    查询以r为半径，采样点为球心的球内距离球心最近的group_size个点
    Args:
        pc: 原始点云 shape"B, N, 3"
        centroids: FPS采样点的下标 shape"B, S, 3"

    Returns:
        group_idx: 分组过后的下标 shape"B, S, G"
    """
    B, N, _ = pc.shape
    device = pc.device
    _, S = centroids.shape
    # idx转换为点坐标
    centroids_point = ids2points(centroids, pc).to(device)  # B, S, 3
    dists = square_distance(centroids_point, pc).to(device)  # B, S, N
    sorted_dists, sorted_idx = dists.sort(dim=-1)   # B, S, N
    # 先将不在球内的点的下标设置为N-1
    sorted_idx[sorted_dists > r ** 2] = N - 1
    # 在选择距离球心最近的max_point个点
    group_idx = sorted_idx[:, :, :group_size]  # B, S, G
    # 但是有可能距离球心最近的max_point个点没有在球内（下标被设为N-1的点），因此用最近的点代替他们
    mask = group_idx == N - 1
    nearest_idx = sorted_idx[:, :, 0].view(B, S, 1).repeat([1, 1, group_size])
    group_idx[mask] = nearest_idx[mask]
    return group_idx


def sample_and_group(pc, features, npoints, group_size, r, group_all=False, rand_seed=None):
    """
    采样并分组
    Args:
        pc: 点云的坐标信息 shape"B, N', 3"
        features: 点云的特征信息（包括坐标） shape"B, N', C" or None
        npoints: 采样点数
        group_size: 每个分组（球）内部最多的点数
        r: 球半径
        group_all: 将所有点云分为一个组
        rand_seed: 随机数种子保证FPS采样的点不变

    Returns:
        centroids: 采样点（球心）的坐标 shape"B, npoint, 3"
        cur_features: 当前采样点的特征（包括坐标）shape"B, npoint, group_size, 3+C"
    """
    # FPS
    B, _, D = pc.shape
    if group_all:
        centroids = torch.zeros(B, 1, D)  # B, 1, 3
        group_points = pc.unsqueeze(dim=1)  # B, 1, N, 3
    else:
        centroids_idx = farthest_points_sampling(pc, npoints, rand_seed)
        centroids = ids2points(centroids_idx, pc)  # B, group_size, 3
        # group
        group_idx = ball_quire(pc, centroids_idx, group_size, r)
        group_points = ids2points(group_idx, pc)  # B, npoints, group_size, 3
        # group后的点集要分别减去中心点的值，即将group后的点集的球心移动到原点
        group_points -= centroids.unsqueeze(dim=2)
    if features is not None:
        if group_all:
            cur_features = torch.cat((group_points, features.unsqueeze(dim=1)), dim=-1)  # B, 1, N, 3 + C
        else:
            group_features = ids2points(group_idx, features)
            cur_features = torch.cat((group_points, group_features), dim=-1)  # B, npoints, group_size, C + 3
    else:
        cur_features = group_points
    return centroids, cur_features


def draw_ball(center, radius, ax):
    """绘制以center为球心，radius为半径的球"""
    # data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    # # plot
    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # surface plot
    # ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')
    # wire frame
    ax.plot_wireframe(x, y, z, rstride=15, cstride=15, colors='g')


def visSampleAndGroup(pc):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pc[..., 0], pc[..., 1], pc[..., 2])
    centroids = farthest_points_sampling(pc, 10)
    # print(centroids)
    sample_points = ids2points(centroids, pc)
    ax.scatter(sample_points[..., 0], sample_points[..., 1], sample_points[..., 2], marker='s', c='g', s=1)
    # print(sample_points.shape)
    group_idx = ball_quire(pc, centroids, group_size=5, r=1)
    sample_group = ids2points(group_idx, pc)
    for i in range(sample_points.shape[1]):
        ax.scatter(sample_group[:, i, :, 0], sample_group[:, i, :, 1], sample_group[:, i, :, 2], marker='s', c='r',
                   s=40)
    for i in range(sample_points.shape[1]):
        draw_ball(center=sample_points.squeeze().numpy()[i], radius=1, ax=ax)
    plt.show()
    # print(sample_group.shape)


if __name__ == '__main__':
    # test
    pc = torch.randn((10, 500, 3))
    visSampleAndGroup(pc[0].unsqueeze(dim=0))
    # test
    group_xyz, group_feature = sample_and_group(pc, None, 10, 5, 1)
    print(group_xyz.shape)
    print(group_feature.shape)

    features = torch.randn((10, 500, 20))
    group_xyz, group_feature = sample_and_group(pc, features, 10, 5, 1)
    print(group_xyz.shape)
    print(group_feature.shape)
