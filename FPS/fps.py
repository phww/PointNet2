#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/7/13 下午3:53
# @Author : PH
# @Version：V 0.1
# @File : fps.py
# @desc :
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 100
random_sample_idx = np.arange(0, N)
def vis_random_sample():
    random_idx = np.random.choice(random_sample_idx, 1, replace=True)
    point = random_pc_copy[random_idx]
    ax2.scatter(point[0][0], point[0][1], point[0][2], c='r', marker='s')
    fig.show()
    plt.pause(0.5)

def vis(point):
    ax.scatter(point[0], point[1], point[2], c='r', marker='s')
    fig.show()
    plt.pause(0.5)


fig = plt.figure(figsize=(10,20))
ax = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# 随机生成点云数据（N，3）
random_pc = np.random.randn(N, 3)
random_pc_copy = random_pc.copy()
ax2.scatter(random_pc_copy[:, 0], random_pc_copy[:, 1], random_pc_copy[:, 2], edgecolors='b', marker='o')
# print(random_pc)
ax.scatter(random_pc[:, 0], random_pc[:, 1], random_pc[:, 2], edgecolors='b', marker='o')
fig.show()
vis_random_sample()

#
max_dist = -1e8
farthest_idx = -1
sample_pc_idx = list()
random_pc_idx = [num for num in range(N)]
# 随机采样第一个点
first_idx = random.randint(0, N)
sample_pc_idx.append(first_idx)
random_pc_idx.remove(first_idx)
vis(random_pc[first_idx])
vis_random_sample()

# 第二个点：距离第一个点最远的点
p1 = random_pc[0]
for j in random_pc_idx:
    p2 = random_pc[j]
    dist = np.sqrt(np.sum((p1 - p2) ** 2))
    if dist > max_dist:
        max_dist = dist
        farthest_idx = j
sample_pc_idx.append(farthest_idx)
random_pc_idx.remove(farthest_idx)
vis(random_pc[farthest_idx])
vis_random_sample()

# FPS
while len(random_pc_idx):
    farthest_idx = -1
    max_dist = -1e9
    for i in sample_pc_idx:
        min_idx = -1
        min_dist = 1e9
        p1 = random_pc[i]
        for j in random_pc_idx:
            p2 = random_pc[j]
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        if min_dist > max_dist:
            farthest_idx = min_idx
    sample_pc_idx.append(farthest_idx)
    random_pc_idx.remove(farthest_idx)
    vis(random_pc[farthest_idx])
    vis_random_sample()
    print(f"select:{farthest_idx}", "rest:", len(random_pc_idx), "select:", len(sample_pc_idx))

