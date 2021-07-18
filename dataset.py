#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/7/16 下午7:00
# @Author : PH
# @Version：V 0.1
# @File : dataset.py
# @desc :
import os
import json
import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset

# label2idx = {'bathtub': 0,
#              'bed': 1,
#              'chair': 2,
#              'desk': 3,
#              'dresser': 4,
#              'monitor': 5,
#              'night_stand': 6,
#              'sofa': 7,
#              'table': 8,
#              'toilet': 9}


# label2idx = {'stairs': 0,
#              'curtain': 1,
#              'sink': 2,
#              'xbox': 3,
#              'plant': 4,
#              'car': 5,
#              'bed': 6,
#              'chair': 7,
#              'tv_stand': 8,
#              'lamp': 9,
#              'door': 10,
#              'wardrobe': 11,
#              'cone': 12,
#              'toilet': 13,
#              'bench': 14,
#              'bowl': 15,
#              'desk': 16,
#              'airplane': 17,
#              'bottle': 18,
#              'mantel': 19,
#              'radio': 20,
#              'person': 21,
#              'cup': 22,
#              'bathtub': 23,
#              'flower_pot': 24,
#              'night_stand': 25,
#              'monitor': 26,
#              'sofa': 27,
#              'vase': 28,
#              'range_hood': 29,
#              'laptop': 30,
#              'table': 31,
#              'keyboard': 32,
#              'piano': 33,
#              'glass_box': 34,
#              'tent': 35,
#              'guitar': 36,
#              'stool': 37,
#              'bookshelf': 38,
#              'dresser': 39}


def read_0ff(path, max_npoints):
    with open(path) as f:
        f.readline()
        head = f.readline().split(' ')
        size = int(head[0])
        xyz = np.zeros((size, 3))
        for i in range(size):
            point = f.readline().strip('\n').split(' ')
            xyz[i, 0] = float(point[0])
            xyz[i, 1] = float(point[1])
            xyz[i, 2] = float(point[2])
    idx = np.random.choice(size, max_npoints)
    xyz = torch.from_numpy(xyz[idx])
    return xyz


class ModelNet(Dataset):
    def __init__(self, dataset_path, max_points=2048, mode='train'):
        super(ModelNet, self).__init__()
        self.max_points = max_points
        self.label = os.listdir(dataset_path)
        with open(os.path.join(dataset_path, "label2idx.json"), 'r') as f:
            self.label2idx = json.load(f)
        self.point_paths = []
        for root, dirs, files in os.walk(dataset_path):
            if mode in root:
                for file in files:
                    if "off" in file:
                        self.point_paths.append(os.path.join(root, file))

    def __getitem__(self, index):
        xyz = read_0ff(self.point_paths[index], self.max_points)

        label = self.point_paths[index].split('/')[6]
        target = self.label2idx[label]
        return xyz, target

    def __len__(self):
        return len(self.point_paths)


def get_loader(dataset_path, batch_size, max_points=2048, mode='train', split_val=0.1):
    dataset = ModelNet(dataset_path, max_points, mode)

    if mode == 'train' and split_val > 0:
        val_len = round(len(dataset) * split_val)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True)
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        return train_loader, val_loader

    elif mode == 'test':
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        return test_loader


if __name__ == '__main__':
    data_loader, val_loader = get_loader("/home/ph/Dataset/ModelNet/ModelNet40",
                                         batch_size=32,
                                         max_points=2048,
                                         mode='train')
    for xyz, targets in data_loader:
        print(xyz.shape)
        print(targets)
        break
