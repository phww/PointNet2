#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/7/18 下午1:28
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc :
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.template import TemplateModel
from model import PointNet2Cls
from dataset import get_loader


class ModelT(TemplateModel):
    def __init__(self, model, optimizer, train_loader, test_loader):
        super(ModelT, self).__init__()
        self.model_list = [model]  # 模型的list
        self.optimizer_list = [optimizer]  # 优化器的list
        self.criterion = nn.CrossEntropyLoss()
        # 数据集
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 下面的可以不设定
        # tensorboard
        self.writer = SummaryWriter()  # 推荐设定
        # 训练时print的间隔
        self.log_per_step = len(self.test_loader.dataset) // (3 * 12)  # 推荐按数据集大小设定
        # 推荐设置学习率衰减
        self.lr_scheduler_type = "loss"  # None "metric" "loss"
        # check_point 目录
        self.ckpt_dir = "./check_point/" + time.strftime("%Y-%m-%d::%H:%M:%S")


def main(epochs):
    model = PointNet2Cls()
    train_loader, val_loader = get_loader("/home/ph/Dataset/ModelNet/ModelNet40",
                                          batch_size=12, max_points=2048, mode='train')
    # test_loader = get_loader("/home/ph/Dataset/ModelNet/ModelNet10",
    #                          batch_size=32, max_points=2048, mode='test')
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model_t = ModelT(model, optimizer, train_loader, val_loader)
    model_t.check_init()
    model_t.get_model_info(fake_inp=torch.randn((12, 2048, 3)))
    for epoch in range(epochs):
        model_t.train_loop()
        model_t.eval_loop()
        print("Done!")
        model_t.print_final_lr()
        model_t.print_best_metrics()


if __name__ == '__main__':
    main(10)
