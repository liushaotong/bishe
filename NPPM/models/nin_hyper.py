#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：bishe 
@File    ：nin_hyper.py
@IDE     ：PyCharm 
@Author  ：lst
@Date    ：2022/11/24 9:36 
'''
import torch
from torch import Tensor
import torch.nn as nn
from .gate_function import virtual_gate
from .nin_experiment_config import DatasetType


class ExperimentBaseModel(nn.Module):
    def __init__(self, dataset_type: DatasetType):
        super().__init__()
        self.dataset_type = dataset_type

    def forward(self, x) -> Tensor:
        raise NotImplementedError


class NiNBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, cfg) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, cfg, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(cfg)

        # if gate_flag is True:
        self.gate = virtual_gate(cfg)

        self.conv2 = nn.Conv2d(cfg, planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes)

        # self.gate_flag = gate_flag

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # if self.gate_flag:
        x = self.gate(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


class NiN(ExperimentBaseModel):
    def __init__(self, depth: int, width: int, base_width: int, dataset_type: DatasetType, cfg=None) -> None:
        super().__init__(dataset_type)
        if cfg == None:
            cfg = [width * base_width] * depth  # [350,350,350,350]
        self.cfg = cfg
        self.base_width = base_width

        blocks = []
        blocks.append(NiNBlock(self.dataset_type.D[0], self.base_width * width, cfg[0]))
        for i in range(1, depth):
            blocks.append(NiNBlock(self.base_width * width, self.base_width * width, cfg[i]))

        self.blocks = nn.Sequential(*blocks)

        self.conv = nn.Conv2d(self.base_width * width, self.dataset_type.K, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(self.dataset_type.K)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.gate_flag = gate_flag

    def forward(self, x):
        x = self.blocks(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)

        return x.squeeze()

    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width)
        self.structure = structure
        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):  #
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.set_structure_value(arch_vector.squeeze()[start:end])
                start = end

                i+=1

    def get_gate_grads(self):
        all_grad = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                #print(m.weights.grad.data)
                all_grad.append(m.get_grads().clone())
        #print(all_grad[0])
        return all_grad

    def foreze_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                #count+=1
            elif isinstance(m, nn.Conv2d):
                m.eval()
                m.weight.requires_grad = False
                #count+=1
            elif isinstance(m, nn.Linear):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.eval()
                #print(m)
                count += 1


if __name__ == '__main__':
    net = NiN(depth=4, width=14, base_width=25, dataset_type=DatasetType.CIFAR10)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
