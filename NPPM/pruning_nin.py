#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：bishe 
@File    ：pruning_nin.py
@IDE     ：PyCharm 
@Author  ：lst
@Date    ：2022/11/30 10:16 
'''
import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
from models.nin_hyper import NiN
from models.gate_function import virtual_gate
from models.hypernet import Simplified_Gate
from models.nin_experiment_config import HParams
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
dir = '/datasets/cifar10/'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not os.path.exists(args.save):
    os.makedirs(args.save)

if not os.path.exists(args.save):
    os.makedirs(args.save)
model = NiN(depth=HParams.model_depth, width=HParams.model_width, base_width=HParams.base_width,
                dataset_type=HParams.dataset_type)
model_name = 'nin'
stat_dict = torch.load('./checkpoint/%s-pruned.pth.tar'%(model_name))
model.load_state_dict(stat_dict['net'])
model.cuda()
width, structure = model.count_structure()
hyper_net = Simplified_Gate(structure=structure, T=0.4, base=3.0,)
hyper_net.cuda()
hyper_net.load_state_dict(stat_dict['hyper_net'])
hyper_net.eval()

with torch.no_grad():
    vector = hyper_net()
parameters = hyper_net.transfrom_output(vector.detach())  # 每一层的结构参数

cfg = []  # 结构cfg
for i in range(len(parameters)):
    cfg.append(int(parameters[i].sum().item()))
newmodel = NiN(depth=HParams.model_depth, width=HParams.model_width, base_width=HParams.base_width,
                dataset_type=HParams.dataset_type, cfg=cfg)
newmodel.cuda()
old_modules = list(model.modules())
new_modules = list(newmodel.modules())
start_mask = torch.ones(3)
soft_gate_count = 0
conv_count = 0
end_mask = parameters[soft_gate_count]







