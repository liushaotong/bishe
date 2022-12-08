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
stat_dict = torch.load('./checkpoint/%s-pruned.pth.tar' % (model_name))
model.load_state_dict(stat_dict['net'])
model.cuda()
width, structure = model.count_structure()
hyper_net = Simplified_Gate(structure=structure, T=0.4, base=3.0, )
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

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        if isinstance(old_modules[layer_id + 1], virtual_gate):  # 剪枝的batch norm层
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

    elif isinstance(m0, nn.Conv2d):
        if conv_count == 3 * HParams.model_depth + 1:  # 最后一个直接复制
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue

        if isinstance(old_modules[layer_id + 2], virtual_gate):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            print(m1.weight.data.size())

            m0_next = old_modules[layer_id + 3]
            m1_next = new_modules[layer_id + 3]
            if isinstance(m0_next, nn.Conv2d):
                w1 = m0_next.weight.data[:, idx1.tolist(), :, :].clone()
                m1_next.weight.data = w1.clone()
                print(m1_next.weight.data.size())

            soft_gate_count += 1
            start_mask = end_mask.clone()  # 改变用于batch norm层
            if soft_gate_count < len(parameters):
                end_mask = parameters[soft_gate_count]  # 改变用于conv层
            continue

        if isinstance(old_modules[layer_id - 1], virtual_gate):
            continue
        m1.weight.data = m0.weight.data.clone()

    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        # m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

model.cpu()
newmodel.cpu()
t_o = print_model_param_nums(model)
t_n = print_model_param_nums(newmodel)
print_model_param_flops(model, input_res=32)
print_model_param_flops(newmodel, input_res=32)

all_parameters = torch.cat(parameters)
pruning_rate = float((all_parameters == 1).sum()) / float(all_parameters.size(0))
print("pruning rate：", pruning_rate)

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()},
           os.path.join(args.save, './checkpoint/%s_new.pth.tar' % (model_name)))
