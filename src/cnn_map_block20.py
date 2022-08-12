#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:13:30 2022

@author: skrunes
"""

# import os
# import logging
from argparse import ArgumentParser
# from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# from torch import optim
# from torch.utils.data import DataLoader, random_split
# from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
import importlib
import src.tools as tools
# from dataset import DirDataset


class CNN(pl.LightningModule):
    # image_size = 64
    def __init__(self,
                 n_blocks: int = 9,
                 n_blocks_filters: int = 64,
                 layers_per_block: int = 2,
                 kernel_size: int = 3,
                 n_channels: int = 4,
                 n_classes: int = 1,
                 data_dir: str = "flat_polecontinent3",
                 standarize_outputs: bool = False,
                 optimizer: str = "Adam",
                 predict_squared: bool = False,
                 predict_inverse: bool = False,
                 loss_fn: str = "masked_mse",
                 q: float = 0.95,
                 padding_type: str = "valid",
                 **kwargs):

        super().__init__()
        self.n_blocks = n_blocks
        self.n_blocks_filters = n_blocks_filters
        self.layers_per_block = layers_per_block
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.standarize_outputs = standarize_outputs
        self.data_dir = data_dir
        self.optimizer = optimizer
        self.predict_squared = predict_squared
        self.predict_inverse = predict_inverse
        self.q = q
        self.loss_fn = loss_fn
        self.padding_type = padding_type
        self.save_hyperparameters()
        if standarize_outputs:
            f = open(data_dir+"norms_std_mean.txt")
            lines = f.readlines()
            assert len(lines) == 2, f"len {len(lines)}"
            self.norm_std = float(lines[0])
            self.norm_mean = float(lines[1])
            f.close()

        def conv(in_channels, out_channels, kernel_size, padding_type):
            # returns a block compsed of a Convolution layer with ReLU activation function
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          padding=padding_type),
                nn.BatchNorm2d(out_channels),
                nn.ELU())

        def block(layers_per_block, in_channels, out_channels, kernel_size, padding_type):
            block = nn.ModuleList()
            for i in range(layers_per_block):
                block.append(conv(in_channels, out_channels, kernel_size, padding_type))
                in_channels = out_channels
            return block

        self.layers = nn.ModuleList()
        self.layers.append(conv(n_channels, n_blocks_filters, kernel_size, padding_type ))
        for i in range(n_blocks):
            if i == 0:
                in_channels = n_blocks_filters + n_channels
            else :
                 in_channels = n_blocks_filters * 2
            self.layers.append(block(layers_per_block, in_channels, n_blocks_filters, kernel_size, padding_type))

        self.layers.append( nn.Sequential(
            nn.Conv2d(n_blocks_filters, n_classes, kernel_size, padding=padding_type)))



    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0 or i == self.n_blocks +1:
                out = layer(x)
            else:
                _, _, H, W = out.shape
                x = transforms.CenterCrop([H, W])(x)
                prev_out = out
                out = torch.cat([x,out], dim=1)
                x = prev_out
                for inner_layer in layer :
                    out = inner_layer(out)
                x = out
        return out
    def training_step(self, batch, batch_nb):
        return tools.step(self, batch)

    def training_epoch_end(self, outputs):
        tools.epoch_end(self, outputs, "train")

    def validation_step(self, batch, batch_nb):
        return tools.step(self, batch)

    def validation_epoch_end(self, outputs):
        tools.epoch_end(self, outputs, "val")

    def configure_optimizers(self):
       return tools.configure_optimizers(self)
