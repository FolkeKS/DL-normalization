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
                 n_channels: int = 3,
                 n_classes: int = 1,
                 data_dir: str = "flat_polecontinent3",
                 standarize_outputs: bool = False,
                 optimizer: str = "Adam",
                 predict_squared: bool = False,
                 predict_inverse: bool = False,
                 loss_fn: str = "masked_mse",
                 q: float = 0.95,
                 **kwargs):

        super().__init__()
        # self.hparams = hparams
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.standarize_outputs = standarize_outputs
        self.data_dir = data_dir
        self.optimizer = optimizer
        self.predict_squared = predict_squared
        self.predict_inverse = predict_inverse
        self.q = q
        self.loss_fn = loss_fn

        if standarize_outputs:
            f = open(data_dir+"norms_std_mean.txt")
            lines = f.readlines()
            assert len(lines) == 2, f"len {len(lines)}"
            self.norm_std = float(lines[0])
            self.norm_mean = float(lines[1])
            f.close()
            #print("std ",self.norm_std," mean ", self.norm_mean)

        def conv(in_channels, out_channels):
            # returns a block compsed of a Convolution layer with ReLU activation function
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding="same", padding_mode="replicate"),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        self.cnv1 = conv(self.n_channels, 32)
        self.cnv2 = conv(32, 64)
        self.cnv3 = conv(64, 64)
        self.cnv4 = conv(64, 64)
        self.cnv5 = conv(64, 64)
        self.last = nn.Sequential(
            nn.Conv2d(64, self.n_classes, 3, padding="same", padding_mode="replicate"))

    def forward(self, x):
        out = self.cnv1(x)
        out = self.cnv2(out)
        out = self.cnv3(out)
        out = self.cnv4(out)
        out = self.cnv5(out)
        out = self.last(out)
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
       return tools.configure_optimizers
