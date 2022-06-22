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
# from dataset import DirDataset


def masked_mse(inputs, targets):
    _, _, H, W = inputs.shape

    # crop targets in case they are padded
    print("H :", H, "W: ", W)
    targets = transforms.CenterCrop([H, W])(targets)
    assert W == 360, f"W = {W}"

    # mask defined where target equals zero
    mask_true = (~targets.eq(0.)).to(torch.float32)
    masked_squared_error = torch.square(torch.flatten(
        mask_true) * (torch.flatten(targets) - torch.flatten(inputs)))
    masked_mse = torch.sum(masked_squared_error) / torch.sum(mask_true)
    return masked_mse


def masked_relative_error(inputs, targets):
    _, _, H, W = inputs.shape

    # crop targets in case they are padded
    targets = transforms.CenterCrop([H, W])(targets)
    assert W == 360
    # mask defined where target equals zero
    mask_true = (~targets.eq(0.)).to(torch.float32)

    masked_rel_error = torch.flatten(mask_true) * (torch.flatten(targets) -
                                                   torch.flatten(inputs))/torch.flatten(targets+1e-12)

    masked_mean_abs = torch.sum(
        torch.abs(masked_rel_error)) / torch.sum(mask_true)
    masked_max = torch.max(torch.abs(masked_rel_error))
    return masked_mean_abs, masked_max


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
        if standarize_outputs:
            f = open(data_dir+"norms_std_mean.txt")
            assert len(f.readlines()) == 2

            self.norm_std = 110439.79485414618
 # float(f.readlines()[0])
            self.norm_mean = 525208.1680871231
# float(f.readlines()[1])
            f.close()

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
        self.last = nn.Sequential(
            nn.Conv2d(64, self.n_classes, 3, padding="same", padding_mode="replicate"))

    def forward(self, x):
        out = self.cnv1(x)
        out = self.cnv2(out)
        out = self.cnv3(out)
        out = self.last(out)
        return out

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        _, _, H, W = y_hat.shape
        if W > 360:
            W = 360
            y_hat = transforms.CenterCrop([H, W])(y_hat)

        loss = masked_mse(y_hat, y)

        if self.standarize_outputs:
            idx = torch.nonzero(y).split(1, dim=1)

            y[idx] = y[idx] * self.norm_std + self.norm_mean
            y_hat = y_hat * self.norm_std + self.norm_mean

        if self.predict_squared == True:
            if self.predict_inverse == True:
                rel_mean, rel_max = masked_relative_error(1/y_hat, 1/y**2)
            elif self.predict_inverse == False:
                rel_mean, rel_max = masked_relative_error(y_hat, y**2)
        else:
            if self.predict_inverse == True:
                rel_mean, rel_max = masked_relative_error(1/y_hat**2, 1/y**2)
            elif self.predict_inverse == False:
                rel_mean, rel_max = masked_relative_error(y_hat**2, y**2)

        return {'loss': loss, 'rel_mean': rel_mean, 'rel_max': rel_max}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mean = torch.stack([x['rel_mean'] for x in outputs]).mean()
        avg_max = torch.stack([x['rel_max'] for x in outputs]).mean()

        self.log("train_loss", avg_loss)
        self.log("train_mean", avg_mean)
        self.log("train_max", avg_max)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        _, _, H, W = y_hat.shape
        if W > 360:
            W = 360
            y_hat = transforms.CenterCrop([H, W])(y_hat)

        loss = masked_mse(y_hat, y)
        if self.standarize_outputs:
            idx = torch.nonzero(y).split(1, dim=1)

            y[idx] = y[idx]*self.norm_std + self.norm_mean
            y_hat = y_hat*self.norm_std + self.norm_mean

        if self.predict_squared == True:
            if self.predict_inverse == True:
                rel_mean, rel_max = masked_relative_error(1/y_hat, 1/y**2)
            elif self.predict_inverse == False:
                rel_mean, rel_max = masked_relative_error(y_hat, y**2)
        else:
            if self.predict_inverse == True:
                rel_mean, rel_max = masked_relative_error(1/y_hat**2, 1/y**2)
            elif self.predict_inverse == False:
                rel_mean, rel_max = masked_relative_error(y_hat**2, y**2)

        return {'loss': loss, 'rel_mean': rel_mean, 'rel_max': rel_max}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mean = torch.stack([x['rel_mean'] for x in outputs]).mean()
        avg_max = torch.stack([x['rel_max'] for x in outputs]).mean()

        self.log("val_loss", avg_loss)
        self.log("val_mean", avg_mean)
        self.log("val_max", avg_max)

    def configure_optimizers(self):
        print(self.parameters())
        if self.optimizer == "Adamax":
            return torch.optim.Adamax(self.parameters())
        elif self.optimizer == "Adam":
            print(self.parameters, self.hparams)
            return torch.optim.Adam(self.parameters())
        else:
            raise NotImplementedError(
                "Optimizer not implemented (available: Adam, Adamax)")
