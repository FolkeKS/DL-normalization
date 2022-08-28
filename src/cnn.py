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

def conv(in_channels, out_channels, kernel_size, padding_type, acti=True):
    """ Creates a convolution layer

    Arguments:
        in_channels (int) : number of input channels
        out_channels (int) : number of output channels
        kernel_size (int) : kernel size
        padding_type (string) : padding type ("valid", "same",...)
        acti (bool, optional): If the activation function is used. The default value is True.

    Returns :
        nn.Conv2d : a convolution layer
    """    
    if acti:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                        padding=padding_type),
            nn.BatchNorm2d(out_channels),
            nn.ELU())
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                        padding=padding_type),
            nn.BatchNorm2d(out_channels))

class Block(nn.Module):
    """ Returns a block composed of a Convolution layer
    """    
    def __init__(self,layers_per_block, in_channels, out_channels, kernel_size, padding_type):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers_per_block-1):
            self.layers.append(conv(in_channels, out_channels, kernel_size, padding_type,True))
            in_channels = out_channels
        self.layers.append(conv(in_channels, out_channels, kernel_size, padding_type,False))
    def forward(self, x):
        """ Applies the model to an input

        Args:
            x (Pytorch tensor): the model input
        Returns:
            Pytorch tensor: the model output
        """      
        out = x
        for layer in self.layers:
            out = layer(out)
        _, _, H, W = out.shape
        x = transforms.CenterCrop([H, W])(x)
        acti = nn.ELU()
        return acti(x+out)

class CNN(pl.LightningModule):
    # image_size = 64
    def __init__(self,
                 n_blocks: int = 4,
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


        self.layers = nn.ModuleList()
        self.layers.append(conv(n_channels, n_blocks_filters, kernel_size, padding_type ))
        for i in range(n_blocks):
            self.layers.append(Block(layers_per_block, n_blocks_filters, n_blocks_filters, kernel_size, padding_type))

        self.layers.append( nn.Sequential(
            nn.Conv2d(n_blocks_filters, n_classes, kernel_size, padding=padding_type)))



    def forward(self, x):
        """ Applies to model to an input

        Args:
            x (Pytorch tensor): the model input
        Returns:
            Pytorch tensor: the model output
        """        
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_nb):
        """Computes the metrics at the end of the training step
        Args:
            batch (_type_): Output of the :class: 'DataLoader'

        Returns:
            dictionary: the metrics
        """    
        return tools.step(self, batch)

    def training_epoch_end(self, outputs):
        """Computes and saves the metrics at the training epoch end

        Args:
            outputs (Pytroch tensor): Metrics of the steps in the epoch
        """    
        tools.epoch_end(self, outputs, "train")

    def validation_step(self, batch, batch_nb):
        """Computes the metrics at the end of the validation step
        Args:
            batch (_type_): Output of the :class: 'DataLoader'

        Returns:
            dictionary: the metrics
        """   
        return tools.step(self, batch)

    def validation_epoch_end(self, outputs):
        """Computes and saves the metrics at the validation epoch end

        Args:
            outputs (Pytroch tensor): Metrics of the steps in the epoch
        """    
        tools.epoch_end(self, outputs, "val")

    def configure_optimizers(self):   
        """Applies the choosen optimizers

        Returns:
            class: the optimizer
        """    
        return tools.configure_optimizers(self)
