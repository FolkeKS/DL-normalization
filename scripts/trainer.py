#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:36:00 2022

@author: skrunes
"""

import os
from argparse import ArgumentParser

import numpy as np
import torch

from src.unet import Unet
#from src.cnn import CNN
import src.cnn_block as cn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from src.data.dataset import DirLightDataset
import torch.multiprocessing
import importlib

importlib.reload(cn)
trainer = LightningCLI(cn.CNN, DirLightDataset, save_config_callback=None)
