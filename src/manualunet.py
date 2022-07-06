#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:37:39 2022

@author: skrunes
"""


import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import math
import functools
import pytorch_lightning as pl
from src.data.dataset import DirDataset
import linecache

def masked_mse(inputs, targets):
    _,_,H,W = inputs.shape
    
    #crop targets in case they are padded
    targets=transforms.CenterCrop([H,W])(targets)
    assert W==360, f"W = {W}"
    
    #mask defined where target equals zero
    mask_true = (~targets.eq(0.)).to(torch.uint8)
    masked_squared_error = torch.square(torch.flatten(mask_true) * (torch.flatten(targets) - torch.flatten(inputs)))
    masked_mse = torch.sum(masked_squared_error) / torch.sum(mask_true)
    return masked_mse

def masked_relative_error(inputs, targets,q=None):
    _,_,H,W = inputs.shape
    
    #crop targets in case they are padded
    targets=transforms.CenterCrop([H,W])(targets)
    assert W==360, f"W = {W}"
    #mask is true where normalization coefficients equals zero
    mask_true = (~targets.eq(0.)).to(torch.uint8)
    
    
    masked_abs_rel_error = torch.flatten(mask_true) * torch.abs((torch.flatten(targets) -
                                                                 torch.flatten(inputs))/torch.flatten(targets+1e-12))
    q_res = torch.zeros(1)
    if q is not None:

        q_res = torch.quantile(
            masked_abs_rel_error[torch.flatten(mask_true)], q)
    masked_mean_abs = torch.sum(masked_abs_rel_error) / torch.sum(mask_true)
    masked_max = torch.max(masked_abs_rel_error)

    masked_squared_rel_error = torch.flatten(mask_true) * torch.pow(torch.flatten(targets) -
                                                                  torch.flatten(inputs),2)/torch.pow(torch.flatten(targets+1e-12),2)

    rmse = torch.sqrt(torch.sum(masked_squared_rel_error) / torch.sum(mask_true))
    return masked_mean_abs, masked_max, q_res, rmse



class ManualUnet(pl.LightningModule):
        

    def __init__(self, data_dir:str="flat_polecontinent3",
                 n_channels: int =2,n_classes:int=1, loss_fn:str="masked_mse",
                 depth:int=5,depth_0_filters:int=64,final_layer_filters:int=5,
                 predict_squared: bool=False, predict_inverse: bool = False, 
                 optimizer:str="Adam",q:float=0.9999,
                 data_transformation: str="standardize", norm_layer:str="Batch",
                 ):
        """
       Example encoder-decoder model

       Args:
           datadir : datadir
           
           n_channels : n_channels
           
           n_classes : n_classes
           
           loss_fn : loss_fn 
           
           depth : depth
           
           depth_0_filters:depth_0_filters

           predict_squared : predict_squared
           
           predict_inverse : predict_inverse
           
           optimizer
       """
    
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.depth_0_filters=depth_0_filters
        self.optimizer = optimizer
        self.predict_squared = predict_squared
        self.predict_inverse = predict_inverse
        self.data_transformation = data_transformation
        self.data_dir = data_dir
        self.loss_fn = loss_fn
        self.q = q
        
        class DownSampleBlock(nn.Module):

            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv_block =  nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=4,
                                     stride=2, padding=1, bias=False),nn.BatchNorm2d(out_channels),
                                     nn.LeakyReLU(0.2,True))


            def forward(self, x):
                x_conv = self.conv_block(x)
        
                return x_conv , x
            
             

        class UpSampleBlock(nn.Module):

            def __init__(self, in_channels, out_channels,innermost=False,outermost=False):
                
                super().__init__()
                
                if innermost:
                    self.conv_block =  nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                kernel_size=4, stride=2,
                                                padding=1, bias=False),
                                                     nn.BatchNorm2d(out_channels),
                                         nn.ReLU(0.2))
                elif outermost:
                    self.conv_block =  nn.Sequential(nn.ConvTranspose2d(in_channels*2, out_channels,
                                                kernel_size=4, stride=2,
                                                padding=1, bias=False),
                                                     nn.BatchNorm2d(out_channels),
                                         nn.Tanh())
                else:
                    self.conv_block =  nn.Sequential(nn.ConvTranspose2d(in_channels*2, out_channels,
                                                kernel_size=4, stride=2,
                                                padding=1, bias=False),
                                                     nn.BatchNorm2d(out_channels),
                                         nn.ReLU(0.2))
                

        
            def forward(self, x, x_skip,innermost=False):
                if not innermost:
                    x = torch.cat([x, x_skip], dim=1)
                x = self.conv_block(x)
        
                return x
            
        class FirstBlock(nn.Module):

            def __init__(self, in_channels, out_channels):
                
                super().__init__()

                self.conv_block =  nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                            kernel_size=3, stride=1,
                                            padding="same", bias=False),
                                                 nn.BatchNorm2d(out_channels),
                                     nn.ReLU(0.2))
    

        
            def forward(self, x):
                x = self.conv_block(x)
        
                return x
        
        class LastBlock(nn.Module):

            def __init__(self, in_channels, out_channels):
                
                super().__init__()

                self.double_conv =  nn.Sequential(nn.Conv2d(in_channels*2, in_channels,
                                            kernel_size=3, stride=1,
                                            padding="same", bias=False),
                                                 nn.BatchNorm2d(in_channels),
                                     nn.ReLU(0.2),nn.Conv2d(in_channels, out_channels,
                                                                 kernel_size=3, stride=1,
                                                                 padding="same", bias=False))
    

        
            def forward(self, x, x_skip):
                x = torch.cat([x, x_skip], dim=1)
                x = self.double_conv(x)
        
                return x
        
        ngf=32
        
        self.first_conv = FirstBlock(self.n_channels, ngf) 
        
        self.downsample_block_1 = DownSampleBlock(ngf,ngf)
        self.downsample_block_2 = DownSampleBlock(ngf,ngf*2)
        self.downsample_block_3 = DownSampleBlock(ngf*2,ngf*4)
        self.downsample_block_4 = DownSampleBlock(ngf*4,ngf*8)
        self.downsample_block_5 = DownSampleBlock(ngf*8,ngf*8)
        
        self.upsample_block_5 = UpSampleBlock(ngf*8,ngf*8,innermost=True)
        self.upsample_block_4 = UpSampleBlock(ngf*8,ngf*4)
        self.upsample_block_3 = UpSampleBlock(ngf*4,ngf*2)
        self.upsample_block_2 = UpSampleBlock(ngf*2,ngf)
        self.upsample_block_1 = UpSampleBlock(ngf,ngf,outermost=True)
        
        self.last_conv = LastBlock(ngf, self.n_classes)     
        
        

        
        if data_transformation == "standardize":
            
            f = open(data_dir+"norms_std_mean.txt","r")
            lines = f.readlines()
            f.seek(0)
            assert len(lines)==2,f"len {len(lines)}"
            
            self.norm_std=float(lines[0])
            self.norm_mean=float(lines[1])
            f.close()
        elif data_transformation == "normalize":
            f = open(data_dir+"norms_min_max.txt","r")
            lines = f.readlines()
            f.seek(0)
            assert len(lines)==2,f"len {len(lines)}"
            
            self.norm_min=float(lines[0])
            self.norm_max=float(lines[1])
 
            f.close()

        self.save_hyperparameters()



    def forward(self, x):
        
        x = self.first_conv.forward(x)

        x, x_skip1 = self.downsample_block_1.forward(x)
        x, x_skip2 = self.downsample_block_2.forward(x)
        x, x_skip3 = self.downsample_block_3.forward(x)
        x, x_skip4 = self.downsample_block_4.forward(x)
        x, x_skip5 = self.downsample_block_5.forward(x)
        
        x = self.upsample_block_5(x,x_skip5,innermost=True)
        x = self.upsample_block_4(x,x_skip5)
        x = self.upsample_block_3(x,x_skip4)
        x = self.upsample_block_2(x,x_skip3)
        x = self.upsample_block_1(x,x_skip2)

        out = self.last_conv(x,x_skip1)

        return out

  
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        _, _, H, W = y_hat.shape
        modif = False
        if W > 360:
            W = 360
            modif = True
        if H > 290:
            H = 290
            modif = True
        if modif:
            y_hat = transforms.CenterCrop([H, W])(y_hat)


        #Calculate loss
        if self.loss_fn=="masked_mse": 
            loss = masked_mse(y_hat, y)

        else:
            raise NotImplementedError(self.loss_fn + " not implemented")   
        
        
        
        if self.data_transformation == "standardize":
            idx = torch.nonzero(y).split(1, dim=1)
            
            y[idx]= y[idx]*self.norm_std + self.norm_mean
            y_hat = y_hat*self.norm_std + self.norm_mean
        elif self.data_transformation == "normalize":
            
            #get non-zero elements to transform, elements equal to 0 serve as the mask and 
            #should not be touched for the true values y
            idx = torch.nonzero(y).split(1, dim=1)
            
            y[idx]= (y[idx]) * (self.norm_max-self.norm_min) +self.norm_min
            y_hat = (y_hat) * (self.norm_max-self.norm_min) +self.norm_min
            
            
        if self.predict_inverse:
            y_hat = 1/y_hat
            y = 1/y
        if not self.predict_squared:
            y_hat = y_hat**2

        rel_mean, rel_max, q_res, rmse = masked_relative_error(
                    y_hat, y**2, self.q)

        return {'loss': loss, 'rel_mean': rel_mean, 'rel_max': rel_max, 'rel_'+str(self.q) + "_quantile": q_res, 'rmse':rmse}
    
    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mean = torch.stack([x['rel_mean'] for x in outputs]).mean()
        avg_max = torch.stack([x['rel_max'] for x in outputs]).mean()
        
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mean = torch.stack([x['rel_mean'] for x in outputs]).mean()
        max_rel = torch.stack([x['rel_max'] for x in outputs]).max()
        avg_rmse = torch.stack([x['rmse'] for x in outputs]).mean()
        avg_q = torch.stack([x['rel_'+str(self.q) + '_quantile'] for x in outputs]).mean()

        self.log("train_loss", avg_loss)
        self.log("train_mean", avg_mean)
        self.log("train_max", max_rel)
        self.log("train_RMSE", avg_rmse)
        self.log("train_"+str(self.q) +'_quantile', avg_q)


    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        _, _, H, W = y_hat.shape
        modif = False
        if W > 360:
            W = 360
            modif = True
        if H > 290:
            H = 290
            modif = True
        if modif:
            y_hat = transforms.CenterCrop([H, W])(y_hat)
            
        #Calculate loss
        if self.loss_fn=="masked_mse": 
            loss = masked_mse(y_hat, y)

        else:
            raise NotImplementedError(self.loss_fn + " not implemented")
            
            
        if self.data_transformation == "standardize":
            idx = torch.nonzero(y).split(1, dim=1)
            
            y[idx]= y[idx]*self.norm_std + self.norm_mean
            y_hat = y_hat*self.norm_std + self.norm_mean
        elif self.data_transformation == "normalize":
            
            #get non-zero elements to transform, elements equal to 0 serve as the mask and 
            #should not be touched for the true values y
            idx = torch.nonzero(y).split(1, dim=1)
            
            y[idx]= (y[idx]) * (self.norm_max-self.norm_min) +self.norm_min
            y_hat = (y_hat) * (self.norm_max-self.norm_min) +self.norm_min
            
            
        if self.predict_inverse:
            y_hat = 1/y_hat
            y = 1/y
        if not self.predict_squared:
            y_hat = y_hat**2

        rel_mean, rel_max, q_res, rmse = masked_relative_error(
                    y_hat, y**2, self.q)

        return {'loss': loss, 'rel_mean': rel_mean, 'rel_max': rel_max, 'rel_'+str(self.q) + "_quantile": q_res, 'rmse':rmse}

    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mean = torch.stack([x['rel_mean'] for x in outputs]).mean()
        max_rel = torch.stack([x['rel_max'] for x in outputs]).max()
        avg_rmse = torch.stack([x['rmse'] for x in outputs]).mean()
        avg_q = torch.stack([x['rel_'+str(self.q) + '_quantile'] for x in outputs]).mean()

        self.log("val_loss", avg_loss)
        self.log("val_mean", avg_mean)
        self.log("val_max", max_rel)
        self.log("val_RMSE", avg_rmse)
        self.log("val_"+str(self.q) +'_quantile', avg_q)


    def configure_optimizers(self):
        #print(self.parameters())
        if self.optimizer=="Adamax":
            return torch.optim.Adamax(self.parameters())
        elif self.optimizer=="Adam":
            print(self.parameters,self.hparams)
            return torch.optim.Adam(self.parameters())
        else:
            raise NotImplementedError("Optimizer not implemented (available: Adam, Adamax)")









