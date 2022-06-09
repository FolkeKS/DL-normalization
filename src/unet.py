#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:11:34 2022

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
    mask_true = (~targets.eq(0.)).to(torch.float32)
    masked_squared_error = torch.square(torch.flatten(mask_true) * (torch.flatten(targets) - torch.flatten(inputs)))
    masked_mse = torch.sum(masked_squared_error) / torch.sum(mask_true)
    return masked_mse

def masked_relative_error(inputs, targets,q=None):
    _,_,H,W = inputs.shape
    
    #crop targets in case they are padded
    targets=transforms.CenterCrop([H,W])(targets)
    assert W==360, f"W = {W}"
    #mask is true where normalization coefficients equals zero
    mask_true = (~targets.eq(0.)).to(torch.float32)
    
    
    masked_abs_rel_error = torch.flatten(mask_true) * torch.abs((torch.flatten(targets) - 
                                                       torch.flatten(inputs))/torch.flatten(targets+1e-12) )
    q_res=torch.zeros(1)
    if q is not None:
        q_res = torch.quantile(masked_abs_rel_error,q)
    masked_mean_abs = torch.sum(masked_abs_rel_error) / torch.sum(mask_true)
    masked_max = torch.max(masked_abs_rel_error)
    return masked_mean_abs,masked_max,q_res
    


class Unet(pl.LightningModule):
    def __init__(self, data_dir:str="flat_polecontinent3",
                 n_channels: int =2,n_classes:int=1, loss_fn:str="masked_mse",
                 depth:int=5,depth_0_filters:int=64,final_layer_filters:int=5,
                 predict_squared: bool=False, predict_inverse: bool = False, 
                 optimizer:str="Adam", q:float=0.95,
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
            
            
        if norm_layer=="Batch":
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer=="Instance":
            self.norm_layer = nn.InstanceNorm2d
        self.save_hyperparameters()


        self.pixmodel = nn.Sequential(UnetGenerator(input_nc=self.n_channels, 
                                                    output_nc = final_layer_filters, 
                                                    num_downs=self.depth, 
                                   ngf=depth_0_filters,
                                   norm_layer=self.norm_layer, use_dropout=True),
                                      nn.Conv2d(final_layer_filters, self.n_classes, 2, 
                                                padding="same",padding_mode="replicate"))


    def forward(self, x):

        out = self.pixmodel(x)

        return out

  
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        _,_,H,W = y_hat.shape
        if W>360:
            W=360
            y_hat=transforms.CenterCrop([H,W])(y_hat)

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
            
            
        if self.predict_squared == True:
            if self.predict_inverse == True:
                rel_mean,rel_max = masked_relative_error(1/y_hat, 1/y**2)
            elif self.predict_inverse == False:
                rel_mean,rel_max = masked_relative_error(y_hat, y**2)
        else:
            if self.predict_inverse == True:
                rel_mean,rel_max = masked_relative_error(1/y_hat**2, 1/y**2)
            elif self.predict_inverse == False:
                rel_mean,rel_max,q_res = masked_relative_error(y_hat**2, y**2,self.q)

        return {'loss': loss,'rel_mean': rel_mean,'rel_max': rel_max, 'rel_'+str(self.q) +'_quantile': q_res}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mean = torch.stack([x['rel_mean'] for x in outputs]).mean()
        max_rel = torch.stack([x['rel_max'] for x in outputs]).max()
        avg_q = torch.stack([x['rel_'+str(self.q) +'_quantile'] for x in outputs]).mean()
        
        self.log("train_loss", avg_loss)
        self.log("train_mean", avg_mean)
        self.log("train_max", max_rel)
        self.log('train_'+str(self.q) +'_quantile', avg_q)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        _,_,H,W = y_hat.shape
        if W>360:
            W=360
            y_hat=transforms.CenterCrop([H,W])(y_hat)
            
            
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
            
            
        if self.predict_squared == True:
            if self.predict_inverse == True:
                rel_mean,rel_max = masked_relative_error(1/y_hat, 1/y**2)
            elif self.predict_inverse == False:
                rel_mean,rel_max = masked_relative_error(y_hat, y**2)
        else:
            if self.predict_inverse == True:
                rel_mean,rel_max = masked_relative_error(1/y_hat**2, 1/y**2)
            elif self.predict_inverse == False:
                rel_mean,rel_max,q_res = masked_relative_error(y_hat**2, y**2,self.q)

        return {'loss': loss,'rel_mean': rel_mean,'rel_max': rel_max,'rel_'+str(self.q) +"_quantile": q_res}

    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mean = torch.stack([x['rel_mean'] for x in outputs]).mean()
        max_rel = torch.stack([x['rel_max'] for x in outputs]).max()
        avg_q = torch.stack([x['rel_'+str(self.q) +'_quantile'] for x in outputs]).mean()
        
        self.log("val_loss", avg_loss)
        self.log("val_mean", avg_mean)
        self.log("val_max", max_rel)
        self.log('val_'+str(self.q) +'_quantile', avg_q)

    def configure_optimizers(self):
        #print(self.parameters())
        if self.optimizer=="Adamax":
            return torch.optim.Adamax(self.parameters())
        elif self.optimizer=="Adam":
            print(self.parameters,self.hparams)
            return torch.optim.Adam(self.parameters())
        else:
            raise NotImplementedError("Optimizer not implemented (available: Adam, Adamax)")

    
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, 
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, 
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, 
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, 
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, 
                                             submodule=unet_block, outermost=True, 
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)   
    
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        
        if input_nc is None:
            input_nc = outer_nc
            
            

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]

            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
  
            
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
