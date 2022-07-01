#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:02:24 2022

@author: coulaud
"""
import torch
import torchvision.transforms as transforms
import numpy as np

def masked_mse(inputs, targets):
    _, _, H, W = inputs.shape
    # crop targets in case they are padded
    targets = transforms.CenterCrop([H, W])(targets)
    assert W == 360, f"W = {W}"
    assert H == 290, f"H = {H}" 
    # mask defined where target equals zero
    mask_true = (~targets.eq(0.)).to(torch.float32)
    masked_squared_error = torch.square(torch.flatten(
        mask_true) * (torch.flatten(targets) - torch.flatten(inputs)))
    masked_mse = torch.sum(masked_squared_error) / torch.sum(mask_true)
    return masked_mse


def masked_relative_error(inputs, targets, q=None):
    _, _, H, W = inputs.shape
    # crop targets in case they are padded
    targets = transforms.CenterCrop([H, W])(targets)
    assert W == 360, f"W = {W}"
    assert H == 290, f"H = {H}"
    # mask is true where normalization coefficients equals zero
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




def epoch_end(self, outputs, step):

    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_mean = torch.stack([x['rel_mean'] for x in outputs]).mean()
    max_rel = torch.stack([x['rel_max'] for x in outputs]).max()
    avg_rmse = torch.stack([x['rmse'] for x in outputs]).mean()
    avg_q = torch.stack([x['rel_'+str(self.q) + '_quantile'] for x in outputs]).mean()

    self.log(step+"_loss", avg_loss)
    self.log(step+"_mean", avg_mean)
    self.log(step+"_max", max_rel)
    self.log(step+"_RMSE", avg_rmse)
    self.log(step+'_'+str(self.q) +'_quantile', avg_q)

def transform_crop(y):
    _, _, H, W = y.shape
    modif = False
    if W > 360:
        W = 360
        modif = True
    if H > 290:
        H = 290
        modif = True
    if modif :
        return  transforms.CenterCrop([H, W])(y)
    else :
        return y


def step(self, batch):
        x, y = batch
        y_hat = transform_crop(self.forward(x))
        # Calculate loss
        if self.loss_fn == "masked_mse":
            loss = masked_mse(y_hat, y)
        else:
            raise NotImplementedError(self.loss_fn + " not implemented")

        if self.standarize_outputs:
            idx = torch.nonzero(y).split(1, dim=1)
            y[idx] = y[idx]*self.norm_std + self.norm_mean
            y_hat = y_hat*self.norm_std + self.norm_mean

        if self.predict_inverse:
            y_hat = 1/y_hat
        if self.predict_squared:
            y_hat = y_hat**2

        rel_mean, rel_max, q_res, rmse = masked_relative_error(
                    y_hat**2, y**2, self.q)

        return {'loss': loss, 'rel_mean': rel_mean, 'rel_max': rel_max, 'rel_'+str(self.q) + "_quantile": q_res, 'rmse':rmse}

def configure_optimizers(self):
    # print(self.parameters())
    if self.optimizer == "Adamax":
        return torch.optim.Adamax(self.parameters())
    elif self.optimizer == "Adam":
        print(self.parameters, self.hparams)
        return torch.optim.Adam(self.parameters())
    else:
        raise NotImplementedError(
            "Optimizer not implemented (available: Adam, Adamax)")