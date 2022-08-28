#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:02:24 2022

@author: coulaud
"""
import torch
import torchvision.transforms as transforms
import numpy as np
torch.set_printoptions(precision=10)

def masked_mse(inputs, targets):
    """Computes the Mean Square Error between the estimation and the true value of gamma

    Args:
        inputs (Pytorch tensor): estimated value of $\gamma$
        targets (Pytorch tensor): the true value of $\gamma$
    Returns:
        float: MSE gamma gamma_hat
    """    
    _, _, H, W = inputs.shape
    # crop targets in case they are padded
    targets = transforms.CenterCrop([H, W])(targets)
    # assert (W == 360 and H == 290) or (W == 290 and H == 360), f"W = {W} H = {H}"
    # mask defined where target equals zero
    mask_true = (~targets.eq(0.)).to(torch.float32)
    masked_squared_error = torch.square(torch.flatten(
        mask_true) * (torch.flatten(targets) - torch.flatten(inputs)))
    masked_mse = (1/torch.sum(mask_true)) * torch.sum(masked_squared_error) 
    return masked_mse

def masked_mse_eps(inputs, targets, standartize, std=1, mean=0, square=False):
    """Computes the mean masked squared relative error epsilon

    Args:
        inputs (Pytorch tensor): Estimated value of $\gamma$
        targets (Pytorch tensor): True value of $\gamma$
        standartize (boolean): if the inputs/targets are standardized
        std (int, optional): std value used for the standardization. Defaults to 1.
        mean (int, optional): mean used for the standardization_. Defaults to 0.
        square (bool, optional): if the inputs/targets are in squares. Defaults to False.

    Returns:
        float: returns the mean of the masked relative error 
    """    
    # |E[eps^2]
    _, _, H, W = inputs.shape
    # crop targets in case they are padded
    targets = transforms.CenterCrop([H, W])(targets)
    # assert (W == 360 and H == 290) or (W == 290 and H == 360), f"W = {W} H = {H}"
    mask_true = (~targets.eq(0.)).to(torch.uint8)
    if standartize:
        targets = targets*std + mean
        inputs = inputs*std + mean
    if not square:
        square_diff = torch.flatten(mask_true) * torch.square(torch.flatten(inputs)) - torch.square(torch.flatten(targets))
        masked_squared_rel_error = torch.square(torch.div(square_diff,torch.square(torch.flatten(targets))+1e-12))
    else:
        diff = torch.flatten(mask_true) * torch.flatten(inputs) - torch.flatten(targets)
        masked_squared_rel_error = torch.square(torch.div(diff,torch.flatten(targets)+1e-12))
    mse = (1/torch.sum(mask_true)) * torch.sum(masked_squared_rel_error)
    return mse

def masked_relative_error(inputs, targets, q=None):
    """Computes the metrics of epsilon: max, mean, quantile, rmse

    Args:
        inputs (Pytorch tensor): Estimated value of $\gamma$
        targets (Pytorch tensor): True value of $\gamma$
        q (int, optional): Quantile to consider. Defaults to None.

    Returns:
        (float*float*float*float): the metrics
    """    
    _, _, H, W = inputs.shape
    # crop targets in case they are padded
    targets = transforms.CenterCrop([H, W])(targets)
    # assert (W == 360 and H == 290) or (W == 290 and H == 360), f"W = {W} H = {H}"
    
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

    rmse = torch.sqrt(masked_mse_eps(inputs, targets, standartize=False, square=True))
    print(masked_max, masked_mean_abs, q_res)
    return masked_mean_abs, masked_max, q_res, rmse


def compute_loss(model,y_hat, y):
    """Computes the required loss

    Args:
        model (_type_): the saved NN model
        y_hat (Pytorch tensor): True value of $\gamma$
        y (Pytorch tensor): Estimated value of $\gamma$
    Raises:
        NotImplementedError: The given loss function is not implemented

    Returns:
        float: the loss
    """    
    if model.loss_fn == "masked_mse":
        return masked_mse(y, y_hat, )
    elif model.loss_fn == "masked_mse_eps":
        return masked_mse_eps(y_hat, y, standartize=True, std=model.norm_std, mean=model.norm_mean)
    else:
        raise NotImplementedError(model.loss_fn + " not implemented")

def epoch_end(self, outputs, step):
    """Compute and save the metrics at the epoch end for the given step

    Args:
        outputs (Pytroch tensor): Metrics of the steps in the epoch
        step (string): either "training" or "validation"
    """    

    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_mean = torch.stack([x['rel_mean'] for x in outputs]).mean()
    max_rel = torch.stack([x['rel_max'] for x in outputs]).max()
    avg_rmse = torch.stack([x['rmse'] for x in outputs]).mean()
    avg_q = torch.stack([x['rel_'+str(self.q) + '_quantile'] for x in outputs]).mean()
    print("Avg loss:", avg_loss)
    self.log(step+"_loss", avg_loss)
    self.log(step+"_mean", avg_mean)
    self.log(step+"_max", max_rel)
    self.log(step+"_RMSE", avg_rmse)
    self.log(step+'_'+str(self.q) +'_quantile', avg_q)

def transform_crop(y):
    """Crop the input tensor

    Args:
        y (Pytorch tensor): tensor to crop

    Returns:
        Pytorch tensor: cropped tensor
    """    
    _, _, H, W = y.shape
    modif = False
    if W > 360:
        W = 360
        modif = True
    if H > 290:
        H = 290
        modif = True
    if modif:
        return transforms.CenterCrop([H, W])(y)
    else :
        return y


def step(self, batch):
    """Computes the metrics at the end of a step

    Args:
        batch (_type_): Output of the :class: 'DataLoader'

    Returns:
        dictionary: dictionary of the metrics
    """    
    x, y = batch
    y_hat = self.forward(x)
    # Calculate loss
    loss = compute_loss(self, y_hat, y)

    if self.standarize_outputs:
        idx = torch.nonzero(y).split(1, dim=1)
        y[idx] = y[idx]*self.norm_std + self.norm_mean
        y_hat = y_hat*self.norm_std + self.norm_mean

    if self.predict_inverse:
        y_hat = 1/y_hat
        y = 1/y
    if not self.predict_squared:
        y_hat = y_hat**2
        y = y**2

    rel_mean, rel_max, q_res, rmse = masked_relative_error(
                y_hat, y, self.q)
    return {'loss': loss, 'rel_mean': rel_mean, 'rel_max': rel_max, 'rel_'+str(self.q) + "_quantile": q_res, 'rmse':rmse}

def configure_optimizers(self):
    """Apply the chosen optimizers

    Raises:
        NotImplementedError: the optimizer is not implemented

    Returns:
        class: the optimizer
    """    
    # print(self.parameters())
    if self.optimizer == "Adamax":
        return torch.optim.Adamax(self.parameters())
    elif self.optimizer == "Adam":
        print(self.parameters, self.hparams)
        return torch.optim.Adam(self.parameters())
    else:
        raise NotImplementedError(
            "Optimizer not implemented (available: Adam, Adamax)")