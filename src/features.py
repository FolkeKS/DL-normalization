#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:38:40 2022

@author: skrunes
"""

import numpy as np
import os
import netCDF4 as nc

def get_std_mean(dirpath,var_names = ["alpha_i_minus","alpha_j_minus","w","norm_coeffs"],
                 n=None,npad_lat=None,npad_lon=None):
    values = dict((var,[]) for var in var_names)

    for file_list in os.listdir(dirpath)[:n]:
        if file_list.split('.')[-1]=="nc":
            with nc.Dataset(dirpath +file_list, 'r') as data:
                for var in var_names:
                    aux = data[var][:]
                    
                    #don't use padded data to calculate std 
                    #NB: this assumes all variables are padded (ie gammas)
                    if npad_lat is not None:
                        aux=aux[npad_lat:-npad_lat,npad_lon:-npad_lon]
                    values[var] = np.concatenate((values[var],aux[np.nonzero(aux)]))
        else:
            print(file_list + " not considered")      
                
    mean = dict((var,np.mean(values[var])) for var in var_names)            
    std = dict((var,np.std(values[var])) for var in var_names)          
                 
    return std,mean 
def get_min_max(dirpath,var_names = ["alpha_i_minus","alpha_j_minus","w","norm_coeffs"],
                n=None,npad_lat=None,npad_lon=None):
    values = dict((var,[]) for var in var_names)
    for file_list in os.listdir(dirpath):
        if file_list.split('.')[-1]=="nc":
            with nc.Dataset(dirpath +file_list, 'r') as data:
                for var in var_names:
                    aux = data[var][:]
                    
                    #don't use padded data to calculate std 
                    #NB: this assumes all variables are padded (ie gammas)
                    if npad_lat is not None:
                        aux=aux[npad_lat:-npad_lat,npad_lon:-npad_lon]
                    
                    values[var] = np.concatenate((values[var],aux[np.nonzero(aux)]))
        else:
            print(file_list + " not considered")       
                
    mins = dict((var,np.min(values[var])) for var in var_names)            
    maxs = dict((var,np.max(values[var])) for var in var_names)          
                 
    return mins,maxs 
def pad_lon(X,pad):
    
    padded = np.empty((X.shape[0],X.shape[1],X.shape[2] + 2 * pad))
    padded[:,:,pad:-pad] = np.copy(X)
    padded[:,:,:pad] = np.copy(padded[:,:,-2*pad:-pad])
    padded[:,:,-pad:] = np.copy(padded[:,:,pad:2*pad])
    
    return padded

def pad_lat(X,pad):
    
    padded = np.empty((X.shape[0],X.shape[1] + 2 * pad,X.shape[2] ))
    padded[:,pad:-pad,:] = np.copy(X)


    padded[:,:pad,:] = np.flip(np.copy(padded[:,pad:2*pad,:]), axis=1)
    #padded[:,-pad:,:] = np.flip(np.roll(np.copy(padded[:,-2*pad:-pad,:]),180), axis=0)

    roll1 = np.flip(np.flip(np.roll(np.copy(padded[:,-2*pad:-pad,X.shape[2]//2:]),180), axis=1),axis=2)
    roll2 = np.flip(np.flip(np.roll(np.copy(padded[:,-2*pad:-pad,:X.shape[2]//2]),180), axis=1),axis=2)

    padded[:,-pad:,:X.shape[2]//2] = roll1
    padded[:,-pad:,X.shape[2]//2:] = roll2
    

    
    return padded

def get_standardized_data(data,var_names,dstd,dmean):
    """
    Parameters
    ----------
    data : inputs and outputs in netCDF format
    var_names : names of variables to be transformed.  

    Returns
    -------
    transformed array

    """

    if len(var_names)==1:
        arr = np.zeros((290+2*31,360+2*28))
        aux = data[var_names[0]][:]
        aux[np.nonzero(aux)] = (aux[np.nonzero(aux)] - 
                                dmean[var_names[0]])/ dstd[var_names[0]]
        arr = np.copy(aux)
    else:
        arr = np.zeros((len(var_names),290+2*31,360+2*28))
        for i,name in enumerate(var_names):
            aux = data[name][:]
            aux[np.nonzero(aux)] = (aux[np.nonzero(aux)] - dmean[name])/ dstd[name]
            arr[i,:,:] = aux
    return arr

def get_normalized_data(data,var_names,dmin,dmax):    
    """
    Parameters
    ----------
    data : inputs and outputs in netCDF format
    var_names : names of variables to be transformed.  

    Returns
    -------
    transformed array

    """

    if len(var_names)==1:
        arr = np.zeros((290+2*31,360+2*28))
        aux = data[var_names[0]][:]
        aux[np.nonzero(aux)] = (aux[np.nonzero(aux)] - 
                                dmin[var_names[0]])/ (dmax[var_names[0]] - 
                                                      dmin[var_names[0]])
        arr = np.copy(aux)
    else:
        arr = np.zeros((len(var_names),290+2*31,360+2*28))
        for i,name in enumerate(var_names):
            aux = data[name][:]
            aux[np.nonzero(aux)] = (aux[np.nonzero(aux)] - 
                                    dmin[name])/ (dmax[name] - 
                                                          dmin[name])
            arr[i,:,:] = aux
    return arr  

def augment_data(X,Y,data_augmentation):
    ret_X = []
    if data_augmentation=="hflip":
        pass
        