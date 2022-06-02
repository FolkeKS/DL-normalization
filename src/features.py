#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:38:40 2022

@author: skrunes
"""

import numpy as np
import os
import netCDF4 as nc

def get_std_mean(dirpath,var_names = ["alpha_i_minus","alpha_j_minus","w","norm_coeffs"],n=None):
    values = dict((var,[]) for var in var_names)

    for file_list in os.listdir(dirpath)[:n]:
        if file_list.split('.')[-1]=="nc":
            with nc.Dataset(dirpath +file_list, 'r') as data:
                for var in var_names:
                    aux = data[var][:]
                    values[var] = np.concatenate((values[var],aux[np.nonzero(aux)]))
        else:
            print(file_list + " not considered")      
                
    mean = dict((var,np.mean(values[var])) for var in var_names)            
    std = dict((var,np.std(values[var])) for var in var_names)          
                 
    return std,mean 
def get_min_max(dirpath,var_names = ["alpha_i_minus","alpha_j_minus","w","norm_coeffs"],n=None):
    values = dict((var,[]) for var in var_names)
    for file_list in os.listdir(dirpath):
        if file_list.split('.')[-1]=="nc":
            with nc.Dataset(dirpath +file_list, 'r') as data:
                for var in var_names:
                    aux = data[var][:]
                    values[var] = np.concatenate((values[var],aux[np.nonzero(aux)]))
        else:
            print(file_list + " not considered")       
                
    mins = dict((var,np.min(values[var])) for var in var_names)            
    maxs = dict((var,np.max(values[var])) for var in var_names)          
                 
    return mins,maxs 
def pad_data(X,pad):
    
    padded = np.empty((X.shape[0],X.shape[1],X.shape[2] + 2 * pad))
    padded[:,:,pad:-pad] = np.copy(X)
    padded[:,:,:pad] = np.copy(X[:,:,-2*pad:-pad])
    padded[:,:,-pad:] = np.copy(X[:,:,pad:2*pad])
    
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
        arr = np.zeros((180,360))
        aux = data[var_names[0]][:]
        aux[np.nonzero(aux)] = (aux[np.nonzero(aux)] - 
                                dmean[var_names[0]])/ dstd[var_names[0]]
        arr = np.copy(aux)
    else:
        arr = np.zeros((len(var_names),180,360))
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
        arr = np.zeros((180,360))
        aux = data[var_names[0]][:]
        aux[np.nonzero(aux)] = (aux[np.nonzero(aux)] - 
                                dmin[var_names[0]])/ (dmax[var_names[0]] - 
                                                      dmin[var_names[0]])
        arr = np.copy(aux)
    else:
        arr = np.zeros((len(var_names),180,360))
        for i,name in enumerate(var_names):
            aux = data[name][:]
            aux[np.nonzero(aux)] = (aux[np.nonzero(aux)] - 
                                    dmin[name])/ (dmax[name] - 
                                                          dmin[name])
            arr[i,:,:] = aux
    return arr  