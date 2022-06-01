#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:15:07 2022

@author: skrunes
"""
import sys
from src.features import get_std_mean, get_min_max,pad_data,get_standardized_data,get_normalized_data
import os
import netCDF4 as nc
import numpy as np
from argparse import ArgumentParser
import json

def main(data_path,n_train,n_valid,pad,data_transformation):

    assert n_train + n_valid <=len(os.listdir(data_path)), f"Not enough data, reduce number of samples"
    save_path = "data/processed/"+ data_path.split("/")[-2]+ str(n_train) +"_samples_" + data_transformation +"/"
    
    
    # Check whether the specified path exists or not
    assert os.path.exists(save_path)==False, save_path + " is not empty"

    os.makedirs(save_path)
    
    os.makedirs(save_path+ "train")
    os.makedirs(save_path+ "train/X/")
    os.makedirs(save_path+ "train/Y/")
    
    os.makedirs(save_path+ "valid")
    os.makedirs(save_path+ "valid/X/")
    os.makedirs(save_path+ "valid/Y/")
    
    if data_transformation=="standardize":
        dstd,dmean= get_std_mean(data_path,["alpha_i_minus","alpha_j_minus","w","norm_coeffs"],n_train)
        
        #Save norm_coeff std and mean for future use
        f = open(save_path +"norms_std_mean.txt","w")
        f.write(str(dstd["norm_coeffs"]))
        f.write("\n")
        f.write(str(dmean["norm_coeffs"]))
        f.close()
        
        f = open(save_path +"dict_std_mean.txt","w")
        f.write(str(dstd))
        f.write("\n")
        f.write(str(dmean))
        f.close()
    elif data_transformation=="normalize":
        dmin,dmax = get_min_max(data_path,["alpha_i_minus","alpha_j_minus","w","norm_coeffs"],n_train)
        
        #Save norm_coeff std and mean for future use
        f = open(save_path +"norms_min_max.txt","w")
        f.write(str(dmin["norm_coeffs"]))
        f.write("\n")
        f.write(str(dmax["norm_coeffs"]))
        f.close()
        
        f = open(save_path +"dict_min_max.txt","w")
        f.write(str(dmin))
        f.write("\n")
        f.write(str(dmax))
        f.close()
    
    
    # Save train data
    
    for file_list in os.listdir(data_path)[:n_train]:
        with nc.Dataset(data_path +file_list, 'r') as data:
            
            if data_transformation=="standardize":
                X=get_standardized_data(data,["alpha_i_minus","alpha_j_minus","w"],dstd,dmean)
                Y=get_standardized_data(data,["norm_coeffs"],dstd,dmean)
                
                
            elif data_transformation=="normalize":
                X=get_normalized_data(data,["alpha_i_minus","alpha_j_minus","w"],dmin,dmax)
                Y=get_normalized_data(data,["norm_coeffs"],dmin,dmax)

            else:
                raise NotImplementedError("available: standardize,normalize")
            
            X = pad_data(X,pad)
            
            fnameX = save_path + "train/X/"+ file_list.split('.')[0]
            fnameY = save_path + "train/Y/"+ file_list.split('.')[0] + "_norm_coeffs"
            np.savez_compressed(fnameX, X)
            np.savez_compressed(fnameY, Y)
     
    # Save validation data
    
    for file_list in os.listdir(data_path)[n_train:n_valid+n_train]:
        with nc.Dataset(data_path +file_list, 'r') as data:
            
            if data_transformation=="standardize":
                X=get_standardized_data(data,["alpha_i_minus","alpha_j_minus","w"],dstd,dmean)
                Y=get_standardized_data(data,["norm_coeffs"],dstd,dmean)
                
            elif data_transformation=="normalize":
                X=get_normalized_data(data,["alpha_i_minus","alpha_j_minus","w"],dmin,dmax)
                Y=get_normalized_data(data,["norm_coeffs"],dmin,dmax)
            else:
                raise NotImplementedError("available: standardize,normalize")
            
            X = pad_data(X,pad)
            
            fnameX = save_path + "valid/X/"+ file_list.split('.')[0]
            fnameY = save_path + "valid/Y/"+ file_list.split('.')[0] + "_norm_coeffs"
            np.savez_compressed(fnameX, X)
            np.savez_compressed(fnameY, Y)
            
        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/demo/isotropic_noise/")
    parser.add_argument("--n_train", type=int, default=None, help="Number of training samples")
    parser.add_argument("--n_valid", type=int, default=None, help="Number of validation samples")
    parser.add_argument("--pad", type=int, default=40)
    parser.add_argument("--data_transformation", type=str, default="standardize")
    

    args = parser.parse_args()

    main(args.data_path,args.n_train,args.n_valid,args.pad,args.data_transformation)