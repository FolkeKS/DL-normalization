#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:17:21 2022

@author: skrunes
"""

import sys
from src.features import get_std_mean, get_min_max,pad_lon,pad_lat, \
                         get_standardized_data,get_normalized_data,augment_data
import os
import netCDF4 as nc
import numpy as np
from argparse import ArgumentParser
import json

def main(data_path,n_train,n_valid, npad_lat,
                npad_lon,data_transformation,valid_path,data_augmentation):
    
    if n_train is not None:
        assert n_train + n_valid <=len(os.listdir(data_path)), f"Not enough data, reduce number of samples"
        save_path = "data/processed/nemo"+ data_path.split("/")[-2]+ str(n_train) +"_samples_" + data_transformation +"/"
    else:
        save_path = "data/processed/nemo"+ data_path.split("/")[-2]+ str(len(os.listdir(data_path))) +"_samples_" + data_transformation +"/"
    
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
        dstd,dmean= get_std_mean(data_path,["alpha_i_minus","alpha_j_minus","w","norm_coeffs"],n_train,npad_lat,npad_lon)
        
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
        if file_list.split('.')[-1]=="nc":
            with nc.Dataset(data_path +file_list, 'r') as data:
                
                if data_transformation=="standardize":
                    X=get_standardized_data(data,["alpha_i_minus","alpha_j_minus","w"],dstd,dmean)
                    Y=get_standardized_data(data,["norm_coeffs"],dstd,dmean)
                    
                    
                elif data_transformation=="normalize":
                    X=get_normalized_data(data,["alpha_i_minus","alpha_j_minus","w"],dmin,dmax)
                    Y=get_normalized_data(data,["norm_coeffs"],dmin,dmax)
    
                else:
                    raise NotImplementedError("available: standardize,normalize")
                
                assert X.shape == (3,290+2*npad_lat,360+2*npad_lon),f"{X.shape}"
                assert Y.shape == (290+2*npad_lat,360+2*npad_lon),f"{Y.shape}"

                #X[0,:,:] = pad_variable_lat(X,npad_lat)
                #X = pad_lon(X,npad_lon)

                
                fnameX = save_path + "train/X/"+ file_list.split('.')[0]
                fnameY = save_path + "train/Y/"+ file_list.split('.')[0] + "_norm_coeffs"
                np.savez_compressed(fnameX, X)
                np.savez_compressed(fnameY, Y)
        else:
            print(file_list + " not considered")
    
    # Save validation data
    if valid_path is None:
        for file_list in os.listdir(data_path)[n_train:n_valid+n_train]:
            if file_list.split('.')[-1]=="nc":
                with nc.Dataset(data_path +file_list, 'r') as data:
                    
                    if data_transformation=="standardize":
                        X=get_standardized_data(data,["alpha_i_minus","alpha_j_minus","w"],dstd,dmean)
                        Y=get_standardized_data(data,["norm_coeffs"],dstd,dmean)
                        
                    elif data_transformation=="normalize":
                        X=get_normalized_data(data,["alpha_i_minus","alpha_j_minus","w"],dmin,dmax)
                        Y=get_normalized_data(data,["norm_coeffs"],dmin,dmax)
                    else:
                        raise NotImplementedError("available: standardize,normalize")
                    
                    #X = pad_lat(X,npad_lat)
                    #X = pad_lon(X,npad_lon)
                    assert X.shape == (3,290+2*npad_lat,360+2*npad_lon),f"{X.shape}"
                    assert Y.shape == (290+2*npad_lat,360+2*npad_lon),f"{Y.shape}"
                    
                    fnameX = save_path + "valid/X/"+ file_list.split('.')[0]
                    fnameY = save_path + "valid/Y/"+ file_list.split('.')[0] + "_norm_coeffs"
                    np.savez_compressed(fnameX, X)
                    np.savez_compressed(fnameY, Y)
                
            else:
                print(file_list + " not considered")
    else:
        for file_list in os.listdir(valid_path):
            if file_list.split('.')[-1]=="nc":
                with nc.Dataset(valid_path +file_list, 'r') as data:
                    
                    if data_transformation=="standardize":
                        X=get_standardized_data(data,["alpha_i_minus","alpha_j_minus","w"],dstd,dmean)
                        Y=get_standardized_data(data,["norm_coeffs"],dstd,dmean)
                        
                    elif data_transformation=="normalize":
                        X=get_normalized_data(data,["alpha_i_minus","alpha_j_minus","w"],dmin,dmax)
                        Y=get_normalized_data(data,["norm_coeffs"],dmin,dmax)
                    else:
                        raise NotImplementedError("available: standardize,normalize")
                    
                    #X = pad_lat(X,npad_lat)
                    #X = pad_lon(X,npad_lon)
                    
                    fnameX = save_path + "valid/X/"+ file_list.split('.')[0]
                    fnameY = save_path + "valid/Y/"+ file_list.split('.')[0] + "_norm_coeffs"
                    np.savez_compressed(fnameX, X)
                    np.savez_compressed(fnameY, Y)
                
            else:
                print(file_list + " not considered")
                
                
    if data_augmentation is not None:
        for file_list in os.listdir(save_path+"train/X/"):
            if file_list.split('.')[-1]=="npz":
                
                fnameX = save_path+"train/X/"+file_list
                fnameY = save_path+"train/Y/"+file_list.split('.')[0]+ "_norm_coeffs.npz"
                X = np.load(fnameX)['arr_0']
                Y = np.load(fnameY)['arr_0']
                
                augmented_X,augmented_Y = augment_data(X,Y,data_augmentation)
                for i,(x,y) in enumerate(zip(augmented_X,augmented_Y)):
                    
                    fnamex = fnameX.split('.')[0] + "_" + str(i)
                    fnamey = save_path+"train/Y/"+file_list.split('.')[0]+  "_" + str(i)+"_norm_coeffs"
                    np.savez_compressed(fnamex, x)
                    np.savez_compressed(fnamey, y)
            
            else:
                print(file_list + " not considered")
        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/demo/isotropic_noise/")
    parser.add_argument("--n_train", type=int, default=None, help="Number of training samples")
    parser.add_argument("--n_valid", type=int, default=None, help="Number of validation samples")
    parser.add_argument("--npad_lat", type=int, default=31)
    parser.add_argument("--npad_lon", type=int, default=20)
    parser.add_argument("--data_transformation", type=str, default="standardize")
    parser.add_argument("--data_augmentation", type=str, default=None)
    parser.add_argument("--valid_path", type=str, default=None)
    

    args = parser.parse_args()

    main(args.data_path,args.n_train,args.n_valid,args.npad_lat,args.npad_lon,
         args.data_transformation,args.valid_path,args.data_augmentation)