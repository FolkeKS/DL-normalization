#!/ usr / bin / env python3
#- * - coding : utf - 8 - * -
"""
Created on Tue Jul 12 14:58:10 2022

@author: coulaud
"""
import shutil
import numpy as np
import os
import ast

def addMap(X, dist_map):
    """Add the sign distance map to the tensor X

    Args:
        X (numpy array): three-channels  image
        dist_map (numpy array): sign distance map

    Returns:
        numpy array: four-channels image
    """    
    _,H,W  = X.shape

    Y    = np.empty((4,H,W))
    Y[0] = X[0,:,:]
    Y[1] = X[1,:,:]
    Y[2] = X[2,:,:]
    Y[3] = dist_map
    return Y    

def rota_90(X):
    _,H,W  = X.shape
      
    alphas_i = np.rot90(X[0,:,:], k=1, axes=(0,1))
    alphas_j = np.rot90(X[1,:,:], k=1, axes=(0,1))
    w        = np.rot90(X[2,:,:], k=1, axes=(0,1))
    dist_map = np.rot90(X[3,:,:], k=1, axes=(0,1))
    alphas_i = np.roll(alphas_i, -1, axis = 1)
    
    X    = np.empty((4,W,H))
    X[0] = alphas_j
    X[1] = alphas_i
    X[2] = w
    X[3] = dist_map
    return X

def rota_180(X):
    _,H,W  = X.shape
    
    alphas_i = np.rot90(X[0,:,:], k=2, axes=(0,1))
    alphas_j = np.rot90(X[1,:,:], k=2, axes=(0,1))
    w        = np.rot90(X[2,:,:], k=2, axes=(0,1))
    dist_map = np.rot90(X[3,:,:], k=2, axes=(0,1))
    alphas_j = np.roll(alphas_j, -1, axis = 1)
    alphas_i = np.roll(alphas_i, 1, axis = 0)
    
    X    = np.empty((4,H,W))
    X[0] = alphas_i
    X[1] = alphas_j
    X[2] = w
    X[3] = dist_map
    return X

def rota_270(X):
    _,H,W  = X.shape
        
    alphas_i = np.rot90(X[0,:,:], k=1, axes=(1,0))
    alphas_j = np.rot90(X[1,:,:], k=1, axes=(1,0))
    w        = np.rot90(X[2,:,:], k=1, axes=(1,0))
    dist_map = np.rot90(X[3,:,:], k=1, axes=(1,0))
    alphas_j = np.roll(alphas_j, 1, axis = 0)
    
    X    = np.empty((4,W,H))
    X[0] = alphas_j
    X[1] = alphas_i
    X[2] = w
    X[3] = dist_map
    return X

def flip_vert(X):
    _,H,W  = X.shape

    alphas_i = np.flipud(X[0,:,:])
    alphas_j = np.flipud(X[1,:,:])
    w        = np.flipud(X[2,:,:])
    dist_map = np.flipud(X[3,:,:])
    alphas_i = np.roll(alphas_i, 1, axis = 0)
    X    = np.empty((4,H,W))
    X[0] = alphas_i
    X[1] = alphas_j
    X[2] = w
    X[3] = dist_map
    return X

def flip_hor(X):
    _,H,W  = X.shape

    alphas_i = np.fliplr(X[0,:,:])
    alphas_j = np.fliplr(X[1,:,:])
    w        = np.fliplr(X[2,:,:])
    dist_map = np.fliplr(X[3,:,:])
    alphas_j = np.roll(alphas_j, -1, axis = 1)
    X    = np.empty((4,H,W))
    X[0] = alphas_i
    X[1] = alphas_j
    X[2] = w
    X[3] = dist_map
    return X

folder_name = "newdata"
data_dir = "data/processed/"+folder_name
dist_map = np.load("data/sign_dist_map_std.npz")['arr_0']
dst_dir = data_dir+"_augmented"
shutil.copytree(data_dir, dst_dir)

lines = open(data_dir+"/dict_std_mean.txt").readlines()
dict_std = ast.literal_eval(lines[0][:-1])
dict_mean = ast.literal_eval(lines[1])

fliphor  = True
flipvert = True
rot90    = True
rot180   = True
rot270   = True

print("Train")
for file in os.listdir(data_dir+"/train/X/"):
    if file.split('.')[-1]=="npz":

        fnameX = data_dir+"/train/X/"+file
        X = addMap(np.load(fnameX)['arr_0'], dist_map)

        saveX = dst_dir+"/train/X/"+file
        os.remove(saveX) 
        np.savez_compressed(saveX, X)
#rota 90
        if rot90:
            saveX = dst_dir+"/train/X/"+file.split('.')[0] + "_90" 
            Xstd = X
            Xstd[0] = X[0,:,:] * dict_std['alpha_i_minus'] + dict_mean['alpha_i_minus']
            Xstd[0,:,:] = (X[0,:,:] - dict_mean['alpha_j_minus']) / dict_std['alpha_j_minus']
            Xstd[1] = X[1,:,:] * dict_std['alpha_j_minus'] + dict_mean['alpha_j_minus']
            Xstd[1,:,:] = (X[1,:,:] - dict_mean['alpha_i_minus']) / dict_std['alpha_i_minus']
            np.savez_compressed(saveX, rota_90(Xstd))
#rota 180
        if rot180:
            saveX = dst_dir+"/train/X/"+file.split('.')[0] + "_180" 
            np.savez_compressed(saveX, rota_180(X))
#rota 270
        if rot270:
            saveX = dst_dir+"/train/X/"+file.split('.')[0] + "_270" 
            Xstd = X
            Xstd[0] = X[0,:,:] * dict_std['alpha_i_minus'] + dict_mean['alpha_i_minus']
            Xstd[0,:,:] = (X[0,:,:] - dict_mean['alpha_j_minus']) / dict_std['alpha_j_minus']
            Xstd[1] = X[1,:,:] * dict_std['alpha_j_minus'] + dict_mean['alpha_j_minus']
            Xstd[1,:,:] = (X[1,:,:] - dict_mean['alpha_i_minus']) / dict_std['alpha_i_minus']
            np.savez_compressed(saveX, rota_270(Xstd))
#flip hor
        if fliphor:
            saveX = dst_dir+"/train/X/"+file.split('.')[0] + "_flip_hor" 
            np.savez_compressed(saveX, flip_hor(X))
#flip vert
        if flipvert:
            saveX = dst_dir+"/train/X/"+file.split('.')[0] + "_flip_vert" 
            np.savez_compressed(saveX, flip_vert(X))

print("X done")


for file in os.listdir(data_dir+"/train/Y/"):
    if file.split('.')[-1]=="npz":

        fnameY = data_dir+"/train/Y/"+file
        Y = np.load(fnameY)['arr_0']

        saveY = dst_dir+"/train/Y/"+file
        os.remove(saveY) 
        np.savez_compressed(saveY, Y)
#rota 90
        if rot90:
            saveY = dst_dir+"/train/Y/"+file.split("_norm_coeffs")[0] + "_90_norm_coeffs"
            np.savez_compressed(saveY, np.rot90(Y, k=1, axes=(0,1)))
#rota 180
        if rot180:
            saveY = dst_dir+"/train/Y/"+file.split("_norm_coeffs")[0] + "_180_norm_coeffs"
            np.savez_compressed(saveY, np.rot90(Y, k=2, axes=(1,0)))
#rota 270
        if rot270:
            saveY = dst_dir+"/train/Y/"+file.split("_norm_coeffs")[0] + "_270_norm_coeffs"
            np.savez_compressed(saveY, np.rot90(Y, k=1, axes=(1,0)))
#flip hor
        if fliphor:
            saveY = dst_dir+"/train/Y/"+file.split("_norm_coeffs")[0] + "_flip_hor_norm_coeffs"
            np.savez_compressed(saveY, np.fliplr(Y))
#flip vert
        if flipvert:
            saveY = dst_dir+"/train/Y/"+file.split("_norm_coeffs")[0] + "_flip_vert_norm_coeffs"
            np.savez_compressed(saveY, np.flipud(Y))

print("Y done")

print("Validation")
for file in os.listdir(data_dir+"/valid/X/"):
    if file.split('.')[-1]=="npz":

        fnameX = data_dir+"/valid/X/"+file
        X = addMap(np.load(fnameX)['arr_0'], dist_map)

        saveX = dst_dir+"/valid/X/"+file
        os.remove(saveX) 
        np.savez_compressed(saveX, X)

print("X done")
for file in os.listdir(data_dir+"/valid/Y/"):
    if file.split('.')[-1]=="npz":

        fnameY = data_dir+"/valid/Y/"+file
        Y = np.load(fnameY)['arr_0']

        saveY = dst_dir+"/valid/Y/"+file
        os.remove(saveY) 
        np.savez_compressed(saveY, Y)
print("Y done")
