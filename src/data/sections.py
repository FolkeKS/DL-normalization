import os
from re import S
import shutil
import numpy as np
import torch
import torchvision.transforms as transforms

def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def crop(X, i, j, n_layers, use_map=0, dist_map=None):
    D,_,_ = X.shape
    Y = np.empty((D+use_map,1+n_layers*2,1+n_layers*2))
    for d in range(D):
        Y[d] = X[d, i-n_layers:i+1+n_layers, j-n_layers:j+1+n_layers]
    if use_map == 1:
        Y[D] = dist_map[i-n_layers:i+1+n_layers, j-n_layers:j+1+n_layers]
    return Y

def load_X_Y(file, data_dir,step):
    xname = data_dir+step+"/X/"+file
    yname = data_dir+step+"/Y/"+file.split('.')[0]+"_norm_coeffs.npz"
    X = np.load(xname)['arr_0']
    Y = np.load(yname)['arr_0']
    return X,Y

def addMap(X, n_layers, distance_map):
    D,H,W = X.shape
    X_map = np.empty((D+1,H, W))
    X_map[0:3,:,:] = X
    X_map[3,:,:] = distance_map
    X = X_map
    return transforms.CenterCrop([292+2*n_layers, 360+2*n_layers])(torch.from_numpy(X)).numpy()


pad_lat = 31
pad_lon = 20
n_layers = 10
dist_map = np.load("data/sign_dist_map.npz")['arr_0']
data_dir = "data/processed/newdata/"
save_dir = "data/processed/sections/"

if not os.path.isdir(save_dir): 
    shutil.copytree(data_dir,save_dir,ignore=ig_f)
    src_files = os.listdir(data_dir)
    for file_name in src_files:
        full_file_name = os.path.join(data_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, save_dir)

    Yp = np.empty((1,1,1))
    for step in (["train","valid"]):
        print(step)
        if step == "train":
            for file in os.listdir(data_dir+step+"/X/"):
                X,Y = load_X_Y(file,data_dir,step)
                mask = np.where(Y==0,True,False)
                for i in range(pad_lon+1, 292+1):   
                    for j in range(pad_lat+1, 360+1):
                        if not mask[i,j]:
                            xname = save_dir+step+"/X/"+file.split('.')[0]+"_"+str(i)+"_"+str(j)
                            yname = save_dir+step+"/Y/"+file.split('.')[0]+"_"+str(i)+"_"+str(j)+"_norm_coeffs"
                            np.savez_compressed(xname, crop(X,i,j,n_layers,1,dist_map))
                            Yp[0,:,:] = Y[i,j]
                            np.savez_compressed(yname, Yp)
                break
        else:
            for file in os.listdir(data_dir+step+"/X/"):
                X,Y = load_X_Y(file,data_dir,step)
                xname = save_dir+step+"/X/"+file.split('.')[0]
                yname = save_dir+step+"/Y/"+file.split('.')[0]+"_norm_coeffs"
                np.savez_compressed(xname, addMap(X,10,dist_map))
                np.savez_compressed(yname, Yp)
                break