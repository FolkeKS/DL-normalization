import shutil
import os
import numpy as np
import torch

import results.test_metrics.models.block as block

import torchvision.transforms as transforms
import importlib
import os
import matplotlib.pyplot as plt
import time
import ast
from results.test_metrics.compute import *

start_time = time.time()


nb_samples=len(os.listdir("data/processed/isotropic_noise10_samples_standardize/valid/X/"))
print(nb_samples, "samples")

data_dir = "data/processed/isotropic_noise10_samples_standardize/"
save_dir = "results/test_metrics/python/"
size_im = (70, 40)

def transfromX(X, n_layers, use_map, distance_map):
    if use_map :
        distance_map = transforms.CenterCrop([180+2*n_layers, 360+2*n_layers])(torch.from_numpy(distance_map)).numpy()
        X = transforms.CenterCrop([180+2*n_layers, 360+2*n_layers])(torch.from_numpy(X)).numpy()
        X_map = np.empty((4,200,380))
        X_map[0:3,:,:] = X
        X_map[3,:,:] = distance_map
        X = X_map
    return transforms.CenterCrop([180+2*n_layers, 360+2*n_layers])(torch.from_numpy(X)).numpy()

def transfromY(Y):
    return transforms.CenterCrop([180, 360])(torch.from_numpy(Y)).numpy()

def load_X_Y(file, n_layers, use_map, distance_map):
    xname = "data/processed/isotropic_noise10_samples_standardize/valid/X/"+file
    yname = "data/processed/isotropic_noise10_samples_standardize/valid/Y/"+file.split('.')[0]+"_norm_coeffs.npz"
    X = transfromX(np.load(xname)['arr_0'],n_layers,use_map,distance_map)
    Y = transfromY(np.load(yname)['arr_0'])
    return X,Y

def compute_eps(X,Y,std_test, mean_test, model):

    X = torch.from_numpy(X[np.newaxis,:]).float()
    mask = np.where(Y==0,True,False)

    Y = Y*std_test + mean_test
    Y_pred = model.forward(X).detach()*std_test + mean_test
    Y_pred = transforms.CenterCrop([180, 360])(Y_pred).numpy()[0,0,:,:]
    Y2 = np.power(Y,2)
    eps = (np.power(Y_pred,2) - Y2)/Y2

    eps = np.ma.masked_array(eps, mask)
    return eps

distance_map_std = np.load("data/python_sign_dist_map_std.npz")['arr_0']
exps = []
model_params = []

### 10 skip co v2
exps.append("10_eps")
model_path ="results/wandb/cnn/python/36c5vu01/checkpoints/epoch=29271-val_loss=0.72680.ckpt"
model_params.append([True,True,distance_map_std,10, block.CNN.load_from_checkpoint(model_path) ])


f = open(data_dir+"dict_std_mean.txt")
lines = f.readlines()
assert len(lines) == 2, f"len {len(lines)}"
dict_std_test = ast.literal_eval(lines[0][:-1])
dict_mean_test = ast.literal_eval(lines[1])
f.close()


f = open(save_dir+'metrics.txt', 'w')
for exp, model_param in zip(exps,model_params):
    alleps = np.empty((nb_samples,180,360))
    mean_eps = np.empty(nb_samples)
    max_eps = np.empty(nb_samples)
    quant_eps = np.empty(nb_samples)
    assert len(model_param) == 5, f"len(model_param) == {len(model_param)} != 5"
    if not model_param[0]:
        continue
    use_map = model_param[1]
    if use_map:
        distance_map = model_param[2]
    else:
        distance_map = None
    n_layers = model_param[3]
    model = model_param[4]
    print("Experiment: ", exp)

    avr_time = 0
    for i,file in enumerate(os.listdir(data_dir+"/valid/X/")):
        X,Y = load_X_Y(file,n_layers,use_map,distance_map)
        eps_tick = time.time()
        eps = compute_eps(X,Y,dict_std_test['norm_coeffs'],dict_mean_test['norm_coeffs'],model)
        eps_tack = time.time()
        alleps[i,:,:] = eps
        mean_eps[i]  = np.abs(eps).mean()
        max_eps[i]   = np.abs(eps).max()
        quant_eps[i] = np.quantile(np.abs(eps[np.nonzero(eps)].flatten()), 0.9999)
        avr_time += eps_tack - eps_tick

    
    display_esp_map(size_im,alleps,np.where(Y==0 ,True,False),save_dir,exp)
    print_results(avr_time,nb_samples,quant_eps,max_eps,mean_eps)
    write_results(f, exp, quant_eps, max_eps, mean_eps)
    np.savez_compressed(save_dir+"eps"+exp, alleps)




print("--- %s seconds ---" % (time.time() - start_time))