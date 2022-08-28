import shutil
import os
import numpy as np
import torch
from pytorch_lightning import Trainer,seed_everything,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import results.test_metrics.models.cnn as cnn
import results.test_metrics.models.cnn_map as cnn_map
import results.test_metrics.models.cnn_map_ELU as cnn_map_ELU
import results.test_metrics.models.cnn_map_PLReLU as cnn_map_PLReLU

import results.test_metrics.models.cnn_map_softplus as cnn_map_softplus
import results.test_metrics.models.cnn_map_softplus96 as cnn_map_softplus96

import results.test_metrics.models.cnn_map_leaky as cnn_map_leaky
import results.test_metrics.models.cnn_map_leaky_02 as cnn_map_leaky_02

import results.test_metrics.models.cnn_block as cnn_block
import results.test_metrics.models.cnn_map_block16 as cnn_map_block16
import results.test_metrics.models.cnn_map_block128 as cnn_map_block128
import results.test_metrics.models.cnn_map_block_v2 as cnn_map_block_v2

import results.test_metrics.models.cnn_map_block as cnn_map_block
import results.test_metrics.models.cnn_map_block_leaky as cnn_map_block_leaky

import torchvision.transforms as transforms
import importlib
import os
import matplotlib.pyplot as plt
import time
import ast
from results.test_metrics.compute import *


start_time = time.time()


if not os.path.isdir("data/processed/final_nemo_tests"): 
    shutil.copytree("data/processed/nemo_bnd_perten_final190_samples_standardize/", "data/processed/final_nemo_tests")

    for step in ("X/", "Y/"):
        list10_train = os.listdir("data/processed/nemonemo_bnd_perten_final10_samples_standardize/train/"+step)
        # list20 = os.listdir("data/processed/nemonemo_bnd_perten20_samples_standardize/train/"+F)

        for file in os.listdir("data/processed/final_nemo_tests/train/"+step):
            if file.split('.')[-1]=="npz":
                if file in list10_train:
                    os.remove("data/processed/final_nemo_tests/train/"+F+file)

                    print(file)
assert len(os.listdir("data/processed/final_nemo_tests/train/X/")) == len(os.listdir("data/processed/final_nemo_tests/train/X/")), f"Size pb"

nb_samples=len(os.listdir("data/processed/final_nemo_tests/train/X/"))
print(nb_samples, "samples")

data_dir = "data/processed/final_nemo_tests/"
save_dir = save_dir = "results/test_metrics/final_partial/"

size_im = (70, 40)






distance_map_std = np.load("data/sign_dist_map_std.npz")['arr_0']
distance_map_std_eucl = np.load("data/sign_dist_map_std_eucl.npz")['arr_0']
data_dir_10 =  "nemonemo_bnd_perten_final10_samples_standardize"

exps = []
model_params = []
### 10 no dist
exps.append("10_no_map")
model_path = "results/wandb/cnn/final/1ztarux8/checkpoints/epoch=42429-val_loss=0.00002.ckpt"
model_params.append([True,False,None,10,cnn.CNN.load_from_checkpoint(model_path)])
### 10 cityblock
exps.append("10_cityblock")
model_path = "results/wandb/cnn/final/2kdh1hkl/checkpoints/epoch=49778-val_loss=0.00002.ckpt"
model_params.append([True,True,distance_map_std,10,cnn_map.CNN.load_from_checkpoint(model_path)])
### 10 euclidean
exps.append("10_euclidean")
model_path = "results/wandb/cnn/final/29vt1p98/checkpoints/epoch=41315-val_loss=0.00002.ckpt"
model_params.append([True,True,distance_map_std,10,cnn_map.CNN.load_from_checkpoint(model_path)])

### 10 flip hor
exps.append("10_1_flip")
model_path = "results/wandb/cnn/final/21inteu3/checkpoints/epoch=15273-val_loss=0.00000.ckpt"
model_params.append([True,True,distance_map_std,10,cnn_map_ELU.CNN.load_from_checkpoint(model_path)])
### 10 ELU
exps.append("10_cityblock_ELU")
model_path = "results/wandb/cnn/final/2qrrcxdy/checkpoints/epoch=49984-val_loss=0.00001.ckpt"
model_params.append([True,True,distance_map_std,10,cnn_map_ELU.CNN.load_from_checkpoint(model_path)])
### 10 Softplus
exps.append("10_cityblock_softplus")
model_path = "results/wandb/cnn/final/2mpjgxqm/checkpoints/epoch=41043-val_loss=0.00000.ckpt"
model_params.append([True,True,distance_map_std,10,cnn_map_softplus.CNN.load_from_checkpoint(model_path)])
### 10 PLReLU
exps.append("PLReLU")
model_path = "results/wandb/cnn/final/2o6hcc2z/checkpoints/epoch=49361-val_loss=0.00001.ckpt"
model_params.append([True,True,distance_map_std,10,cnn_map_PLReLU.CNN.load_from_checkpoint(model_path)])
### 10 softplus + 96 channels
exps.append("10_cityblock_softplus96")
model_path = "results/wandb/cnn/final/m672pcsq/checkpoints/epoch=37894-val_loss=0.00000.ckpt"
model_params.append([True,True,distance_map_std,10,cnn_map_softplus96.CNN.load_from_checkpoint(model_path)])
### Skip  co + EULU
exps.append("10_skip_co_elu")
model_path = "results/wandb/cnn/final/24buv9cn/checkpoints/epoch=49763-val_loss=0.61642.ckpt"
model_params.append([True,True,distance_map_std,10,cnn_map_block.CNN.load_from_checkpoint(model_path)])
### Skip  co + EUU + 128 ch
exps.append("10_skip_co_elu_128")
model_path = "results/wandb/cnn/final/pn0eig6k/checkpoints/epoch=48924-val_loss=0.61642.ckpt"
model_params.append([True,True,distance_map_std,10,cnn_map_block128.CNN.load_from_checkpoint(model_path)])
### Skip  co + EULU + 16 l
exps.append("10_skip_co_elu_16")
model_path = "results/wandb/cnn/final/9eheeyt0/checkpoints/epoch=49779-val_loss=0.61642.ckpt"
model_params.append([True,True,distance_map_std,16,cnn_map_block16.CNN.load_from_checkpoint(model_path)])

f = open(data_dir+"dict_std_mean.txt")
lines = f.readlines()
assert len(lines) == 2, f"len {len(lines)}"
dict_std_test = ast.literal_eval(lines[0][:-1])
dict_mean_test = ast.literal_eval(lines[1])
f.close()


f = open(save_dir+'metrics.txt', 'w')
for exp, model_param in zip(exps, model_params):
    alleps = np.empty((nb_samples,292,360))
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
    lines = open("data/processed/nemonemo_bnd_perten_final10_samples_standardize/dict_std_mean.txt").readlines()
    dict_std_train = ast.literal_eval(lines[0][:-1])
    dict_mean_train = ast.literal_eval(lines[1])
    avr_time = 0
    for i,file in enumerate(os.listdir("data/processed/final_nemo_tests/train/X/")):
        X,Y = load_X_Y("final_nemo_tests","train",file,n_layers,use_map,distance_map)
        ####
        # De standardize % test data
        ###
        X[0,:,:] = X[0,:,:] * dict_std_test['alpha_i_minus'] + dict_mean_test['alpha_i_minus']
        X[1,:,:] = X[1,:,:] * dict_std_test['alpha_j_minus'] + dict_mean_test['alpha_j_minus']
        X[2,:,:] = X[2,:,:] * dict_std_test['w'] + dict_mean_test['w']
        ###
        # Re standardize % train data
        ###
        X[0,:,:] = (X[0,:,:] - dict_mean_train['alpha_i_minus']) / dict_std_train['alpha_i_minus'] 
        X[1,:,:] = (X[1,:,:] - dict_mean_train['alpha_j_minus']) / dict_std_train['alpha_j_minus'] 
        X[2,:,:] = (X[2,:,:] - dict_mean_train['w']) / dict_std_train['w'] 
        eps_tick = time.time()
        eps = compute_eps_restd(X,Y,dict_std_test['norm_coeffs'],dict_mean_test['norm_coeffs'],dict_std_train['norm_coeffs'],dict_mean_train['norm_coeffs'],model)
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