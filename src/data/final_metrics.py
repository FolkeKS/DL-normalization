import shutil
import os
import numpy as np
import torch
from pytorch_lightning import Trainer,seed_everything,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import src.cnn as cnn
import src.cnn_map as cnn_map
import src.cnn_map_ELU as cnn_map_ELU
import src.cnn_map_PLReLU as cnn_map_PLReLU

import src.cnn_map_softplus as cnn_map_softplus
import src.cnn_map_softplus96 as cnn_map_softplus96

import src.cnn_map_leaky as cnn_map_leaky
import src.cnn_map_leaky_02 as cnn_map_leaky_02

import src.cnn_block as cnn_block
import src.cnn_map_block16 as cnn_map_block16
import src.cnn_map_block128 as cnn_map_block128
import src.cnn_map_block_v2 as cnn_map_block_v2

import src.cnn_map_block as cnn_map_block
import src.cnn_map_block_leaky as cnn_map_block_leaky

import torchvision.transforms as transforms
import importlib
import os
import matplotlib.pyplot as plt
import time
import ast
start_time = time.time()


if not os.path.isdir("data/processed/final_nemo_tests"): 
    shutil.copytree("data/processed/nemo_bnd_perten_final190_samples_standardize/", "data/processed/final_nemo_tests")

    for F in ("X/", "Y/"):
        list10_train = os.listdir("data/processed/nemonemo_bnd_perten_final10_samples_standardize/train/"+F)
        list10_valid = os.listdir("data/processed/nemonemo_bnd_perten_final10_samples_standardize/valid/"+F)

        # list20 = os.listdir("data/processed/nemonemo_bnd_perten20_samples_standardize/train/"+F)

        for file in os.listdir("data/processed/final_nemo_tests/train/"+F):
            if file.split('.')[-1]=="npz":
                if file in list10_train or file in list10_valid:# or file in list20:
                    os.remove("data/processed/final_nemo_tests/train/"+F+file)

                    print(file)
assert len(os.listdir("data/processed/final_nemo_tests/train/X/")) == len(os.listdir("data/processed/final_nemo_tests/train/X/")), f"Size pb"

nb_samples=len(os.listdir("data/processed/final_nemo_tests/train/X/"))
print(nb_samples, "samples")

data_dir = "data/processed/final_nemo_tests/"
save_dir = "notebooks/final/comparison/"
size_im = (70, 40)

def transfromX(X, n_layers, use_map, distance_map):
    if use_map :
        X_map = np.empty((4,354,416))
        X_map[0:3,:,:] = X
        X_map[3,:,:] = distance_map
        X = X_map
    return transforms.CenterCrop([292+2*n_layers, 360+2*n_layers])(torch.from_numpy(X)).numpy()
def transfromY(Y):
    return transforms.CenterCrop([292, 360])(torch.from_numpy(Y)).numpy()

def load_X_Y(file, n_layers, use_map, distance_map):
    xname = "data/processed/final_nemo_tests/train/X/"+file
    yname = "data/processed/final_nemo_tests/train/Y/"+file.split('.')[0]+"_norm_coeffs.npz"
    X = transfromX(np.load(xname)['arr_0'],n_layers,use_map,distance_map)
    Y = transfromY(np.load(yname)['arr_0'])
    return X,Y

def get_model( model_path, use_map, use_block, v2_block, acti): 
    if use_map :
        if use_block:
            if v2_block:
                print("Model: cnn_map_block_v2")
                model = cnn_map_block_v2.CNN.load_from_checkpoint(model_path)
            elif not v2_block:
                if acti == "elu128":
                    print("Model: cnn_map_block ELU128")
                    model = cnn_map_block128.CNN.load_from_checkpoint(model_path)  
                elif acti == "elu16":
                    print("Model: cnn_map_block ELU16")
                    model = cnn_map_block16.CNN.load_from_checkpoint(model_path)  
                else:
                    print("Model: cnn_map_block ELU")
                    model = cnn_map_block.CNN.load_from_checkpoint(model_path)  
        elif acti == "ELU":
                print("Model: cnn_map_ELU")
                model = cnn_map_ELU.CNN.load_from_checkpoint(model_path) 
        elif acti == "softplus":
                print("Model: cnn_map_softplus")
                model = cnn_map_softplus.CNN.load_from_checkpoint(model_path)
        elif acti == "softplus96":
                print("Model: cnn_map_softplus96")
                model = cnn_map_softplus96.CNN.load_from_checkpoint(model_path)        
        elif acti == "PLReLU":
                print("Model: cnn_map_PLReLU")
                model = cnn_map_PLReLU.CNN.load_from_checkpoint(model_path)                
        else:
            print("Model: cnn_map")
            model = cnn_map.CNN.load_from_checkpoint(model_path)  
    elif use_block:
        print("Model: cnn_block")
        model = cnn_block.CNN.load_from_checkpoint(model_path)  
    else:
        print("Model: cnn")
        model = cnn.CNN.load_from_checkpoint(model_path)  
    return model

def compute_eps(X,Y,std_test, mean_test,std_train, mean_train, model, use_map):


    X = torch.from_numpy(X[np.newaxis,:]).float()
    mask = np.where(Y==0,True,False)


    Y = Y*std_test + mean_test
    Y_pred = model.forward(X).detach()*std_train + mean_train
    Y_pred = transforms.CenterCrop([292, 360])(Y_pred).numpy()[0,0,:,:]

    Y2 = np.power(Y,2)
    eps = (np.power(Y_pred,2) - Y2)/Y2
    eps = np.ma.masked_array(eps, mask)

    return eps

distance_map_std = np.load("data/sign_dist_map_std.npz")['arr_0']
distance_map_std_eucl = np.load("data/sign_dist_map_std_eucl.npz")['arr_0']
data_dir_10 =  "nemonemo_bnd_perten_final10_samples_standardize"


model_paths = []
exps = []
model_params = []
### 10 no dist
exps.append("10_no_map")
model_paths.append("results/wandb/cnn/final/1ztarux8/checkpoints/epoch=42429-val_loss=0.00002.ckpt")
model_params.append([True,False,None,False,False,data_dir_10,10,None])
### 10 cityblock
exps.append("10_cityblock")
model_paths.append("results/wandb/cnn/final/2kdh1hkl/checkpoints/epoch=49778-val_loss=0.00002.ckpt")
model_params.append([True,True,distance_map_std,False,False,data_dir_10,10,None])
### 10 euclidean
exps.append("10_euclidean")
model_paths.append("results/wandb/cnn/final/29vt1p98/checkpoints/epoch=41315-val_loss=0.00002.ckpt")
model_params.append([True,True,distance_map_std_eucl,False,False,data_dir_10,10,None])
# ### 10 flip vert
# exps.append("10_1")
# model_paths.append("results/wandb/cnn/final")
# model_params.append([True,distance_map_std_eucl,False,False,None,False,data_dir_10,10])
### 10 flip hor
exps.append("10_1")
model_paths.append("results/wandb/cnn/final/21inteu3/checkpoints/epoch=15273-val_loss=0.00000.ckpt")
model_params.append([True,True,distance_map_std,False,False,data_dir_10,10,"ELU"])
### 10 ELU
exps.append("10_cityblock_ELU")
model_paths.append("results/wandb/cnn/final/2qrrcxdy/checkpoints/epoch=49984-val_loss=0.00001.ckpt")
model_params.append([True,True,distance_map_std,False,False,data_dir_10,10,"ELU"])
### 10 Softplus
exps.append("10_cityblock_softplus")
model_paths.append("results/wandb/cnn/final/2mpjgxqm/checkpoints/epoch=41043-val_loss=0.00000.ckpt")
model_params.append([True,True,distance_map_std,False,False,data_dir_10,10,"softplus"])
### 10 PLReLU
exps.append("PLReLU")
model_paths.append("results/wandb/cnn/final/2o6hcc2z/checkpoints/epoch=49361-val_loss=0.00001.ckpt")
model_params.append([True,True,distance_map_std,False,False,data_dir_10,10,"PLReLU"])
### 10 softplus + 96 channels
exps.append("10_cityblock_softplus96")
model_paths.append("results/wandb/cnn/final/m672pcsq/checkpoints/epoch=37894-val_loss=0.00000.ckpt")
model_params.append([True,True,distance_map_std,False,False,data_dir_10,10,"softplus96"])
### Skip  co + EULU
exps.append("10_skip_co_elu")
model_paths.append("results/wandb/cnn/final/24buv9cn/checkpoints/epoch=49763-val_loss=0.61642.ckpt")
model_params.append([True,True,distance_map_std,True,False,data_dir_10,10,None])
### Skip  co + EULU 128
exps.append("10_skip_co_elu_128")
model_paths.append("results/wandb/cnn/final/pn0eig6k/checkpoints/epoch=48924-val_loss=0.61642.ckpt")
model_params.append([True,True,distance_map_std,True,False,data_dir_10,10,"elu128"])
### Skip  co + EULU 16
exps.append("10_skip_co_elu_16")
model_paths.append("results/wandb/cnn/final/9eheeyt0/checkpoints/epoch=49779-val_loss=0.61642.ckpt")
model_params.append([True,True,distance_map_std,True,False,data_dir_10,16,"elu16"])

f = open(data_dir+"dict_std_mean.txt")
lines = f.readlines()
assert len(lines) == 2, f"len {len(lines)}"
dict_std_test = ast.literal_eval(lines[0][:-1])
dict_mean_test = ast.literal_eval(lines[1])
f.close()


f = open(save_dir+'metrics.txt', 'w')
for exp, model_path, model_param in zip(exps, model_paths,model_params):
    alleps = np.empty((nb_samples,292,360))
    meanmean = np.empty(nb_samples)
    maxmean = np.empty(nb_samples)
    quantmean = np.empty(nb_samples)
    assert len(model_param) == 8, f"len(model_param) == {len(model_param)} != 8"
    if not model_param[0]:
        continue
    use_map = model_param[1]
    if use_map:
        distance_map = model_param[2]
    else:
        distance_map = None
    use_block = model_param[3]
    v2_block = model_param[4] 
    data_train = model_param[5]
    n_layers = model_param[6]
    fct_acti = model_param[7]
    print("Experiment: ", exp)
    print("Training folder: ", data_train)
    model = get_model(model_path,use_map,use_block,v2_block,fct_acti)
    lines = open("data/processed/"+data_train+"/dict_std_mean.txt").readlines()
    dict_std_train = ast.literal_eval(lines[0][:-1])
    dict_mean_train = ast.literal_eval(lines[1])
    avr_time = 0
    for i,file in enumerate(os.listdir("data/processed/final_nemo_tests/train/X/")):
        X,Y = load_X_Y(file,n_layers,use_map,distance_map)
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
        eps = compute_eps(X,Y,dict_std_test['norm_coeffs'],dict_mean_test['norm_coeffs'],dict_std_train['norm_coeffs'],dict_mean_train['norm_coeffs'],model,use_map)
        eps_tack = time.time()
        alleps[i,:,:] = eps
        meanmean[i]  = np.abs(eps).mean()
        maxmean[i]   = np.abs(eps).max()
        quantmean[i] = np.quantile(np.abs(eps[np.nonzero(eps)].flatten()), 0.9999)
        avr_time += eps_tack - eps_tick

    mask = np.where(Y==0,True,False)

    fig,axs = plt.subplots(figsize=size_im,ncols=1)
    c = axs.imshow(np.ma.masked_array(np.mean(alleps,axis=0), mask),origin="lower",vmin=-0.05,vmax=0.05,cmap="seismic")
    cbar_ax = fig.add_axes([0.266, 0.05, 0.4673, 0.02])
    cb = plt.colorbar(c,cax=cbar_ax,orientation='horizontal')
    cb.remove()
    fig.savefig(save_dir+exp, bbox_inches='tight')
    print("---  avr time : ",format(avr_time/nb_samples, ".2f"), "  ---")
    print(np.mean(quantmean), np.std(quantmean))
    print(np.mean(maxmean), np.std(maxmean))
    print(np.mean(meanmean), np.std(meanmean))
    print()

    f.write(exp + "\n")

    f.write("- Quant: " + str(np.mean(quantmean)) + "\n")
    f.write("- Max:   " + str(np.mean(maxmean)) + "\n")
    f.write("- Mean:  " + str(np.mean(meanmean)) + "\n\n")
    np.savez_compressed(save_dir+"eps"+exp, alleps) 




print("--- %s seconds ---" % (time.time() - start_time))