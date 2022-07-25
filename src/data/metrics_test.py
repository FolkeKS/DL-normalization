import shutil
import numpy as np
import os
import numpy as np
import torch
from pytorch_lightning import Trainer,seed_everything,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import src.cnn_map as cnn_map
import src.cnn_map_leaky as cnn_map_leaky
import src.cnn_map_leaky_02 as cnn_map_leaky_02

import src.cnn_block as cnn_block
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


if not os.path.isdir("data/processed/nemo_tests"): 
    shutil.copytree("data/processed/nemo_bnd_fixed/", "data/processed/nemo_tests")

    for F in ("X/", "Y/"):
        list10 = os.listdir("data/processed/nemonemo_bnd_perten10_samples_standardize/train/"+F)
        list20 = os.listdir("data/processed/nemonemo_bnd_perten20_samples_standardize/train/"+F)

        for file in os.listdir("data/processed/nemo_tests/train/"+F):
            if file.split('.')[-1]=="npz":
                if file in list10 or file in list20:
                    os.remove("data/processed/nemo_tests/train/"+F+file)

                    print(file)
assert len(os.listdir("data/processed/nemo_tests/train/X/")) == len(os.listdir("data/processed/nemo_tests/train/X/")), f"Size pb"

nb_samples=len(os.listdir("data/processed/nemo_tests/train/X/"))
print(nb_samples, "samples")

data_dir = "data/processed/nemo_tests/"
save_dir = "notebooks/comparison/"
size_im = (70, 40)


def transfrom(X):
    return transforms.CenterCrop([310, 380])(torch.from_numpy(X)).numpy()

def load_X_Y(file):
    xname = "data/processed/nemo_tests/train/X/"+file
    yname = "data/processed/nemo_tests/train/Y/"+file.split('.')[0]+"_norm_coeffs.npz"
    X = transfrom(np.load(xname)['arr_0'])
    Y = transfrom(np.load(yname)['arr_0'])
    return X,Y

def get_model( model_path, use_map, distance_map, use_block, leaky, leaky_val, v2_block): 
    if use_map :
        if use_block:
            if v2_block:
                print("Model: cnn_map_block_v2")
                model = cnn_map_block_v2.CNN.load_from_checkpoint(model_path)
            elif not v2_block:
                if leaky:
                    print("Model: cnn_map_block_leaky")
                    model = cnn_map_block_leaky.CNN.load_from_checkpoint(model_path) 
                else:
                    print("Model: cnn_map_block")
                    model = cnn_map_block.CNN.load_from_checkpoint(model_path)  

        elif leaky: 
            if leaky_val == 0.1:
                print("Model: cnn_map_leaky")
                model = cnn_map_leaky.CNN.load_from_checkpoint(model_path)  
            elif leaky_val == 0.2:
                print("Model: cnn_map_leaky_02")
                model = cnn_map_leaky_02.CNN.load_from_checkpoint(model_path)  
        else:
            print("Model: cnn_map")
            model = cnn_map.CNN.load_from_checkpoint(model_path)  
    elif use_block:
        print("Model: cnn_block")
        model = cnn_block.CNN.load_from_checkpoint(model_path)  

    return model

def compute_eps(X,Y,std_test, mean_test,std_train, mean_train, model, use_map):
    if use_map :
        X_map = np.empty((4,310,380))
        X_map[0:3,:,:] = X
        X_map[3,:,:] = distance_map
        X = X_map
        
    Y = transforms.CenterCrop([290, 360])(torch.from_numpy(Y)).numpy()
    X = transforms.CenterCrop([310, 380])(torch.from_numpy(X[np.newaxis,:]).float())
    mask = np.where(Y==0,True,False)


    Y = Y*std_test + mean_test
    Y_pred = model.forward(X).detach()*std_train + mean_train
    Y_pred = transforms.CenterCrop([290, 360])(Y_pred).numpy()[0,0,:,:]

    Y2 = np.power(Y,2)
    eps = (np.power(Y_pred,2) - Y2)/Y2
    eps = np.ma.masked_array(eps, mask)

    return eps


distance_map_std = transfrom(np.load("data/sign_distance_map_std.npy"))
model_paths = []
exps = []
model_params = []

### 10
exps.append("10_1")
model_paths.append("results/wandb/cnn/10_1/checkpoints/epoch=47888-val_loss=0.00001.ckpt")
model_params.append([True,distance_map_std,False,False,None,False, "nemonemo_bnd_perten10_samples_standardize"])
### 10 + rot_90
exps.append("10_rot90")
model_paths.append("results/wandb/cnn/rot90/checkpoints/epoch=10570-val_loss=0.00076.ckpt")
model_params.append([True,distance_map_std,False,False,None,False,"nemonemo_bnd_perten10_samples_standardize_augmented_rot90"])
### 10 + fliplr
exps.append("10_fliplr")
model_paths.append("results/wandb/cnn/flip_hor_10/checkpoints/epoch=23421-val_loss=0.00001.ckpt")
model_params.append([True,distance_map_std,False,False,None,False,"nemonemo_bnd_perten10_samples_standardize"])
### 20
exps.append("20_1")
model_paths.append("results/wandb/cnn/20_1/checkpoints/epoch=24910-val_loss=0.00001.ckpt")
model_params.append([True,distance_map_std,False,False,None,False,"nemonemo_bnd_perten20_samples_standardize"])
### 20 LeakyReLU 0.1
exps.append("20_1_leaky_0_1")
model_paths.append("results/wandb/cnn/20_leaky/checkpoints/epoch=24962-val_loss=0.00001.ckpt")
model_params.append([True,distance_map_std,False,True,0.1,False,"nemonemo_bnd_perten20_samples_standardize"])
### 20 LeakyReLU 0.2
exps.append("20_1_leaky_0_2")
model_paths.append("results/wandb/cnn/20_leaky_02/checkpoints/epoch=24675-val_loss=0.00001.ckpt")
model_params.append([True,distance_map_std,False,True,0.2,False,"nemonemo_bnd_perten20_samples_standardize"])
### Skip co no map
exps.append("skip_co")
model_paths.append("results/wandb/cnn/skip_co/checkpoints/epoch=24400-val_loss=0.60961.ckpt")
model_params.append([False,None,True,False,None,False,"nemonemo_bnd_perten20_samples_standardize"])
### Skip co map
exps.append("skip_co_map")
model_paths.append("results/wandb/cnn/skip_co_map/checkpoints/epoch=23951-val_loss=0.60961.ckpt")
model_params.append([True,distance_map_std,True,False,None,False,"nemonemo_bnd_perten20_samples_standardize"])
### Skip co V2 map
exps.append("skip_co_v2_map")
model_paths.append("results/wandb/cnn/block_v2/checkpoints/epoch=24505-val_loss=0.60961.ckpt")
model_params.append([True,distance_map_std,True,False,None,True,"nemonemo_bnd_perten20_samples_standardize"])
### Skip co map leaky
exps.append("skip_co_map_leaky")
model_paths.append("results/wandb/cnn/skip_co_map_leaky/checkpoints/epoch=11690-val_loss=0.60962.ckpt")
model_params.append([True,distance_map_std,True,True,None,False,"nemonemo_bnd_perten20_samples_standardize"])


f = open(data_dir+"dict_std_mean.txt")
lines = f.readlines()
assert len(lines) == 2, f"len {len(lines)}"
dict_std_test = ast.literal_eval(lines[0][:-1])
dict_mean_test = ast.literal_eval(lines[1])
f.close()


f = open(save_dir+'metrics.txt', 'w')
for exp, model_path, model_param in zip(exps, model_paths,model_params):

    epsmean = np.empty((nb_samples,290,360))
    meanmean = np.empty(nb_samples)
    maxmean = np.empty(nb_samples)
    quantmean = np.empty(nb_samples)
    assert len(model_param) == 7, f"len(model_param) == {len(model_param)} != 7"

    use_map = model_param[0]
    distance_map = model_param[1]
    use_block = model_param[2]
    use_leaky = model_param[3]
    leaky_val = model_param[4]
    v2_block = model_param[5] 
    data_train = model_param[6]

    print("Experiment: ", exp)
    print("Training folder: ", model_param[6])
    model = get_model(model_path,use_map,distance_map,use_block,use_leaky,leaky_val,v2_block)

    lines = open("data/processed/"+data_train+"/dict_std_mean.txt").readlines()
    dict_std_train = ast.literal_eval(lines[0][:-1])
    dict_mean_train = ast.literal_eval(lines[1])
    for i,file in enumerate(os.listdir("data/processed/nemo_tests/train/X/")):
        X,Y = load_X_Y(file)
        ####
        # De standardize
        ###
        X[0,:,:] = X[0,:,:] * dict_std_test['alpha_i_minus'] + dict_mean_test['alpha_i_minus']
        X[1,:,:] = X[1,:,:] * dict_std_test['alpha_j_minus'] + dict_mean_test['alpha_j_minus']
        X[2,:,:] = X[2,:,:] * dict_std_test['w'] + dict_mean_test['w']
        ###
        # Re standardize
        ###
        X[0,:,:] = (X[0,:,:] - dict_mean_train['alpha_i_minus']) / dict_std_train['alpha_i_minus'] 
        X[1,:,:] = (X[1,:,:] - dict_mean_train['alpha_j_minus']) / dict_std_train['alpha_j_minus'] 
        X[2,:,:] = (X[2,:,:] - dict_mean_train['w']) / dict_std_train['w'] 


        eps = compute_eps(X,Y,dict_std_test['norm_coeffs'],dict_mean_test['norm_coeffs'],dict_std_train['norm_coeffs'],dict_mean_train['norm_coeffs'],model,use_map)
        epsmean[i,:,:] = eps
        meanmean[i]  = eps.mean()
        maxmean[i]   = eps.max()
        quantmean[i] = np.quantile(eps[np.nonzero(eps)].flatten(), 0.9999)


    mask = np.where(Y[10:-10,10:-10]==0,True,False)

    fig,axs = plt.subplots(figsize=size_im,ncols=1)
    c = axs.imshow(np.ma.masked_array(np.mean(epsmean,axis=0), mask),origin="lower",vmin=-0.05,vmax=0.05,cmap="seismic")
    cbar_ax = fig.add_axes([0.266, 0.05, 0.4673, 0.02])
    cb = plt.colorbar(c,cax=cbar_ax,orientation='horizontal')
    cb.remove()
    fig.savefig(save_dir+exp)
    print(np.mean(quantmean))
    print(np.mean(maxmean))
    print(np.mean(meanmean))
    print()

    f.write(exp + "\n")

    f.write("- Quant: " + str(np.mean(quantmean)) + "\n")
    f.write("- Max:   " + str(np.mean(maxmean)) + "\n")
    f.write("- Mean:  " + str(np.mean(meanmean)) + "\n\n")
    

print("--- %s seconds ---" % (time.time() - start_time))