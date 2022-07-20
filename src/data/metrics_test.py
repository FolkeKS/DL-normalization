import shutil
import numpy as np
import os
import numpy as np
import torch
from pytorch_lightning import Trainer,seed_everything,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import src.cnn as cnn
import src.cnn_map as cnn_map
import torchvision.transforms as transforms
import importlib
import os
import matplotlib.pyplot as plt

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
save_dir = "notebooks/aug_comparison/"
size_im = (70, 40)
f = open(data_dir+"norms_std_mean.txt")
lines = f.readlines()
assert len(lines) == 2, f"len {len(lines)}"
std = float(lines[0])
mean = float(lines[1])
f.close()

def transfrom(X):
    return transforms.CenterCrop([310, 380])(torch.from_numpy(X)).numpy()

def compute_eps(X,Y, model_path,std,mean,use_map=False,distance_map=None): 

    if use_map :
        X_map = np.empty((4,310,380))
        X_map[0:3,:,:] = X
        X_map[3,:,:] = distance_map
        X = X_map
        model = cnn_map.CNN.load_from_checkpoint(model_path)  
        
    else :
        model = cnn.CNN.load_from_checkpoint(model_path)   
    
    
    Y = transforms.CenterCrop([290, 360])(torch.from_numpy(Y)).numpy()
    X = transforms.CenterCrop([310, 380])(torch.from_numpy(X[np.newaxis,:]).float())
    mask = np.where(Y==0,True,False)
    Y = Y*std + mean
    Y_pred = model.forward(X).detach()*std + mean
    Y_pred =  transforms.CenterCrop([290, 360])(Y_pred).numpy()[0,0,:,:]

    Y2 = np.power(Y,2)
    eps = (np.power(Y_pred,2) - Y2)/Y2
    eps = np.ma.masked_array(eps, mask)

    return eps



model_path = "results/wandb/cnn/rot90/checkpoints/epoch=10570-val_loss=0.00076.ckpt"
distance_map_std = transfrom(np.load("data/sign_distance_map_std.npy"))

epsmean = np.empty((nb_samples,290,360))
meanmean = np.empty(nb_samples)
maxmean = np.empty(nb_samples)
quantmean = np.empty(nb_samples)
for i,file in enumerate(os.listdir("data/processed/nemo_tests/train/X/")):
    xname = "data/processed/nemo_tests/train/X/"+file
    yname = "data/processed/nemo_tests/train/Y/"+file.split('.')[0]+"_norm_coeffs.npz"
    X = transfrom(np.load(xname)['arr_0'])
    Y = transfrom(np.load(yname)['arr_0'])
    

    eps = compute_eps(X,Y,model_path,std,mean,True,distance_map_std)
    epsmean[i,:,:] = eps
    meanmean[i] = eps.mean()
    maxmean[i] = eps.max()
    quantmean[i] = np.quantile(eps[np.nonzero(eps)].flatten(), 0.9999)


mask = np.where(Y[10:-10,10:-10]==0,True,False)

fig,axs = plt.subplots(figsize=size_im,ncols=1)
c = axs.imshow(np.ma.masked_array(np.mean(epsmean,axis=0), mask),origin="lower",vmin=-0.05,vmax=0.05,cmap="seismic")
cbar_ax = fig.add_axes([0.266, 0.05, 0.4673, 0.02])
cb = plt.colorbar(c,cax=cbar_ax,orientation='horizontal')
cb.remove()
fig.savefig(save_dir+"rot90")

print("Mean:", np.mean(meanmean))
print("Max:", np.mean(maxmean))
print("Quant:", np.mean(quantmean))