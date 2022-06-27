import numpy as np
import os
import os.path
import src.visualization.display as dsp
import netCDF4 as nc
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer,seed_everything,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import src.unet as unet
import src.cnn as cnn
import scipy.ndimage.morphology as scm
import importlib
import matplotlib.ticker as mtick
importlib.reload(dsp)



class Plots:
    def __init__(self, model_path, data_path, model_arch):
        self.model = model_path
        self.data_path = data_path
        if model_arch == "cnn":
            self.model=cnn.CNN.load_from_checkpoint(model_path)
        elif model_arch == "unet":
            self.model=unet.Unet.load_from_checkpoint(model_path)
        else :
            raise NotImplementedError(model_arch+ " not implemented")
        f = open(data_path+"norms_std_mean.txt")
        lines = f.readlines()
        assert len(lines) == 2, f"len {len(lines)}"
        self.std = float(lines[0])
        self.mean = float(lines[1])
        f.close()

    def train_bias(self):
        eps = np.empty((len(os.listdir(self.data_path+"train/X/")),160,360))
        for i,file_list in enumerate(os.listdir(self.data_path+"train/X/")):
            X=np.load(self.data_path+"train/X/"+file_list)['arr_0']
            X=torch.from_numpy(X[np.newaxis,:]).float()
            X = X[:,:,10:-10,:]
            Y=np.load(self.data_path+"train/Y/" + file_list.split(".")[0]+"_norm_coeffs.npz")['arr_0']
            if i==0:
                mask = np.where(Y[10:-10,:]==0,True,False) 
            Y = Y[10:-10,:]*self.std + self.mean
            Y_pred = self.model.forward(X).detach().numpy()[0,0,:,44:-44]*self.std + self.mean
            aux = (Y_pred**2 - Y**2)/Y**2
            eps[i,:,:] = np.ma.masked_array(aux, mask)
        lon = np.arange(-180, 180)
        lat = np.arange(-80, 80)
        lon2d, lat2d = np.meshgrid(lon, lat)
        dsp.plot_error_globe(np.ma.masked_array(np.mean(eps,axis=0), mask), lon2d, lat2d,vmin=-0.01,vmax=0.01,cmap="RdYlBu",
                            cmap_title="($\hat{\gamma}^2 - \gamma^2)/\gamma^2$",title="train bias")
        print("mean: ",np.mean(np.abs(np.ma.masked_array(np.mean(eps,axis=0), mask))))
        print("max: ",np.max(np.abs(np.ma.masked_array(np.mean(eps,axis=0), mask))))


    def validation_bias(self):
        eps = np.empty((len(os.listdir(self.data_path+"train/X/")),160,360))
        for i,file_list in enumerate(os.listdir(self.data_path+"train/X/")):
            X=np.load(self.data_path+"train/X/"+file_list)['arr_0']
            X=torch.from_numpy(X[np.newaxis,:]).float()
            X = X[:,:,10:-10,:]
            Y=np.load(self.data_path+"train/Y/" + file_list.split(".")[0]+"_norm_coeffs.npz")['arr_0']
            if i==0:
                mask = np.where(Y[10:-10,:]==0,True,False) 
            Y = Y[10:-10,:]*self.std + self.mean
            Y_pred = self.model.forward(X).detach().numpy()[0,0,:,44:-44]*self.std + self.mean
            aux = (Y_pred**2 - Y**2)/Y**2
            eps[i,:,:] = np.ma.masked_array(aux, mask)
        lon = np.arange(-180, 180)
        lat = np.arange(-80, 80)
        lon2d, lat2d = np.meshgrid(lon, lat)
        dsp.plot_error_globe(np.ma.masked_array(np.mean(eps,axis=0), mask), lon2d, lat2d,vmin=-0.01,vmax=0.01,cmap="RdYlBu",
                            cmap_title="($\hat{\gamma}^2 - \gamma^2)/\gamma^2$",title="validation bias")
        print("mean: ",np.mean(np.abs(np.ma.masked_array(np.mean(eps,axis=0), mask))))
        print("max: ",np.max(np.abs(np.ma.masked_array(np.mean(eps,axis=0), mask))))


    def validation_var(self):
        eps = np.empty((len(os.listdir(self.data_path+"train/X/")),160,360))
        for i,file_list in enumerate(os.listdir(self.data_path+"train/X/")):
            X=np.load(self.data_path+"train/X/"+file_list)['arr_0']
            X=torch.from_numpy(X[np.newaxis,:]).float()
            X = X[:,:,10:-10,:]
            Y=np.load(self.data_path+"train/Y/" + file_list.split(".")[0]+"_norm_coeffs.npz")['arr_0']
            if i==0:
                mask = np.where(Y[10:-10,:]==0,True,False) 
            Y = Y[10:-10,:]*self.std + self.mean
            Y_pred = self.model.forward(X).detach().numpy()[0,0,:,44:-44]*self.std + self.mean
            aux = (Y_pred**2 - Y**2)/Y**2
            eps[i,:,:] = np.ma.masked_array(aux, mask)
        lon = np.arange(-180, 180)
        lat = np.arange(-80, 80)
        lon2d, lat2d = np.meshgrid(lon, lat)
        dsp.plot_error_globe(np.ma.masked_array(np.sum((eps - np.mean(eps,axis=0))**2, axis=0)/10, mask), lon2d, lat2d,vmin=0,vmax=0.0002,cmap="YlOrBr",
                            cmap_title="($\hat{\gamma}^2 - \gamma^2)/\gamma^2$",title="validation variance")
        print("mean: ",np.mean(np.ma.masked_array(np.sum((eps - np.mean(eps,axis=0))**2, axis=0)/10, mask)))
        print("max:  ",np.max(np.ma.masked_array(np.sum((eps - np.mean(eps,axis=0))**2, axis=0)/10, mask)) )
        print("min:  ",np.min(np.ma.masked_array(np.sum((eps - np.mean(eps,axis=0))**2, axis=0)/10, mask)) )