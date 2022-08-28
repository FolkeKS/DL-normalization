import numpy as np
import torch
from pytorch_lightning import Trainer,seed_everything,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchvision.transforms as transforms
import importlib
import matplotlib.pyplot as plt



def transfromX(X, n_layers, use_map, distance_map):
    if use_map :
        X_map = np.empty((4,354,416))
        X_map[0:3,:,:] = X
        X_map[3,:,:] = distance_map
        X = X_map
    return transforms.CenterCrop([292+2*n_layers, 360+2*n_layers])(torch.from_numpy(X)).numpy()

def transfromY(Y):
    return transforms.CenterCrop([292, 360])(torch.from_numpy(Y)).numpy()

def load_X_Y(folder, step,file, n_layers, use_map, distance_map):
    xname = "data/processed/"+folder+"/"+step+"/X/"+file
    yname = "data/processed/"+folder+"/"+step+"/Y/"+file.split('.')[0]+"_norm_coeffs.npz"
    X = transfromX(np.load(xname)['arr_0'],n_layers,use_map,distance_map)
    Y = transfromY(np.load(yname)['arr_0'])
    return X,Y

def compute_eps(X,Y,std, mean, model):

    X = torch.from_numpy(X[np.newaxis,:]).float()
    mask = np.where(Y==0,True,False)

    Y = Y*std + mean
    Y_pred = model.forward(X).detach()*std + mean
    Y_pred = transforms.CenterCrop([292, 360])(Y_pred).numpy()[0,0,:,:]

    Y2 = np.power(Y,2)
    eps = (np.power(Y_pred,2) - Y2)/Y2
    eps = np.ma.masked_array(eps, mask)

    return eps

def compute_eps_restd(X,Y,std_test, mean_test,std_train, mean_train, model):
    X = torch.from_numpy(X[np.newaxis,:]).float()
    mask = np.where(Y==0,True,False)

    Y = Y*std_test + mean_test
    Y_pred = model.forward(X).detach()*std_train + mean_train
    Y_pred = transforms.CenterCrop([292, 360])(Y_pred).numpy()[0,0,:,:]

    Y2 = np.power(Y,2)
    eps = (np.power(Y_pred,2) - Y2)/Y2
    eps = np.ma.masked_array(eps, mask)

    return eps


def print_results(avr_time,nb_samples,quant_eps,max_eps,mean_eps):
    print("---  avr time : ",format(avr_time/nb_samples, ".2f"), "  ---")
    print(np.mean(quant_eps), np.std(quant_eps))
    print(np.mean(max_eps), np.std(max_eps))
    print(np.mean(mean_eps), np.std(mean_eps))
    print()

def write_results(f, exp, quant_eps, max_eps, mean_eps):
    f.write(exp + "\n")

    f.write("- Quant: " + str(np.mean(quant_eps))+ ", " + str(np.std(quant_eps)) + "\n")
    f.write("- Max:   " + str(np.mean(max_eps))+ ", " + str(np.std(max_eps)) + "\n")
    f.write("- Mean:  " + str(np.mean(mean_eps))+ ", " + str(np.std(mean_eps)) + "\n\n")

def display_esp_map(size_im,alleps,mask,save_dir,exp):
    fig,axs = plt.subplots(figsize=size_im,ncols=1)
    c = axs.imshow(np.ma.masked_array(np.mean(alleps,axis=0), mask),origin="lower",vmin=-0.05,vmax=0.05,cmap="seismic")
    cbar_ax = fig.add_axes([0.266, 0.05, 0.4673, 0.02])
    cb = plt.colorbar(c,cax=cbar_ax,orientation='horizontal')
    cb.remove()
    fig.savefig(save_dir+exp, bbox_inches='tight')