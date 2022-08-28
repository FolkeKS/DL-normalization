import os
import glob

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torchvision.transforms as transforms


class DirDataset(Dataset):
    def __init__(self, X_dir, Y_dir):
        self.X_dir = X_dir
        self.Y_dir = Y_dir
        try:
            self.ids = [s.split('.')[0] for s in os.listdir(self.X_dir)]
        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        X_files = glob.glob(os.path.join(self.X_dir, idx+'.*'))
        Y_files = glob.glob(os.path.join(self.Y_dir, idx+'_norm_coeffs.*'))
        assert len(X_files) == 1, f'{idx}: {X_files}'
        assert len(Y_files) == 1, f'{idx}: {Y_files}'
        # Load the input/true data 
        X = torch.from_numpy(np.load(X_files[0])['arr_0']).float()
        Y = torch.from_numpy(np.load(Y_files[0])['arr_0']).float()
        #  Load the distance map
        distance_map = np.load("data/python_sign_dist_map_std.npz")['arr_0']
        distance_map = torch.from_numpy(distance_map).float()
        # Crop the input data 
        distance_map = transforms.CenterCrop([200, 360+2*10])(distance_map)
        X = transforms.CenterCrop([200, 360+2*10])(X)
        # Add the distance map        
        X = torch.cat((X,torch.unsqueeze(distance_map, 0)),0)

        return X, \
            Y


class DirLightDataset(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, data_dir: str = "data/processed/demo_isotropic_noise3_samples_normalize/",
                 num_workers: int = 0, gpus: int = 0):  # test_dir = "globe_test"):
        """
        batch_size:int=32

        data_dir:str="flat_polecontinent3"

        num_workers:int=8

        gpus:int=0
        """

        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.gpus = gpus
        #self.test_dir = test_dir
        #self.dataset  = DirDataset(f'./dataset/{self.data_dir}/train', f'./dataset/{self.data_dir}/train_norms')

    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        pass
        #self.dataset  = DirDataset(f'./dataset/{self.data_dir}/train', f'./dataset/{self.data_dir}/train_norms')

    def setup(self, stage=None):
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.

        self.train_ds = DirDataset(
            self.data_dir + "/train/X", self.data_dir + "/train/Y")
        self.val_ds = DirDataset(
            self.data_dir + "/valid/X", self.data_dir + "/valid/Y")
        #train_loader = DataLoader(train_ds, num_workers=8,batch_size=32, pin_memory=True, shuffle=True)

    def train_dataloader(self):

        train_loader = DataLoader(self.train_ds,
                                  batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers,
                                  pin_memory=(self.gpus != 0),
                                  persistent_workers=(self.num_workers != 0))  # disable persistant workers if 0 workers
        return train_loader

    def val_dataloader(self):

        valid_loader = DataLoader(self.val_ds,
                                  batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers,
                                  pin_memory=(self.gpus != 0),
                                  persistent_workers=(self.num_workers != 0))  # disable persistant workers if 0 workers)
        return valid_loader