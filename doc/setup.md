
# Table of Contents

1.  [Install the project](#org989aaa5)
2.  [Set up conda environment:](#orgdf21c8a)
    1.  [Create the environment (might be slow):](#org4efc28b)
    2.  [Activate the environment:](#orgdf6df53)
    3.  [Setup project:](#org0585ea2)
3.  [Set up wandb for experiment tracking:](#org22d4517)
4.  [Train a model :](#orgc583404)
    1.  [On a computer](#org735486a)
    2.  [On a cluster using SLURM](#org910c3a6)
    3.  [Configurations](#orgbe2f6cf)
5.  [Modify the code](#org59849fb)
6.  [Computing the metrics on the test data set](#orgd91dc75)
7.  [Compute the sign distance map](#orgfd17094)
8.  [Perform the data augmentation](#orgbe4142d)



<a id="org989aaa5"></a>

# Install the project

    git clone https://github.com/FolkeKS/DL-normalization.git


<a id="orgdf21c8a"></a>

# Set up conda environment:


<a id="org4efc28b"></a>

## Create the environment (might be slow):

    conda env create --file environment.yml

-   If the installation is stuck on `Solving environment: |` try:

    conda config --set channel_priority strict

-   Revert with:

    conda config - -set channel_priority true


<a id="orgdf6df53"></a>

## Activate the environment:

    bash conda activate DL-normalization


<a id="org0585ea2"></a>

## Setup project:

    bash pip install -e .


<a id="org22d4517"></a>

# Set up wandb for experiment tracking:

-   Sign up at <https://wandb.ai/site> and log in
-   Find your API-key at <https://wandb.ai/authorize>
-   With your conda environment activated, run the following command and provide API-key
    
        wandb login


<a id="orgc583404"></a>

# Train a model :


<a id="org735486a"></a>

## On a computer

    python scripts/trainer.py fit --config  configs/demo.yaml


<a id="org910c3a6"></a>

## On a cluster using SLURM

    sbatch scripts/train.bash


<a id="orgbe2f6cf"></a>

## Configurations

The configugration can be modified in the yaml files in <configs/>
The model part, shown below, describes the parameter to instantiate the class CNN in <src/cnn.py>.

    n_blocks: 4
    n_blocks_filters: 64
    layers_per_block: 2
    kernel_size: 3
    n_channels: 4
    n_classes: 1
    q: 0.9999
    standarize_outputs: true
    predict_squared: false
    predict_inverse: false
    loss_fn: masked_mse
    padding_type: "valid"
    optimizer: Adam
    data_dir: data/processed/sections/

This is the equivalent to

    class CNN(pl.LightningModule):
        def __init__(self,
            n_blocks: int = 4,
            n_blocks_filters: int = 64,
            layers_per_block: int = 2,
            kernel_size: int = 3,
            n_channels: int = 4,
            n_classes: int = 1,
            q: float = 0.9999,
            standarize_outputs: bool = False,
            predict_squared: bool = False,
            predict_inverse: bool = False,
            loss_fn: str = "masked_mse",
            padding_type: str = "valid",
            optimizer: str = "Adam",
            data_dir: str = "data/processed/newdata/",
            **kwargs):

The data part describes the parameter to instantiate the class DirLightDataset in <src/data/dataset.py>

    class DirLightDataset(pl.LightningDataModule):
        def __init__(self,
            batch_size: int = 1,
            data_dir: str = "data/processed/newdata/",
            num_workers: int = 0,
            gpus: int = 0):

-   In order to make deterministic runs we set put `seed_everything: 42` (line 1) and `deterministic: true` (line 78).
-   To continue a run after it has been stopped we set  `ckpt_path: "results/wandb/cnn/36c5vu01/checkpoints/last.ckpt"` (line 115). With `"cnn/"` the wandb project in which the run was created, and `"36c5vu01"` the &rsquo;id&rsquo; of the run. We also set `version: 36c5vu01` (line 11) to make wandb continue the training in the run instead of creating a new run. The `global_step` variable will be reset, so the charts must be visualized with `epoch` or `step` as x-axis.
-   It is also possible to choose how the best checkpoint is selected (lines 20-35), we choose the checkpoint that minimizes the loss in validation, but other choices are possible as choosing the one minimizing the quantile. However, we have not found how to use two strategies for the same run.


<a id="org59849fb"></a>

# Modify the code

<a id="orge9d26f1"></a>
The model is loaded in scripts/trainer.py. Modifications are to be done in the config file but there are also modifications to make in the module loading the data in <src/data/dataset.py>:

-   adjust the padding, by default the padding is 31 for the latitude and 28 for the longitude. The images are cropped on the fly, the dimension of the image taken by the CNN is $4 \times 292+2\ell \times 360 + 2\ell$ with $\ell$ the number of layers
-   adding the sign distance map

For computational time, these modifications should be made to the data directly before starting the training.

    class DirDataset(Dataset):
    def __getitem__(self, i):
        idx = self.ids[i]
        X_files = glob.glob(os.path.join(self.X_dir, idx+'.*'))
        Y_files = glob.glob(os.path.join(self.Y_dir, idx+'_norm_coeffs.*'))
    #Load the input / true data
        X = torch.from_numpy(np.load(X_files[0])['arr_0']).float()
        Y = torch.from_numpy(np.load(Y_files[0])['arr_0']).float()
    #Load the distance map
        distance_map = np.load("data/python_sign_dist_map_std.npz")['arr_0']
        distance_map = torch.from_numpy(distance_map).float()
    #Crop the input data
        distance_map = transforms.CenterCrop([200, 360+2*10])(distance_map)
        X = transforms.CenterCrop([200, 360+2*10])(X)
    #Add the distance map
        X = torch.cat((X,torch.unsqueeze(distance_map, 0)),0)
        return X, \
            Y


<a id="orgd91dc75"></a>

# Computing the metrics on the test data set

As explained in the report, we used one data set for the python data two data sets **Partial std = finaldata** and **Full std = newdata** for the Nemovar data. However, for the **Partial std** data set the training data were standardized using 10 samples while the test data were standardized using 180 samples. So, to compute the metrics the test data had to be destandardized and then restandardized. For **Full std** all the data are standarized using the 10 training samples.
We compute the mean and standard deviation of the mean/max/quantile 99,99% of the absolute relative error over the test dataset. We save the tensor of the relative error for each sample and save an image of the mean of the relative error. The code and results are in the repository <results/test_metrics/> each repository corresponds to a data set. To run the computation of the metrics for the root directory of the project the python script. The configuration of the runs is added to a list `model_params`, each element of this list is a list with the index corresponding to:

0.  Boolean: compute or not the metrics
1.  Boolean: if the run uses the sign distance map
2.  Numpy array: the sign distance map
3.  Int: number of layers
4.  Class: the model loaded with the correct src


<a id="orgfd17094"></a>

# Compute the sign distance map

The sign distance map is computed in the notebook <notebooks/Distance_map.ipynb>. We use the method `scipy.ndimage.distance_transform_bf` to compute the distance [(doc link)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_bf.html).


<a id="orgbe4142d"></a>

# Perform the data augmentation

The script to use is <src/data/augmentation.py> and the combinations of transformations are tested in the notebook <notebooks/test_augmentation.ipynb>. As the sign distance map is added to the input data during the augmentation, it is important to not add it an other time as described in Section [5](#orge9d26f1).

