
# Initial model + ELU
import results.test_metrics.models.cnn_map_ELU as cnn_map_ELU
# Skip conections (1st architecture) 10/16/20 layers + ELU
import results.test_metrics.models.cnn_map_block as cnn_map_block
import results.test_metrics.models.cnn_map_block16 as cnn_map_block16
import results.test_metrics.models.cnn_map_block20 as cnn_map_block20
# Skip connnections (2nd architecture, resnet-like) + ELU
import results.test_metrics.models.block as block

import importlib
import os
import matplotlib.pyplot as plt
import time
import ast
from results.test_metrics.compute import *

start_time = time.time()


nb_samples=len(os.listdir("data/processed/newdata/valid/X/"))
print(nb_samples, "samples")

data_dir = "data/processed/newdata/"
save_dir = "results/test_metrics/newdata_full/"
size_im = (70, 40)



distance_map_std = np.load("data/sign_dist_map_std.npz")['arr_0']
distance_map_std_eucl = np.load("data/sign_dist_map_std_eucl.npz")['arr_0']


exps = []
model_params = []
### 10 skip co ELU eps
exps.append("10_eps_skipco_ELU")
model_path ="results/wandb/cnn/newdata/16ktmc02/checkpoints/epoch=49270-val_loss=0.61641.ckpt"
model_params.append([True,True,distance_map_std,10, cnn_map_block.CNN.load_from_checkpoint(model_path) ])

### 10 skip co ELU mse ga
exps.append("10_gam_skipco_ELU")
model_path ="results/wandb/cnn/newdata/2q9wmsu0/checkpoints/epoch=49933-val_loss=0.00000.ckpt"
model_params.append([True,True,distance_map_std,10, cnn_map_block.CNN.load_from_checkpoint(model_path) ])

### 10 skip co + flip hor
exps.append("10_eps_skipco_flip_hor")
model_path ="results/wandb/cnn/newdata/cif3vbsw/checkpoints/epoch=23989-val_loss=0.61641.ckpt"
model_params.append([True,True,distance_map_std,10, cnn_map_block.CNN.load_from_checkpoint(model_path) ])


### 10 skip co ELU eps 16l
exps.append("10_eps_skipco_ELU_16l")
model_path ="results/wandb/cnn/newdata/1ypk0h7s/checkpoints/epoch=48900-val_loss=0.61641.ckpt"
model_params.append([True,True,distance_map_std,16, cnn_map_block16.CNN.load_from_checkpoint(model_path) ])

### 10 skip co ELU eps Z0l
exps.append("10_eps_skipco_ELU_20l")
model_path ="results/wandb/cnn/newdata/2w3wp1xs/checkpoints/epoch=33891-val_loss=0.61642.ckpt"
model_params.append([True,True,distance_map_std,20, cnn_map_block20.CNN.load_from_checkpoint(model_path) ])

### 10 skip co v2
exps.append("10_eps_skipco_v2")
model_path ="results/wandb/cnn/newdata/yyyre7s5/checkpoints/epoch=49303-val_loss=0.61641.ckpt"
model_params.append([True,True,distance_map_std,10, block.CNN.load_from_checkpoint(model_path) ])

### 10 ELU eps
exps.append("10_eps_ELU")
model_path ="results/wandb/cnn/newdata/qct996h0/checkpoints/epoch=49104-val_loss=0.61641.ckpt"
model_params.append([True,True,distance_map_std,10, cnn_map_ELU.CNN.load_from_checkpoint(model_path) ])

### 10 ELU mse gam
exps.append("10_gam_ELU")
model_path ="results/wandb/cnn/newdata/34lncdun/checkpoints/epoch=46812-val_loss=0.00000.ckpt"
model_params.append([True,True,distance_map_std,10, cnn_map_ELU.CNN.load_from_checkpoint(model_path) ])

f = open(data_dir+"dict_std_mean.txt")
lines = f.readlines()
assert len(lines) == 2, f"len {len(lines)}"
dict_std_test = ast.literal_eval(lines[0][:-1])
dict_mean_test = ast.literal_eval(lines[1])
f.close()


f = open(save_dir+'metrics.txt', 'w')

for exp, model_param in zip(exps,model_params):
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

    avr_time = 0
    for i,file in enumerate(os.listdir(data_dir+"/valid/X/")):
        X,Y = load_X_Y("newdata","valid",file,n_layers,use_map,distance_map)
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