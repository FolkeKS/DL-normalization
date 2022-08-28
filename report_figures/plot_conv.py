#!/ usr / bin / env python3
#- * - coding : utf - 8 - * -
"""
Created on Fri Aug  5 11:25:48 2022

@author: coulaud
"""
import matplotlib.font_manager as font_manager
import subprocess
import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import numpy as np

dfELU_train = pd.read_csv("dfELUtrain.csv")
dfELU_val = pd.read_csv("dfELUval.csv")
dfReLU_train = pd.read_csv("dfReLUtrain.csv")
dfReLU_val = pd.read_csv("dfReLUval.csv")


dfELU = dfELU_train.merge(dfELU_val, how='inner')
dfReLU = dfReLU_train.merge(dfReLU_val, how='inner')

y = ['train_loss', 'val_loss', 'train_mean', 'val_mean', 'train_max', 'val_max']

dfELU.rename(columns={'epoch': 'Epochs'}, inplace=True)
dfReLU.rename(columns={'epoch': 'Epochs'}, inplace=True)


kpse_cp = subprocess.run(
    ['kpsewhich', '-var-value', 'TEXMFDIST'], capture_output=True, check=True)
font_loc1 = os.path.join(kpse_cp.stdout.decode(
    'utf8').strip(), 'fonts', 'opentype', 'public', 'tex-gyre')
print(f'loading TeX Gyre fonts from "{font_loc1}"')
font_dirs = [font_loc1]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'TeX Gyre Termes'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['text.usetex'] = True

color = {"train_loss": "#1f77b4", "val_loss": "#ff7f0e", "train_mean": "#2ca02c",
         "val_mean": "#d62728", "train_max": "#9467bd", "val_max": "gold", }

figure, axes = plt.subplots(1, 2, sharey=True, sharex=True)
figure.subplots_adjust(wspace=0)
dfELU.plot(x="Epochs", y=y, logy=True,
           ax=axes[0], legend=False, lw=0.01, color=color)
dfReLU.plot(x="Epochs", y=y, logy=True, ax=axes[1], lw=0.01, color=color)


axes[0].tick_params(axis='y', which='major', left='in')
axes[1].tick_params(axis='y', which='major', direction='inout', right=True)

plt.yticks([1e-5, 1e-3, 1e-1, 1],
           labels=["$10^{-5}$", "$10^{-3}$", "$10^{-1}$", "1"])
plt.xticks(ticks=[0, 2e4, 4e4], labels=[
           "0", r"$2 \times 10^{4}$", r"$4 \times 10^{4}$"])

axes[0].title.set_text('ELU')
axes[1].title.set_text('ReLU')

leg = plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(-0.01, 1.0))
for legobj in leg.legendHandles:
    legobj.set_linewidth(1.0)


figure.savefig("../Desktop/convergences.eps",
               format="eps", bbox_inches='tight')
