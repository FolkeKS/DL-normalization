#!/ usr / bin / env python3
#- * - coding : utf - 8 - * -
"""
Created on Thu Aug 11 15:26:38 2022

@author: coulaud
"""

#!/ usr / bin / env python3
#- * - coding : utf - 8 - * -
"""
Created on Fri Aug  5 10:56:10 2022

@author: coulaud
"""


import numpy as np
import matplotlib.ticker as ticker
import subprocess
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import os
api = wandb.Api(timeout=19)

for step in ["train", "val"]:
    run = api.run("/normalisation_internship/cnn/runs/2obl1jzk")

    history = run.scan_history(
        keys=["epoch", step+"_loss", step+"_max", step+"_mean", step+"_0.9999_quantile"])

    rows = []
    j = 0
    for i, row in enumerate(history):
        if row['epoch'] != i+j:
            new_row = row
            new_row['epoch'] = i+j
            rows.append(new_row)
            j += 1
        rows.append(row)

    dfELU = pd.DataFrame(rows)
    dfELU.to_csv("df90"+step+".csv", index=False)


df90_train = pd.read_csv("df90train.csv")
df90_val = pd.read_csv("df90val.csv")


df90 = df90_train.merge(df90_val, how='inner')

y = ['train_mean', 'val_mean', 'train_max', 'val_max']

df90.rename(columns={'epoch': 'Epochs'}, inplace=True)


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

color = {"train_mean": "#2ca02c", "val_mean": "#d62728",
         "train_max": "#9467bd", "val_max": "gold"}


color = {"train_mean": "#2ca02c",
         "val_mean": "#d62728", "train_max": "#9467bd", "val_max": "gold"}
#figure, axes = plt.subplots()
#figure.subplots_adjust(wspace = 0)
#dfELU.plot(x = "Epochs", y = y, logy = True,
#ax = axes[0], legend = False, lw = 0.01, color = color)
#dfReLU.plot(x = "Epochs", y = y, logy = True, ax = axes[1], lw = 0.01, color = color)

#axes[0].tick_params(axis = 'y', which = 'major', left = 'in')
#axes[1].tick_params(axis = 'y', which = 'major', direction = 'inout', right = True)
df90.plot(x="Epochs", y=y, logy=True, legend=False, lw=0.01, color=color)

#plt.yticks([ 1e-2, 1e-1, 1e-1, 10 ],
#labels = ["1e-2", "1e-1", "1", "10"])
#plt.xticks(ticks = [ 0, 2e4, 4e4 ], labels = [
#"0", r "$2 \times 10^{4}$", r "$4 \times 10^{4}$"])

#axes[0].title.set_text('ELU')
#axes[1].title.set_text('ReLU')

leg = plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
for legobj in leg.legendHandles:
    legobj.set_linewidth(1.0)


plt.savefig("../Desktop/conv_rota.eps",
            format="eps", bbox_inches='tight')
