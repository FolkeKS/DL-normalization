#!/ usr / bin / env python3
#- * - coding : utf - 8 - * -
"""
Created on Fri Aug  5 10:56:10 2022

@author: coulaud
"""

import matplotlib.pyplot as plt
import wandb
import pandas as pd
import os

api = wandb.Api(timeout=19)

for step in ["train", "val"]:
    run = api.run("/normalisation_internship/cnn/runs/2qrrcxdy")

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
    dfELU.to_csv("dfELU"+step+".csv", index=False)

    run = api.run("/normalisation_internship/cnn/runs/2kdh1hkl")

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

    dfReLU = pd.DataFrame(rows)
    dfReLU.to_csv("dfReLU"+step+".csv", index=False)
