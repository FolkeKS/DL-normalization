# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import wandb
import pandas as pd
import os
print(os.getcwd())
api = wandb.Api()
run = api.run("/normalisation_internship/cnn/runs/16ktmc02")

history = run.scan_history(
    keys=["epoch", "val_loss", "val_max", "val_mean", "val_0.9999_quantile"])

rows = []
j = 0
for i, row in enumerate(history):
    if row['epoch'] != i+j:
        new_row = row
        new_row['epoch'] = i+j
        rows.append(new_row)
        j += 1
    rows.append(row)


dfGood = pd.DataFrame(rows)
dfGood.to_csv("dfGood.csv", index=False)
