#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:14:14 2022

@author: coulaud
"""

import matplotlib.pyplot as plt
import wandb
import pandas as pd
import numpy as np

df = pd.read_csv("dfGood.csv")


plt.figure(figsize=(30, 10))
# plt.grid()
# plt.xticks([1,2,3,4,5,8,10])
# plt.yticks([0.0,0.05,0.1,0.15])
plt.ylim(np.min(np.float32(df['val_loss'])), 0.616450)
plt.xlim(30000, 50000)
plt.plot(df['epoch'], np.float32(df['val_loss']))
# plt.yscale('log')
plt.show()
