#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:24:59 2022

@author: coulaud
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import matplotlib.font_manager as font_manager

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
map = np.load("../Documents/DL-normalization/data/sign_dist_map.npz")['arr_0']

nw_map = map[np.where(map > 0)]

fig, ax = plt.subplots()
plt.hist(nw_map, np.unique(nw_map), alpha=0.95, rwidth=1, edgecolor="#1f77b4")
plt.ylabel("Number of pixels", fontsize=16)
plt.xlabel("Manhattan distance from coasts", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax.axvline(np.median(nw_map), color="black", ls="--", label="Median")
ax.legend()
plt.show()


fig.savefig("../Desktop/hist_dist.eps", format="eps",
            bbox_inches='tight', transparent=True)
