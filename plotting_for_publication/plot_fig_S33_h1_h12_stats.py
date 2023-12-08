import os
import numpy as np
import pandas as pd
import pickle
import sys
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm # for color map

import config

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

hl_start, hl_end = 117659, 120344  # the within host sweep region
species_name = 'Bacteroides_vulgatus_57955'
save_path = os.path.join(config.analysis_directory, 'sharing_pileup',
                         'empirical', "%s"%species_name)
thresholds = np.loadtxt(os.path.join(save_path, 'between_host_thresholds.txt'))
cumu_runs = np.loadtxt(os.path.join(save_path, 'between_host.csv'))

df = pd.read_csv(os.path.join(config.plotting_intermediate_directory, 'haplotype_stats.csv'))

fig, axes = plt.subplots(3, 1, figsize=(6, 4), dpi=600)
snp_window = 10000
# axes[0].plot(np.convolve(has_snp, np.ones(snp_window)/snp_window, mode='Same'))
xs = np.arange(cumu_runs.shape[0])
zeros = np.zeros(cumu_runs.shape[0])
axes[0].fill_between(xs, zeros, cumu_runs[:, 0], color='tab:blue', alpha=0.5, rasterized=True)
axes[0].plot(cumu_runs[:, 0], label=thresholds[0], rasterized=True)

from scipy.interpolate import interp1d
y = np.array(df['H2'] / df['Heterozygosity'])
x = np.array((df['Start'] + df['End']) / 2).astype(int)
f = interp1d(x, y)
xnew = np.arange(min(x), max(x))
c_val = cumu_runs[xnew, 0]
axes[2].scatter(xnew, f(xnew), s=1, c=cm.Blues(np.abs(c_val / max(c_val))), label='H2/H1', rasterized=True)

axes[1].plot(x[:-4], df['Heterozygosity'][:-4], markersize=1, label='Homozygosity (H1)', rasterized=True)
axes[1].plot(x[:-4], df['H12'][:-4], markersize=1, label='H12', rasterized=True)

axes[0].axvspan(hl_start, hl_end, color='red', alpha=0.2, ymin=0., linewidth=1, zorder=3)
axes[1].axvspan(hl_start, hl_end, color='red', alpha=0.2, ymin=0., linewidth=1, zorder=3)
axes[2].axvspan(hl_start, hl_end, color='red', alpha=0.2, ymin=0., linewidth=1, zorder=3)

for start, end in [(61000, 63000), (143000, 145000), (235000, 237000)]:
    axes[0].axvspan(start, end, color='yellow', alpha=0.3, ymin=0., linewidth=1, zorder=3)
    axes[1].axvspan(start, end, color='yellow', alpha=0.3, ymin=0., linewidth=1, zorder=3)
    axes[2].axvspan(start, end, color='yellow', alpha=0.3, ymin=0., linewidth=1, zorder=3)

for i in range(2):
    axes[i].set_xticklabels([])
axes[1].legend()
# axes[2].legend()
axes[0].set_ylabel('Sharing probability')
axes[2].set_ylabel('H2 / H1')
axes[2].set_xlabel('Location along core genome')

axes[0].set_xlim([0, cumu_runs.shape[0]])
axes[1].set_xlim([0, cumu_runs.shape[0]])
axes[2].set_xlim([0, cumu_runs.shape[0]])
axes[0].set_ylim(ymin=0)

for i in range(3):
    axes[i].text(-0.05, 1.05, ['A', 'B', 'C'][i], transform=axes[i].transAxes,
         fontsize=9, fontweight='bold', va='top', ha='left')

fig.savefig(os.path.join(config.figure_directory, 'supp', 'S33_supp_h1_h12_scan.pdf'))