import sys
import os
import json
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
sys.path.append("..")
import config


def plot_mirror(between_host_path, within_host_path, ax, threshold_lens, ind_to_plot=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    between_cumu_runs = np.loadtxt(between_host_path)
    within_cumu_runs = np.loadtxt(within_host_path)
    color_idx = 0
    # decide which of the cumu_runs to plot
    if ind_to_plot is None:
        to_plot = range(between_cumu_runs.shape[1])
    else:
        to_plot = ind_to_plot

    for i in to_plot:
        ax.plot(between_cumu_runs[:, i], linewidth=1, color=colors[color_idx],
                label="{}".format(int(threshold_lens[i])))
        ax.plot(-within_cumu_runs[:, i], linewidth=1, color=colors[color_idx])
        color_idx += 1
    ax.hlines(0, 0, between_cumu_runs.shape[0], 'black', linewidth=1)
    ax.set_xlim([0, between_cumu_runs.shape[0]])
    ax.set_ylim([-0.35, 0.35])
    ax.set_xlabel("4D core genome location")
    ax.set_ylabel("sharing fraction")
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_yticklabels(np.around(map(np.abs, ax.get_yticks()), decimals=1))


base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')
between_path = os.path.join(base_path, 'cutoff_0.001.csv')
within_path = os.path.join(base_path, 'within_host_cutoff_0.001.csv')
thresholds = np.loadtxt(os.path.join(base_path, 'thresholds.txt'))

fig, ax = plt.subplots(figsize=(7, 2.5))
plot_mirror(between_path, within_path, ax, threshold_lens=thresholds, ind_to_plot=[0, 3, 6])
fig.savefig('test_within_pileup.pdf', bbox_inches='tight')
