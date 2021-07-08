import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append("..")
import config
from utils import pileup_utils


def plot_single_side(ckpt_path, threshold_lens, pileup_ax, histo_ax):
    cumu_runs = np.loadtxt(ckpt_path)
    histo_bins = np.linspace(0, np.max(cumu_runs), 100)
    for i in range(cumu_runs.shape[1]):
        dat = cumu_runs[:, i]  # / float(rep)
        pileup_ax.plot(dat, linewidth=1, label="{}".format(threshold_lens[i]))
        _ = histo_ax.hist(dat, bins=histo_bins, orientation='horizontal')
    pileup_ax.legend()
    pileup_ax.set_ylim([0, 0.3])
    pileup_ax.set_xlim([0, cumu_runs.shape[0]])
    pileup_ax.set_xlabel('4D core genome location')
    pileup_ax.set_ylabel('sharing fraction')


def plot_mirror(between_host_path, within_host_path, ax):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    between_cumu_runs = np.loadtxt(between_host_path)
    within_cumu_runs = np.loadtxt(within_host_path)
    for i in range(between_cumu_runs.shape[1]):
        ax.plot(between_cumu_runs[:, i], linewidth=1, color=colors[i])
        ax.plot(-within_cumu_runs[:, i], linewidth=1, color=colors[i])
    ax.hlines(0, 0, between_cumu_runs.shape[0], 'black', linewidth=1)
    ax.set_xlim([0, between_cumu_runs.shape[0]])
    ax.set_ylim([-0.3, 0.3])
    ax.set_xlabel("4D core genome location")
    ax.set_ylabel("sharing fraction")
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_yticklabels(np.around(map(np.abs, ax.get_yticks()), decimals=1))

# set up figure
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

fig, axes = plt.subplots(1, 2, figsize=(7, 2), sharey=True, gridspec_kw={'width_ratios': [4, 1]})
save_path = os.path.join(config.analysis_directory, 'IBS_locations', 'full_genome', 'B_vulgatus_cutoff_0.001.csv')
plot_single_side(save_path, [15, 20, 25, 30, 35], axes[0], axes[1])

fig.savefig('test_pileup.pdf', bbox_inches='tight')
