import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ks_2samp
import sys
import random
sys.path.append("..")
import config

def plot_max_run_histo(ax, species_name):
    max_run_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs')
    within_host_max_runs = np.loadtxt(os.path.join(max_run_dir, species_name + '_within.txt'), ndmin=1)
    between_host_max_runs = np.loadtxt(os.path.join(max_run_dir, species_name + '_between.txt'))
    if len(between_host_max_runs)>5000:
        between_host_max_runs = random.sample(list(between_host_max_runs), 5000)
    ks_dist, p_val = ks_2samp(within_host_max_runs, between_host_max_runs, alternative='less')

    bins = np.arange(max(max(between_host_max_runs), max(within_host_max_runs)))
    ax.hist([between_host_max_runs, within_host_max_runs], bins=100, density=True,
            cumulative=-1, histtype='step', label=['Between host', 'Within host'], color=[config.between_host_color, config.within_host_color])
    ax.set_title(' '.join(species_name.split('_')[:2]))
    # ax.set_title("$n_w={}, p={:.1e}$".format(len(within_host_max_runs), p_val))
    ax.text(0.95, 0.8, '$n_w={}$\n$p={:.1e}$'.format(len(within_host_max_runs), p_val),
            horizontalalignment='right', verticalalignment='center', transform = ax.transAxes)


def plot_long_run_histo(ax, species_name):
    filepath = os.path.join(config.analysis_directory, 'typical_pairs', 'long_run_sums', species_name+'between.csv')
    between_long_runs = np.loadtxt(filepath)
    filepath = os.path.join(config.analysis_directory, 'typical_pairs', 'long_run_sums', species_name+'within.csv')
    within_long_runs = np.loadtxt(filepath)

    ks_dist, p_val = ks_2samp(within_long_runs, between_long_runs, alternative='less')
    bins = np.arange(max(max(between_long_runs), max(within_long_runs)))
    _ = ax.hist(within_long_runs, cumulative=-1, density=True, bins=bins, histtype='step', label='within-host', color=config.within_host_color)
    _ = ax.hist(between_long_runs, cumulative=-1, density=True, bins=bins, histtype='step', label='between-host', color=config.between_host_color)
    # ax.set_title("length threshold: {0}, p={1:.2e}".format(int(len_threshold), p_val))
    ax.set_title("$p={0:.2e}$".format(p_val))
    ax.set_xlabel('Cumulative length of long shared fragments (4D syn sites), $x$')
    ax.set_ylabel('Fraction of pairs greater than $x$')
    ax.legend()


species_to_plot = [
    'Alistipes_putredinis_61533',
    'Alistipes_shahii_62199',
    'Bacteroides_stercoris_56735',
    'Bacteroides_thetaiotaomicron_56941',
    'Bacteroides_vulgatus_57955_diff_clade',
    'Parabacteroides_distasonis_56985',
    'Parabacteroides_merdae_56972'
]

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
# mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

fig, axes = plt.subplots(2, 4, figsize=(7, 2.5))
plt.subplots_adjust(hspace=0.6, wspace=0.4)
for i in range(2):
    for j in range(4):
        species_ind = i*4+j
        if species_ind >= len(species_to_plot):
            break
        species = species_to_plot[species_ind]
        ax = axes[i, j]
        plot_max_run_histo(ax, species)

axes[0, 0].set_ylabel('Fraction longer than $x$')
axes[1, 0].set_ylabel('Fraction longer than $x$')
axes[1, 0].set_xlabel('Max shared fragment length\n(4D syn sites), $x$')
axes[1, 1].set_xlabel('Max shared fragment length\n(4D syn sites), $x$')
axes[1, 2].set_xlabel('Max shared fragment length\n(4D syn sites), $x$')
axes[0, 3].set_xlabel('Max shared fragment length\n(4D syn sites), $x$')
axes[1, 2].legend(bbox_to_anchor=(1.8, 0.5), loc='center')
fig.delaxes(axes[1, 3])


fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_max_run_dist.pdf'), bbox_inches='tight')

fig2, ax = plt.subplots(figsize=(4, 3))
plot_long_run_histo(ax, 'Eubacterium_rectale_56927')
fig2.savefig(os.path.join(config.figure_directory, 'supp', 'supp_long_run_dist.pdf'), bbox_inches='tight')
