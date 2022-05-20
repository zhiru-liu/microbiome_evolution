""" Use python3 to use updated scipy package that has ks_2samp"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import ks_2samp
sys.path.append("..")
import config

# set up figure
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize']  = 'small'

max_run_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs')
plot_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs_plots')
files = [
    'Alistipes_putredinis_61533',
    'Alistipes_shahii_62199',
    'Bacteroides_stercoris_56735',
    'Bacteroides_thetaiotaomicron_56941',
    'Bacteroides_vulgatus_57955_diff_clade',
    # 'Bacteroides_vulgatus_57955_same_clade',
    # 'Eubacterium_rectale_56927',
    'Parabacteroides_distasonis_56985',
    'Parabacteroides_merdae_56972'
]

plot_idx = 0
nrow = 2
ncol = 4
fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6, 3))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for species_name in files:
    row = plot_idx // ncol
    col = plot_idx % ncol

    within_host_max_runs = np.loadtxt(os.path.join(max_run_dir, species_name + '_within.txt'), ndmin=1)
    between_host_max_runs = np.loadtxt(os.path.join(max_run_dir, species_name + '_between.txt'))
    # in order to use one sided ks test, need to use python3's scipy
    ks_dist, p_val = ks_2samp(within_host_max_runs, between_host_max_runs, alternative='less')

    ax = axes[row, col]
    ax.hist([between_host_max_runs, within_host_max_runs], bins=100, density=True,
            cumulative=-1, histtype='step', label=['Between host', 'Within host'])
    # ax.set_title("$n_w={}, p={:.1e}$".format(len(within_host_max_runs), p_val))
    ax.text(0.8, 0.8, "$n_w={}$\n$p={:.1e}$".format(len(within_host_max_runs), p_val))
    plot_idx += 1

ax.legend(bbox_to_anchor=(2, 0.5))

for ax in axes[:, 0]:
    ax.set_ylabel('Fraction longer than')
for ax in axes[-1, :]:
    ax.set_xlabel('Max homozygous run length \n(4D syn sites)')
fig.delaxes(axes[-1, -1])

fig.savefig(os.path.join(config.figure_directory, 'supp_within_between_max_runs.pdf'), bbox_inches='tight')
plt.close()
