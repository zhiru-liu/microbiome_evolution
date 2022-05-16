import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random
import config
from utils import figure_utils, parallel_utils, typical_pair_utils

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

files_to_plot = os.listdir(os.path.join(config.analysis_directory, 'closely_related', 'simulated_transfers'))
files_to_plot = list(filter(lambda x: not x.startswith('.'), files_to_plot))

cols = int(5)
rows = int(np.ceil(len(files_to_plot) / float(cols)))
fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 1.5*rows))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

# species = ['Bacteroides_vulgatus_57955', 'Alistipes_shahii_62199', 'Eubacterium_rectale_56927', 'Bacteroides_fragilis_54507']
# species = ['Alistipes_shahii_62199']


def invert_bins(arr):
    # handy function to take the mid points of bins and return edges of bins
    dx = arr[1] - arr[0]
    start = arr[0] - dx / 2
    end = arr[-1] + dx
    return np.arange(start, end, dx)

bottom_offset = 1e-3  # some bars are thinner than the bottom axis
for i in range(len(files_to_plot)):
    idx = np.unravel_index(i, axes.shape)
    ax = axes[idx]
    species_name = files_to_plot[i].split('.')[0]

    # load simulated transfer distribution
    histo = np.loadtxt(os.path.join(config.hmm_data_directory, species_name + '.csv'))

    # load HMM inferred transfer distribution
    save_path = os.path.join(config.analysis_directory,
                             "closely_related", "third_pass", "{}_all_transfers.pickle".format(species_name))
    full_df = pd.read_pickle(save_path)

    sim_transfers = np.loadtxt(os.path.join(
        config.analysis_directory, 'closely_related', 'simulated_transfers', species_name+'.csv'))
    sim_transfers = sim_transfers[~np.isnan(sim_transfers)]

    if 'vulgatus' in species_name:
        # vulgatus has 80 bins because we separated between and within clade transfer
        mids = histo[0, :40]
        density = (histo[1, :40] + histo[1, 40:]) / np.sum(histo[1, :])
    else:
        mids = histo[0, :]
        density = histo[1, :] / histo[1, :].sum()

    # simulated
    ax.bar(mids, density, width=mids[1] - mids[0], label='simulated', alpha=0.5)
    # bins = np.arange(0, sim_transfers.max() + mids[1]-mids[0], mids[1]-mids[0])
    # counts, bins = np.histogram(sim_transfers, bins=bins)
    # new_mids = (bins[:-1] + bins[1:]) / 2
    # ax.bar(new_mids, counts / np.sum(counts).astype(float), width=mids[1] - mids[0], label='simulated', alpha=0.5)
    # ax.hist(sim_transfers, cumulative=-1, density=True, bins=bins, alpha=0.5)

    # bins = invert_bins(histo[0, :])
    bins = np.arange(0, full_df['divergences'].max() + mids[1]-mids[0], mids[1]-mids[0])
    counts, bins = np.histogram(full_df['divergences'], bins=bins)
    new_mids = (bins[:-1] + bins[1:]) / 2
    ax.bar(new_mids, counts / np.sum(counts).astype(float), width=mids[1] - mids[0], label='empirical', alpha=0.5)
    # ax.hist(full_df['divergences'], cumulative=-1, density=True, bins=bins, alpha=0.5)
    ax.legend()
    # ax.set_xlabel('transfer divergence')
    ax.set_title(figure_utils.get_pretty_species_name(species_name))
    ax.set_ylim(bottom=-bottom_offset)

for i in range(axes.shape[0]):
    axes[i, 0].set_ylabel('probability density')
for j in range(axes.shape[1]):
    axes[-1, j].set_xlabel('transfer divergence (syn)')

fig.savefig(os.path.join(config.figure_directory, 'supp_transfer_histo_suppresions_no_loc_control.pdf'), bbox_inches='tight')
