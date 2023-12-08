import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import scipy.stats
from scipy.special import kl_div
import random
from scipy.stats import ks_2samp
import config
from utils import figure_utils

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
count = 0
ks_dat = []
rescaled_ks_dat = []
for i in range(len(files_to_plot)):
    species_name = files_to_plot[i].split('.')[0]
    if 'Lachnospiraceae' in species_name:
        continue
    idx = np.unravel_index(count, axes.shape)
    ax = axes[idx]
    inset_ax = inset_axes(ax,
                            width="40%",  # width = 30% of parent_bbox
                            height="40%",
                            loc='upper right')

    # load simulated transfer distribution
    histo = np.loadtxt(os.path.join(config.hmm_data_directory, species_name + '.csv'))

    # load HMM inferred transfer distribution
    save_path = os.path.join(config.analysis_directory,
                             "closely_related", "third_pass", "{}_all_transfers.pickle".format(species_name))
    run_df = pd.read_pickle(save_path)
    data_dir = os.path.join(config.analysis_directory, "closely_related")
    raw_df = pd.read_pickle(os.path.join(data_dir, 'third_pass', species_name + '.pickle'))

    cf_cutoff = config.clonal_fraction_cutoff
    good_pairs = raw_df[raw_df['clonal fractions'] > cf_cutoff]['pairs']
    mask = run_df['pairs'].isin(good_pairs)
    full_df = run_df[mask]

    # sim_transfers = np.loadtxt(os.path.join(
    #     config.analysis_directory, 'closely_related', 'simulated_transfers', species_name+'.csv'))
    sim_transfers = np.loadtxt(os.path.join(
        config.analysis_directory, 'closely_related', 'simulated_transfers_cphmm', species_name+'.csv'))
    sim_transfers = sim_transfers[~np.isnan(sim_transfers)]
    obs_transfers = full_df['synonymous divergences']

    if 'vulgatus' in species_name:
        # vulgatus has 80 bins because we separated between and within clade transfer
        mids = histo[0, :40]
        density = (histo[1, :40] + histo[1, 40:]) / np.sum(histo[1, :])
    else:
        mids = histo[0, :]
        density = histo[1, :] / histo[1, :].sum()

    # simulated
    # ax.bar(mids, density, width=mids[1] - mids[0], label='simulated', alpha=0.5)
    step = mids[1]-mids[0]
    step *= 2

    bins = np.arange(0, max(sim_transfers.max(), obs_transfers.max()) + step, step)
    sim_hist = np.histogram(sim_transfers, bins=bins)
    counts, bins = sim_hist
    new_mids = (bins[:-1] + bins[1:]) / 2
    sim_density = counts / np.sum(counts).astype(float)
    ax.bar(new_mids, sim_density, width=step, label='simulated', alpha=0.5)
    # inset_ax.hist(sim_transfers, cumulative=-1, density=True, bins=bins, alpha=0.5)
    X, Y = figure_utils.plot_ecdf(inset_ax, sim_transfers, return_xy=True)
    inset_ax.fill_between(X, Y, 0, color='tab:blue', alpha=.3, rasterized=True)

    # bins = invert_bins(histo[0, :])
    counts, bins = np.histogram(obs_transfers, bins=bins)
    new_mids = (bins[:-1] + bins[1:]) / 2
    obs_density = counts / np.sum(counts).astype(float)
    ax.bar(new_mids, obs_density, width=step, label='observed', alpha=0.5)
    # inset_ax.hist(obs_transfers, cumulative=-1, density=True, bins=bins, alpha=0.5)
    X, Y = figure_utils.plot_ecdf(inset_ax, obs_transfers, return_xy=True)
    inset_ax.fill_between(X, Y, 0, color='tab:orange', alpha=.3, rasterized=True)
    # ax.legend()
    # ax.set_xlabel('transfer divergence')
    ax.set_title(figure_utils.get_pretty_species_name(species_name))
    ax.set_ylim(bottom=-bottom_offset)
    inset_ax.set_xlim(xmax=inset_ax.get_xlim()[1] / 2)
    count += 1

    divergence_ratio = np.mean(sim_transfers) / np.mean(obs_transfers)
    # ks_dist, p_val = ks_2samp(obs_transfers, sim_transfers, alternative='greater')
    # res = cramervonmises_2samp(obs_transfers, sim_transfers)
    # hist_dist = scipy.stats.rv_histogram(sim_hist)
    # res = scipy.stats.cramervonmises_2samp(obs_transfers, sim_transfers, method='exact')
    # res = scipy.stats.ks_2samp(obs_transfers, sim_transfers, alternative='greater')
    # ks_dat.append((species_name, ks_dist, p_val))
    # print(ks_dat[-1])
    # ks_dist, p_val = ks_2samp(obs_transfers * divergence_ratio, sim_transfers)
    # rescaled_ks_dat.append((species_name, ks_dist, p_val))
    # print(species_name, ks_dist, p_val)

for i in range(axes.shape[0]):
    axes[i, 0].set_ylabel('probability density')
for j in range(axes.shape[1]-1):
    axes[-1, j].set_xlabel('transfer divergence (syn)')
axes[-2, -1].set_xlabel('transfer divergence (syn)')
axes[-1, -2].legend(loc='center', bbox_to_anchor=(1.7, 0.45), fontsize=8)
fig.delaxes(axes[-1, -1])

# fig.savefig(os.path.join(config.figure_directory, 'supp_transfer_histo_suppresions_no_loc_control.pdf'), bbox_inches='tight')
# ks_df = pd.DataFrame(ks_dat)
# ks_df.to_csv(os.path.join(config.plotting_intermediate_directory, 'transfer_distribution_ks_test.csv'))
# ks_df = pd.DataFrame(rescaled_ks_dat)
# ks_df.to_csv(os.path.join(config.plotting_intermediate_directory, 'transfer_distribution_ks_test_rescaled.csv'))
fig.savefig(os.path.join(config.figure_directory, 'supp', 'S23_supp_transfer_histograms_cphmm.pdf'), bbox_inches='tight')
