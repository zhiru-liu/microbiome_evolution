import os
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import dask.array as da
sys.path.append("..")
import config
from utils import parallel_utils, core_gene_utils


mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# loading necessary data
species_name = "Bacteroides_vulgatus_57955"
dh = parallel_utils.DataHoarder(species_name, mode="within")
core_genes = core_gene_utils.get_sorted_core_genes(species_name)
base_dir = os.path.join(config.data_directory, 'zarr_snps', species_name)
alt_arr = da.from_zarr('{}/full_alt.zarr'.format(base_dir))
depth_arr = da.from_zarr('{}/full_depth.zarr'.format(base_dir))
rechunked_alt_arr = alt_arr.rechunk((1000000, 10))
rechunked_depth_arr = depth_arr.rechunk((1000000, 10))


def compute_mean_depth(sample_idx):
    depths = rechunked_depth_arr[:, sample_idx]
    depths = depths.compute()
    mean_depth = np.mean(depths[depths > 0])
    return mean_depth


def filter_raw_data(sample_idx, start_site, end_site):
    alts = rechunked_alt_arr[start_site:end_site, sample_idx]
    depths = rechunked_depth_arr[start_site:end_site, sample_idx]
    alts = alts.compute()
    depths = depths.compute()
    return alts, depths


def plot_local_polymorphism(ax, sample_id):
    return


def plot_allele_freq_zoomin(axes, dh, sample_pair):
    idx1 = parallel_utils.get_raw_data_idx_for_sample(species_name, sample_pair[0])
    idx2 = parallel_utils.get_raw_data_idx_for_sample(species_name, sample_pair[1])

    # core site idx to all site idx
    core_to_all = np.where(dh.general_mask)[0]
    # hard coded plotting range
    start = core_to_all[117500]
    end = core_to_all[121800]

    # get mean depth for copy number
    mean_depth_before = compute_mean_depth(idx1)
    mean_depth_after = compute_mean_depth(idx2)

    # get raw allele frequencies polarized by first time pt
    a_before, d_before = filter_raw_data(idx1, start, end)
    a_after, d_after = filter_raw_data(idx2, start, end)
    good_sites = (d_before > 0) & (d_after > 0)
    freq_before = np.nan_to_num(a_before / d_before.astype(float))[good_sites]
    freq_after = np.nan_to_num(a_after / d_after.astype(float))[good_sites]
    to_flip = freq_before > 0.5
    freq_before[to_flip] = 1 - freq_before[to_flip]
    freq_after[to_flip] = 1 - freq_after[to_flip]

    xs = np.arange(len(freq_before))
    axes[0].plot(xs[freq_before < 0.1], freq_before[freq_before < 0.1], '.',
                 label='Alt allele frequency', rasterized=True, color=mpl_colors[0])
    axes[0].plot(xs[freq_before > 0.1], freq_before[freq_before > 0.1], '.', color=mpl_colors[0])

    axes[1].plot(xs[freq_after < 0.1], freq_after[freq_after < 0.1], '.',
                 label='Alt allele frequency', rasterized=True, color=mpl_colors[0])
    axes[1].plot(xs[freq_after > 0.1], freq_after[freq_after > 0.1], '.', color=mpl_colors[0])

    non_core = np.invert(np.isin(dh.gene_names[start:end][good_sites], core_genes)).astype(int)
    non_core_starts = np.nonzero((non_core[1:] - non_core[:-1]) > 0)[0]
    non_core_ends = np.nonzero((non_core[1:] - non_core[:-1]) < 0)[0]
    for i in range(len(non_core_starts)):
        axes[0].axvspan(non_core_starts[i], non_core_ends[i], alpha=0.1,
                        color='b', label='_' * i + 'Non-core sites', linewidth=0)
        axes[1].axvspan(non_core_starts[i], non_core_ends[i], alpha=0.1,
                        color='b', label='_' * i + 'Non-core sites', linewidth=0)

    N = 100
    copy_num = d_before[good_sites] / mean_depth_before
    local_copy_before = np.convolve(copy_num, np.ones((N,)) / N, mode='same')
    copy_num = d_after[good_sites] / mean_depth_after
    local_copy_after = np.convolve(copy_num, np.ones((N,)) / N, mode='same')

    axes[0].plot(local_copy_before, 'grey', label='Local rel copynumber')
    axes[1].plot(local_copy_after, 'grey')

    axes[1].set_xlabel('Covered site index')
    axes[0].legend(loc='upper right')
    return


# setting up figures
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

fig = plt.figure(figsize=(7, 5.5))

outer_grid = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 3.5], hspace=0.2, figure=fig)

top_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1.5,2],wspace=0,subplot_spec=outer_grid[0])

local_div_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.2, subplot_spec=top_grid[1])

bottom_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], wspace=0.2, subplot_spec=outer_grid[1])

zoomin_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.2, subplot_spec=bottom_grid[0])

histo_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.2, subplot_spec=bottom_grid[-1])

# adding axes
local_ax1 = fig.add_subplot(local_div_grid[0])
local_ax2 = fig.add_subplot(local_div_grid[1])

zoomin_ax1 = fig.add_subplot(zoomin_grid[0])
zoomin_ax2 = fig.add_subplot(zoomin_grid[1])
histo_ax1 = fig.add_subplot(histo_grid[0])
histo_ax2= fig.add_subplot(histo_grid[1])

# plotting
plot_allele_freq_zoomin([zoomin_ax1, zoomin_ax2], dh, ['700114218', '700171115'])

fig.savefig('test_denovo.pdf', bbox_inches="tight")
