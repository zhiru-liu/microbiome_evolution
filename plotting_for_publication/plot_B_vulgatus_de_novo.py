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
# dh = parallel_utils.DataHoarder(species_name, mode="within")
general_mask = parallel_utils.get_general_site_mask(species_name)
snp_info = parallel_utils.get_snp_info(species_name)
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


def filter_raw_data(sample_idx):
    alts = rechunked_alt_arr[:, sample_idx]
    depths = rechunked_depth_arr[:, sample_idx]
    alts = alts.compute()
    depths = depths.compute()
    return alts, depths


def plot_local_polymorphism(axes, sample_pair):
    idx1 = parallel_utils.get_raw_data_idx_for_sample(species_name, sample_pair[0])
    idx2 = parallel_utils.get_raw_data_idx_for_sample(species_name, sample_pair[1])

    a_before, d_before = filter_raw_data(idx1)
    freq_before = np.nan_to_num(a_before / d_before.astype(float))
    a_after, d_after = filter_raw_data(idx2)
    freq_after = np.nan_to_num(a_after / d_after.astype(float))

    intermediate_snvs_before = (freq_before > 0.1) & (freq_before < 0.9)
    intermediate_snvs_after = (freq_after > 0.1) & (freq_after < 0.9)

    intermediate_snvs_before = intermediate_snvs_before[general_mask]
    intermediate_snvs_after = intermediate_snvs_after[general_mask]

    axes[0].plot(np.convolve(intermediate_snvs_before, np.ones(1000)/1000.))
    axes[1].plot(np.convolve(intermediate_snvs_after, np.ones(1000)/1000.))
    axes[0].set_xlabel([])
    axes[0].set_xticklabels([])


def plot_allele_freq_zoomin(axes, histo_axes, sample_pair):
    idx1 = parallel_utils.get_raw_data_idx_for_sample(species_name, sample_pair[0])
    idx2 = parallel_utils.get_raw_data_idx_for_sample(species_name, sample_pair[1])

    # core site idx to all site idx
    core_to_all = np.where(general_mask)[0]
    # hard coded plotting range
    start = core_to_all[117500]
    end = core_to_all[121800]
    print(start, end)

    # get mean depth for copy number
    mean_depth_before = compute_mean_depth(idx1)
    mean_depth_after = compute_mean_depth(idx2)

    # get raw allele frequencies polarized by first time pt
    a_before, d_before = filter_raw_data(idx1)
    a_after, d_after = filter_raw_data(idx2)
    good_sites_before = (d_before > 0)
    good_sites_after = (d_after > 0)
    freq_before = np.nan_to_num(a_before / d_before.astype(float))
    freq_after = np.nan_to_num(a_after / d_after.astype(float))
    to_flip = freq_before > 0.5
    freq_before[to_flip] = 1 - freq_before[to_flip]
    freq_after[to_flip] = 1 - freq_after[to_flip]

    # plot the full site frequency spectrum
    histo_axes[0].hist(freq_before, orientation='horizontal', bins=100)
    histo_axes[1].hist(freq_after, orientation='horizontal', bins=100)
    histo_axes[0].set_ylim([0, 1])
    histo_axes[1].set_ylim([0, 1])
    histo_axes[0].set_xlim([0, 2000])
    histo_axes[1].set_xlim([0, 2000])

    good_sites_before = good_sites_before[start:end]
    good_sites_after = good_sites_after[start:end]
    freq_before = freq_before[start:end]
    freq_after = freq_after[start:end]
    freq_before = freq_before[good_sites_before]
    freq_after = freq_after[good_sites_after]

    xs_before = snp_info[1][start:end][good_sites_before]  # locations
    xs_after = snp_info[1][start:end][good_sites_after]  # locations
    # xs = np.arange(len(freq_before))
    axes[0].plot(xs_before[freq_before < 0.1], freq_before[freq_before < 0.1], '.', markersize=1,
                 label='Alt allele frequency', rasterized=True, color=mpl_colors[0])
    axes[0].plot(xs_before[freq_before > 0.1], freq_before[freq_before > 0.1], '.', color=mpl_colors[0])

    axes[1].plot(xs_after[freq_after < 0.1], freq_after[freq_after < 0.1], '.', markersize=1,
                 label='Alt allele frequency', rasterized=True, color=mpl_colors[0])
    axes[1].plot(xs_after[freq_after > 0.1], freq_after[freq_after > 0.1], '.', color=mpl_colors[0])

    non_core = np.invert(np.isin(snp_info[2][start:end], core_genes)).astype(int)
    non_core_starts = np.nonzero((non_core[1:] - non_core[:-1]) > 0)[0]
    non_core_ends = np.nonzero((non_core[1:] - non_core[:-1]) < 0)[0]
    for i in range(len(non_core_starts)):
        start_idx = snp_info[1][start:end][non_core_starts[i]]
        end_idx = snp_info[1][start:end][non_core_ends[i]]
        axes[0].axvspan(start_idx, end_idx, alpha=0.1,
                        color='b', label='_' * i + 'Non-core sites', linewidth=0)
        axes[1].axvspan(start_idx, end_idx, alpha=0.1,
                        color='b', label='_' * i + 'Non-core sites', linewidth=0)

    N = 1
    copy_num = d_before[start:end] / mean_depth_before
    local_copy_before = np.convolve(copy_num, np.ones((N,)) / N, mode='same')
    copy_num = d_after[start:end] / mean_depth_after
    local_copy_after = np.convolve(copy_num, np.ones((N,)) / N, mode='same')

    axes[0].plot(snp_info[1][start:end], local_copy_before, 'grey', label='Local rel copynumber')
    axes[1].plot(snp_info[1][start:end], local_copy_after, 'grey')

    axes[0].set_xticklabels([])
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
plot_allele_freq_zoomin([zoomin_ax1, zoomin_ax2], [histo_ax1, histo_ax2], ['700114218', '700171115'])
plot_local_polymorphism([local_ax1, local_ax2], ['700114218', '700171115'])

fig.savefig('test_denovo.pdf', bbox_inches="tight")
