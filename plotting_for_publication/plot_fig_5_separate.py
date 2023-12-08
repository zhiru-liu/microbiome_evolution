import os
import numpy as np
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import dask.array as da
sys.path.append("..")
import config
from utils import snp_data_utils, core_gene_utils, close_pair_utils, figure_utils


mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# loading necessary data
species_name = "Bacteroides_vulgatus_57955"
# dh = snp_data_utils.DataHoarder(species_name, mode="within")

general_mask = snp_data_utils.get_general_site_mask(species_name)
snp_info = snp_data_utils.get_snp_info(species_name)
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
    idx1 = snp_data_utils.get_raw_data_idx_for_sample(species_name, sample_pair[0])
    idx2 = snp_data_utils.get_raw_data_idx_for_sample(species_name, sample_pair[1])

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
    axes[0].set_xlabel('')
    axes[0].set_xticklabels([])
    axes[0].set_ylim([-0.002, 0.06])
    axes[1].set_ylim([-0.002, 0.06])
    axes[1].set_xlabel("Core genome synonymous site location")
    axes[1].set_ylabel("Intermediate frequency \nSNV density")


def plot_allele_freq_zoomin(axes, histo_axes, copy_axes, bar_axes, sample_pair, plot_locations=True):
    idx1 = snp_data_utils.get_raw_data_idx_for_sample(species_name, sample_pair[0])
    idx2 = snp_data_utils.get_raw_data_idx_for_sample(species_name, sample_pair[1])

    # core site idx to all site idx
    core_to_all = np.where(general_mask)[0]
    # hard coded plotting range
    start = core_to_all[117500]
    end = core_to_all[121800]
    # start = core_to_all[47500]
    # end = core_to_all[51800]
    print(start, end)

    # get mean depth for copy number
    mean_depth_before = compute_mean_depth(idx1)
    mean_depth_after = compute_mean_depth(idx2)

    # get raw allele frequencies polarized by first time pt
    a_before, d_before = filter_raw_data(idx1)
    a_after, d_after = filter_raw_data(idx2)
    # good_sites_before = (d_before > 0)
    # good_sites_after = (d_after > 0)
    good_sites = (d_before > 0) & (d_after > 0)
    freq_before = np.nan_to_num(a_before / d_before.astype(float))
    freq_after = np.nan_to_num(a_after / d_after.astype(float))
    to_flip = freq_before > 0.5
    freq_before[to_flip] = 1 - freq_before[to_flip]
    freq_after[to_flip] = 1 - freq_after[to_flip]

    # plot the full site frequency spectrum
    histo_axes[0].hist(freq_before[good_sites], orientation='horizontal', bins=100)
    histo_axes[1].hist(freq_after[good_sites], orientation='horizontal', bins=100)
    histo_axes[0].set_xlim([0, 1000])
    histo_axes[1].set_xlim([0, 1000])
    histo_axes[0].set_xticklabels([])
    histo_axes[1].set_xticks([0, 500, 1000])
    histo_axes[1].set_xticklabels(['0', '500', '1k'])
    histo_axes[0].set_yticklabels([])
    histo_axes[1].set_yticklabels([])
    histo_axes[0].spines['top'].set_visible(False)
    histo_axes[0].spines['right'].set_visible(False)
    histo_axes[1].spines['top'].set_visible(False)
    histo_axes[1].spines['right'].set_visible(False)
    histo_axes[1].set_xlabel('Site counts\n(all sites)')

    # good_sites_before = good_sites_before[start:end]
    # good_sites_after = good_sites_after[start:end]
    good_sites = good_sites[start:end]
    freq_before = freq_before[start:end]
    freq_after = freq_after[start:end]
    freq_before = freq_before[good_sites]
    freq_after = freq_after[good_sites]

    # finding the minimal and maximal region that engaged in recombination
    freq_change_sites = np.where((freq_before > 0.2) & (freq_after < 0.2))[0]
    if len(freq_change_sites) != 0:
        region_start = freq_change_sites[0]
        region_end = freq_change_sites[-1]
        print("minimal spanning region is %d to %d" % (region_start, region_end))
        minimal_genes = np.unique(snp_info[2][start:end][good_sites][region_start:region_end])

        remained_snps = np.where(freq_after > 0.5)[0]
        region_start = max(remained_snps[remained_snps < 10000])
        region_end = min(remained_snps[remained_snps > 30000])
        print("maximal spanning region is %d to %d" % (region_start, region_end))
        maximal_genes = np.unique(snp_info[2][start:end][good_sites][region_start:region_end])[1:-1]
    else:
        maximal_genes = None
        minimal_genes = None

    # xs_before = snp_info[1][start:end][good_sites_before]  # locations
    # xs_after = snp_info[1][start:end][good_sites_after]  # locations
    if plot_locations:
        xs = snp_info[1][start:end][good_sites]
        print("ref genome region: {} - {}".format(xs[0], xs[-1]))
    else:
        xs = np.arange(len(freq_before))
    axes[0].plot(xs[freq_before < 0.1], freq_before[freq_before < 0.1], '.', markersize=2,
                 label='SNVs', rasterized=True, color=mpl_colors[0])
    axes[0].plot(xs[freq_before > 0.1], freq_before[freq_before > 0.1], '.', markersize=2, color=mpl_colors[0])

    axes[1].plot(xs[freq_after < 0.1], freq_after[freq_after < 0.1], '.', markersize=2,
                 label='SNV frequency', rasterized=True, color=mpl_colors[0])
    axes[1].plot(xs[freq_after > 0.1], freq_after[freq_after > 0.1], '.', markersize=2, color=mpl_colors[0])

    x1 = xs
    y1 = freq_before
    x2 = xs
    y2 = freq_after
    figure_utils.save_figure_data([x1, y1, x2, y2], ['loc0', 'freq0', 'loc1', 'freq1'], config.figure_data_directory, 'fig5/5c')

    non_core = np.invert(np.isin(snp_info[2][start:end][good_sites], core_genes)).astype(int)
    non_core_starts = np.nonzero((non_core[1:] - non_core[:-1]) > 0)[0]
    non_core_starts = xs[non_core_starts]
    non_core_ends = np.nonzero((non_core[1:] - non_core[:-1]) < 0)[0]
    non_core_ends = xs[non_core_ends]
    for i in range(len(non_core_starts)):
        # start_idx = snp_info[1][start:end][non_core_starts[i]]
        # end_idx = snp_info[1][start:end][non_core_ends[i]]
        axes[0].axvspan(non_core_starts[i], non_core_ends[i], alpha=0.1,
                        color='tab:grey', label='_' * i + 'Non-core genes', linewidth=0)
        axes[1].axvspan(non_core_starts[i], non_core_ends[i], alpha=0.1,
                        color='tab:grey', label='_' * i + 'Non-core genes', linewidth=0)


    copy_axes[0].axvspan(-10, -9, alpha=0.1,
                    color='tab:grey', label='Non-core genes', linewidth=0)
    copy_axes[0].plot(-10, 1, '.', markersize=2,
                 label='SNVs', color=mpl_colors[0])

    N = 1000
    # copy_num = d_before[start:end][good_sites] / mean_depth_before
    copy_num = d_before[start:end][good_sites]
    local_copy_before = np.convolve(copy_num, np.ones((N,)) / N, mode='same')
    # copy_num = d_after[start:end][good_sites] / mean_depth_after
    copy_num = d_after[start:end][good_sites]
    local_copy_after = np.convolve(copy_num, np.ones((N,)) / N, mode='same')

    #     copy_axes[0].plot(xs, local_copy_before, 'grey', label='Local rel copynumber')
    zeros = np.zeros(xs.shape[0])
    copy_axes[0].fill_between(xs, zeros, local_copy_before, alpha=0.2, rasterized=True, color='tab:blue')
    #     copy_axes[1].plot(xs, local_copy_after, 'grey')
    copy_axes[1].fill_between(xs, zeros, local_copy_after, alpha=0.2, rasterized=True, color='tab:blue')
    copy_axes[0].axhline(mean_depth_before, color='grey', linestyle='--', linewidth=0.5, alpha=0.8)
    copy_axes[1].axhline(mean_depth_after, color='grey', linestyle='--', linewidth=0.5, alpha=0.8)

    copy_axes[0].set_xticklabels([])
    copy_axes[1].set_xticklabels([])
    copy_axes[0].set_xlim([0., xs.shape[0]])
    copy_axes[1].set_xlim([0., xs.shape[0]])
    # copy_axes[0].set_ylim([0, 1.6])
    # copy_axes[1].set_ylim([0, 1.6])
    # copy_axes[0].set_yticklabels(['0', '1.0'])
    # copy_axes[1].set_yticklabels(['0', '1.0'])
    copy_axes[0].minorticks_on()
    copy_axes[0].xaxis.set_tick_params(which='minor', bottom=False)
    copy_axes[1].minorticks_on()
    copy_axes[1].xaxis.set_tick_params(which='minor', bottom=False)

    axes[0].set_xticklabels([])
    axes[1].set_xlabel('Location along genome \n(highlighted region only)')

    axes[0].set_ylim([0, 1.])
    axes[1].set_ylim([0, 1.])
    axes[0].set_xlim([0., xs.shape[0]])
    axes[1].set_xlim([0., xs.shape[0]])

    histo_axes[0].set_ylim([0, axes[0].get_ylim()[1]])
    histo_axes[0].set_yticks([0, 0.5, 1])
    histo_axes[1].set_ylim([0, axes[1].get_ylim()[1]])
    histo_axes[1].set_yticks([0, 0.5, 1])

    sample1_freq = 1 - 0.7109375  # hard coded; freq from snp_data_utils.get_single_peak_sample_mask
    sample2_freq = 0.67283951
    dic = {'linewidth': 1, 'color': 'grey', 'linestyle': '--', 'alpha': 1}
    #     axes[0].axhline(sample1_freq, label='Strain frequency', **dic)
    #     axes[1].axhline(sample2_freq, **dic)
    #     histo_axes[0].axhline(sample1_freq,**dic)
    #     histo_axes[1].axhline(sample2_freq,**dic)

    #     print(minimal_genes[np.isin(minimal_genes, core_genes)])

    strain1_color = '#A9D0F7'
    strain2_color = '#EBD0F7'
    alpha = 1
    bar_axes[0].bar(1.8, sample1_freq, width=1, align='edge', color=strain2_color, alpha=alpha, linewidth=0.5,
                    edgecolor='grey')
    bar_axes[0].bar(1.8, 1 - sample1_freq, align='edge', bottom=sample1_freq, width=1, color=strain1_color, alpha=alpha,
                    linewidth=0.5, edgecolor='grey')
    bar_axes[1].bar(1.8, sample2_freq, width=1, align='edge', color=strain2_color, alpha=alpha, linewidth=0.5,
                    edgecolor='grey')
    bar_axes[1].bar(1.8, 1 - sample2_freq, align='edge', bottom=sample2_freq, width=1, color=strain1_color, alpha=alpha,
                    linewidth=0.5, edgecolor='grey')
    #     xs = np.linspace(0, 0.5)
    #     bar_axes[0].plot(xs, np.ones(xs.shape)*sample1_freq, color='k')
    #     bar_axes[1].plot(xs, np.ones(xs.shape)*sample2_freq, color='k')

    bar_axes[0].set_xlim([0, 3])
    bar_axes[1].set_xlim([0, 3])
    bar_axes[0].set_ylim([-0.01, 1])
    bar_axes[1].set_ylim([-0.01, 1])
    #     bar_axes[0].set_xticks([])
    bar_axes[1].set_yticks([])
    bar_axes[1].set_xticks([])
    bar_axes[1].set_yticks([])
    bar_axes[1].set_xlabel('Inferred \nstrain \ncomposition')
    bar_axes[1].xaxis.set_label_coords(0.77, -0.15)
    bar_axes[0].axis('off')
    bar_axes[1].spines['top'].set_visible(False)
    bar_axes[1].spines['right'].set_visible(False)
    bar_axes[1].spines['bottom'].set_visible(False)
    bar_axes[1].spines['left'].set_visible(False)
    #     bar_axes[0].text(0, 0.5, "Inferred strain freq.", rotation='vertical')

    # axes[0].legend(loc='lower left', bbox_to_anchor=(1.02, 1.05), ncol=1, fontsize=6)
    axes[0].set_ylabel("SNV\nfrequency", rotation='horizontal', verticalalignment='center', horizontalalignment='center', labelpad=25)
    axes[1].set_ylabel("SNV\nfrequency", rotation='horizontal', verticalalignment='center', horizontalalignment='center', labelpad=25)
    copy_axes[0].set_ylabel("Coverage", rotation='horizontal', verticalalignment='center', horizontalalignment='center', labelpad=25)
    copy_axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,)
    copy_axes[1].set_ylabel("Coverage", rotation='horizontal', verticalalignment='center', horizontalalignment='center', labelpad=25)

    axes[0].set_xticks([0, 10000, 20000, 30000, 40000])
    axes[1].set_xticks([0, 10000, 20000, 30000, 40000])
    return minimal_genes, maximal_genes

def plot_max_run_histo(ax, species_name):
    max_run_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs')
    within_host_max_runs = np.loadtxt(os.path.join(max_run_dir, species_name + '_within.txt'), ndmin=1)
    between_host_max_runs = np.loadtxt(os.path.join(max_run_dir, species_name + '_between.txt'))
    ax.hist([between_host_max_runs, within_host_max_runs], bins=100, density=True,
            cumulative=-1, histtype='step', color=[config.between_host_color, config.within_host_color])
    # ax.set_xlabel('Max homozygous run length\n(4D syn sites), $x$')
    # ax.set_xlabel(r'Max sharing length, $\ell$')
    ax.set_xlabel('Length of longest \n'
                  r'sharing tract, $\ell$')
    ax.set_ylabel(r'Fraction pairs$\geq\ell$')
    items = species_name.split('_')
    name = ' '.join([items[0][0]+'.', items[1]])
    if 'vulgatus' in species_name:
        p_val = '9.1\\times 10^{-1}'  # copied from supp_plot_within_host_run_length_enrichment.py
    else:
        # E rectale
        p_val = '4.9\\times 10^{-3}'  # copied from supp_plot_within_host_run_length_enrichment.py
    ax.text(0.95, 0.85, name, transform=ax.transAxes, va='bottom', ha='right', fontsize=6)
    ax.text(0.95, 0.8, "$n_w={}$\n$P={}$".format(len(within_host_max_runs), p_val),
            transform=ax.transAxes, va='top', ha='right', fontsize=6)


def plot_example_snps(axes):
    cache_file = os.path.join(config.plotting_intermediate_directory, "fig5_within_snp1.csv")
    within_snp_vec1 = np.loadtxt(cache_file).astype(bool)
    cache_file = os.path.join(config.plotting_intermediate_directory, "fig5_within_snp2.csv")
    within_snp_vec2 = np.loadtxt(cache_file).astype(bool)
    cache_file = os.path.join(config.plotting_intermediate_directory, "fig5_between_snp1.csv")
    between_snp_vec1 = np.loadtxt(cache_file).astype(bool)
    cache_file = os.path.join(config.plotting_intermediate_directory, "fig5_between_snp2.csv")
    between_snp_vec2 = np.loadtxt(cache_file).astype(bool)

    blk_size = 1000
    snp_blk = close_pair_utils.to_block(within_snp_vec1, blk_size)
    barcode1 = np.concatenate([snp_blk > 0, [0]])
    snp_blk = close_pair_utils.to_block(within_snp_vec2, blk_size)
    barcode2 = np.concatenate([snp_blk > 0, [0]])
    snp_blk = close_pair_utils.to_block(between_snp_vec1, blk_size)
    barcode3 = np.concatenate([snp_blk > 0, [0]])
    snp_blk = close_pair_utils.to_block(between_snp_vec2, blk_size)
    barcode4 = np.concatenate([snp_blk > 0, [0]])

    # xlim = min(xlim, len(snp_blk) - 1)
    axes[0].imshow(np.expand_dims(barcode1, axis=0), aspect='auto',
                   cmap=mpl.colors.ListedColormap(['white', config.within_host_color]), interpolation='nearest')
    axes[1].imshow(np.expand_dims(barcode2, axis=0), aspect='auto',
                   cmap=mpl.colors.ListedColormap(['white', config.within_host_color]), interpolation='nearest')
    axes[2].imshow(np.expand_dims(barcode3, axis=0), aspect='auto',
                   cmap=mpl.colors.ListedColormap(['white', config.between_host_color]), interpolation='nearest')
    axes[3].imshow(np.expand_dims(barcode4, axis=0), aspect='auto',
                   cmap=mpl.colors.ListedColormap(['white', config.between_host_color]), interpolation='nearest')

    # axes[0].plot(within_snp_vec1, linewidth=0.3)
    # axes[1].plot(within_snp_vec2, linewidth=0.3)
    # axes[2].plot(between_snp_vec1, linewidth=0.3)
    # axes[3].plot(between_snp_vec2, linewidth=0.3)
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xticklabels([])
    axes[3].set_xticklabels([])
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[2].set_xticks([])
    axes[3].set_xticks([])

    ymin = axes[1].get_ylim()[0]

    # axes[2].arrow(x=110, y=-2, dx=0, dy=10, width=2, facecolor='red', edgecolor='none', clip_on = False)
    xloc = 116.5
    axes[1].annotate('',
                xy=(xloc, ymin),
                xytext=(xloc, ymin + 0.55),
                arrowprops=dict(facecolor='tab:orange', shrink=0.0, width=3, headwidth=6, headlength=5))

    xmax = min(len(within_snp_vec1), len(within_snp_vec2), len(between_snp_vec1), len(between_snp_vec2))
    for ax in axes:
        ax.set_xlim([0, xmax / blk_size])
        ax.set_yticklabels([])
        ax.set_yticks([])


def save_interesting_genes(genes, path):
    gene_ids = np.array(['fig|'+gene for gene in genes])
    all_genes = pd.read_csv(os.path.join(config.data_directory, 'genome_features', '%s.csv' % species_name))
    gene_data = all_genes[all_genes['PATRIC ID'].isin(gene_ids)]
    gene_data.to_csv(path)


# setting up figures
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['legend.frameon'] = False

# fig = plt.figure(figsize=(7, 4.5))
#
# outer_grid = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 3.25], hspace=0.35, figure=fig)
#
# top_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1.8,1],wspace=0,subplot_spec=outer_grid[0])
#
# local_div_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.5, subplot_spec=top_grid[1])
# snp_grid_1 = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.3, subplot_spec=local_div_grid[0])
# snp_grid_2 = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.3, subplot_spec=local_div_grid[1])
#
# bottom_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[3., 1], wspace=0.3, subplot_spec=outer_grid[1])
#
# bottom_right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.3, subplot_spec=bottom_grid[1])

# zoomin panel
fig_zoom = plt.figure(figsize=(5.25, 2.7))
bottom_left_grid = gridspec.GridSpec(1, 3, width_ratios=[4, 1, 1], wspace=0.05, figure=fig_zoom)

fig_snps = plt.figure(figsize=(2.5, 1.8))
snp_grid = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.3, figure=fig_snps)

fig_max = plt.figure(figsize=(4, 1.2))
max_grid = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3, figure=fig_max)

zoomin_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.15, subplot_spec=bottom_left_grid[0])
zoomin1_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.3, 1], hspace=0.25, subplot_spec=zoomin_grid[0])
zoomin2_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.3, 1], hspace=0.25, subplot_spec=zoomin_grid[1])

histo_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.15, subplot_spec=bottom_left_grid[1])
histo1_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.3, 1], hspace=0.25, subplot_spec=histo_grid[0])
histo2_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.3, 1], hspace=0.25, subplot_spec=histo_grid[1])

bar_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], hspace=0.15, subplot_spec=bottom_left_grid[2])
bar1_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.3, 1], hspace=0.25, subplot_spec=bar_grid[0])
bar2_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.3, 1], hspace=0.25, subplot_spec=bar_grid[1])

zoomin_ax1 = fig_zoom.add_subplot(zoomin1_grid[1])
zoomin_ax2 = fig_zoom.add_subplot(zoomin2_grid[1])
copy_ax1 = fig_zoom.add_subplot(zoomin1_grid[0])
copy_ax2 = fig_zoom.add_subplot(zoomin2_grid[0])

histo_ax1 = fig_zoom.add_subplot(histo1_grid[1])
histo_ax2= fig_zoom.add_subplot(histo2_grid[1])
bar_ax1 = fig_zoom.add_subplot(bar1_grid[1])
bar_ax2= fig_zoom.add_subplot(bar2_grid[1])
# plotting
minimal_genes, maximal_genes = plot_allele_freq_zoomin([zoomin_ax1, zoomin_ax2], [histo_ax1, histo_ax2], [copy_ax1, copy_ax2], [bar_ax1, bar_ax2], ['700114218', '700171115'], plot_locations=False)

# snp panel
# adding axes
snp_ax1 = fig_snps.add_subplot(snp_grid[0])
snp_ax2 = fig_snps.add_subplot(snp_grid[1])
snp_ax3 = fig_snps.add_subplot(snp_grid[-2])
snp_ax4 = fig_snps.add_subplot(snp_grid[-1])

plot_example_snps([snp_ax1, snp_ax2, snp_ax3, snp_ax4])
snp_ax4.set_xlabel("SNVs along core genome")

# max run panels
max_run_ax1 = fig_max.add_subplot(max_grid[0])
max_run_ax2 = fig_max.add_subplot(max_grid[1])

plot_max_run_histo(max_run_ax1, 'Bacteroides_vulgatus_57955_same_clade')
plot_max_run_histo(max_run_ax2, 'Eubacterium_rectale_56927')
max_run_ax1.set_xlim([0, 10000])
max_run_ax1.set_xticks([0, 5000, 10000])
max_run_ax2.set_xlim([0, 5000])
max_run_ax2.set_ylabel('')

# also plot the mean transfer length
transfer_df_path = os.path.join(config.analysis_directory, 'closely_related/third_pass',
                                'Eubacterium_rectale_56927_all_transfers.pickle')
transfer_df = pd.read_pickle(transfer_df_path)
Er_mean = transfer_df['lengths'][(transfer_df['clonal divergence']<1e-4) & (transfer_df['clonal fraction'] > 0.75) & (transfer_df['types']==0)].mean() * config.second_pass_block_size

transfer_df_path = os.path.join(config.analysis_directory, 'closely_related/third_pass',
                                'Bacteroides_vulgatus_57955_all_transfers.pickle')
transfer_df = pd.read_pickle(transfer_df_path)
Bv_mean = transfer_df['lengths'][(transfer_df['clonal divergence']<1e-4) & (transfer_df['clonal fraction'] > 0.75) & (transfer_df['types']==0)].mean() * config.second_pass_block_size
max_run_ax1.plot([], [], color=config.between_host_color, label='Between hosts')
max_run_ax1.plot([], [], color=config.within_host_color, label='Within hosts')
max_run_ax1.legend(loc='lower center', bbox_to_anchor=(1.1, 1.0), ncol=2,)
l = max_run_ax1.axvline(x=Bv_mean, linestyle='--', linewidth=0.5, color='grey', label='mean transfer length')
max_run_ax2.axvline(x=Er_mean, linestyle='--', linewidth=0.5, color='grey')


fig_zoom.savefig(os.path.join(config.figure_directory, 'final_fig', 'fig5_zoom.pdf'), bbox_inches="tight", dpi=600)
fig_snps.savefig(os.path.join(config.figure_directory, 'final_fig', 'fig5_snps.pdf'), bbox_inches="tight", dpi=600)
fig_max.savefig(os.path.join(config.figure_directory, 'final_fig', 'fig5_max_run.pdf'), bbox_inches="tight", dpi=600)