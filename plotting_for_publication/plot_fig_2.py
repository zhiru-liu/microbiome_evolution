import os
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import pandas as pd
sys.path.append("..")
import config
from utils import close_pair_utils, parallel_utils

# plotting functions
pi_color = 'tab:grey'
# within_color = 'tab:blue'
within_color = '#2b83ba'
# between_color = 'tab:orange'
between_color = '#fdae61'
snp_color = 'tab:red'


def plot_typical_pair(ax, dh, pair):
    cache_file = os.path.join(config.plotting_intermediate_directory, "cached_close_pair_{}.csv".format(pair))
    if os.path.exists(cache_file):
        snp_vec = np.loadtxt(cache_file)
    else:
        snp_vec, _ = dh.get_snp_vector(pair)
        np.savetxt(cache_file, snp_vec)

    window_size = 1000
    local_pi = np.convolve(snp_vec, np.ones(window_size) / float(window_size), mode='same')
    snp_locs = np.nonzero(snp_vec)[0]
    shown_snp_locs = snp_locs

    ax.plot(local_pi, label='local heterozygosity', color=pi_color, linewidth=1)
    ax.plot(shown_snp_locs, np.zeros(shown_snp_locs.shape), '|', color=snp_color, label='individual snps', markersize=2)
    ax.set_ylim([-0.005, 0.05])
    ax.set_yticks((0.0, 0.04))
    labels = ['0.0', '0.04']
    ax.set_xlim([0, 264000])
    ax.set_xticks([0, 5e4, 1e5, 1.5e5, 2e5, 2.5e5])
    ax.set_xticklabels(['0', '50', '100', '150', '200', '250'])
    ax.set_xlabel('Location along core genome (kb)')
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))


def plot_example_pair(ax, dh, pair, full_df, if_legend=True, patch_ymin=None):
    cache_file = os.path.join(config.plotting_intermediate_directory, "cached_close_pair_{}.csv".format(pair))
    if os.path.exists(cache_file):
        snp_vec = np.loadtxt(cache_file)
    else:
        snp_vec, _ = dh.get_snp_vector(pair)
        np.savetxt(cache_file, snp_vec)

    window_size = 1000
    local_pi = np.convolve(snp_vec, np.ones(window_size) / float(window_size), mode='same')

    snp_locs = np.nonzero(snp_vec)[0]
    shown_snp_locs = snp_locs

    block_size = 10
    sub_df = full_df[full_df['pairs'] == pair]

    xs = np.arange(len(snp_vec))
    between_ys = np.zeros(xs.shape)
    within_ys = np.zeros(xs.shape)
    patches = []  # for collecting all the rectangles
    patch_height = 1e-2
    patches.append(Rectangle((0, patch_ymin), 264000, patch_height, facecolor='grey', edgecolor=None, alpha=0.3, clip_on=False))
    for _, row in sub_df.iterrows():
        start = row['starts'] * block_size
        end = (row['ends'] + 1) * block_size
        if row['types'] == 0:
            color = within_color
        else:
            color = between_color
        # if row['types'] == 0:
        #     within_ys[start:end] = 1
        # else:
        #     between_ys[start:end] = 1
        ax.axvspan(start, end, facecolor=color, alpha=0.3)
        patches.append(Rectangle((start, patch_ymin), end-start, patch_height, facecolor=color, edgecolor=None, clip_on=False))
    for patch in patches:
        ax.add_patch(patch)
    ax.axvspan(-2, -1, facecolor=within_color, alpha=0.3, label='within-clade')
    ax.axvspan(-2, -1, facecolor=between_color, alpha=0.3, label='between-clade')
    # ax.plot(xs, -between_ys * 0.03, color=between_color, linewidth=1)
    # ax.plot(xs, -within_ys * 0.03, color=within_color, linewidth=1)

    ax.plot(local_pi, label='SNP density', color=pi_color, linewidth=1)
    ax.plot(shown_snp_locs, np.zeros(len(shown_snp_locs)), '|', color=snp_color, label='individual snps', markersize=2,
            markeredgewidth=0.5)

    ax.set_yticks((0.0, 0.04, 0.08))
    labels = ['0', '4', '8']
    if if_legend:
        ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))
    ax.set_yticklabels(labels)
    # ax.set_title(pair)
    ax.set_ylim([-0.002, 0.08])

    ax.set_xlim([0, 264000])
    ax.set_xticks([0, 5e4, 1e5, 1.5e5, 2e5, 2.5e5])
    ax.set_xticklabels(['0', '50', '100', '150', '200', '250'])
    ax.set_xlabel('Location along core genome (kb)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_scatter(ax, x, y1, y2, mask, if_trend_line=True):
    print("Estimated within-clade transfer/divergence is {:e}".format(np.mean(y1[x>0]/x[x>0])))
    s1 = ax.scatter(x[mask], y1[mask], s=2, c=within_color, label='Within clade transfers')
    s2 = ax.scatter(x[mask], -y2[mask], s=2, c=between_color, label='Between clade transfers')
    # s1 = ax.scatter(x[~mask], y1[~mask], s=0.5, c='grey')
    # s2 = ax.scatter(x[~mask], -y2[~mask], s=0.5, c='grey')
    trend_directory = os.path.join(config.plotting_intermediate_directory, "B_vulgatus_trend_line.csv")
    if os.path.exists(trend_directory) and if_trend_line:
        # add trend line
        trend_data = pd.read_csv(trend_directory)
        mean_count = trend_data['within_y'][np.argmin(np.abs(trend_data['within_x'] - 1e-4))]
        print("mean at 1e-4: {:.2f}".format(mean_count))
        ax.plot(trend_data['within_x'], trend_data['within_y'])
        ax.plot(trend_data['between_x'], -trend_data['between_y'])
        ax.fill_between(trend_data['within_x'], trend_data['within_y'] - trend_data['within_sigma'],
                           trend_data['within_y'] + trend_data['within_sigma'], alpha=0.25)
        ax.fill_between(trend_data['between_x'], - trend_data['between_y'] - trend_data['between_sigma'],
                           - trend_data['between_y'] + trend_data['between_sigma'], alpha=0.25)

    ax.plot(x, np.zeros(x.shape), 'k-')

    # ax.set_xlim([0, 2e-4])
    ax.set_xlim([0, config.Bv_clonal_div_cutoff])
    ax.set_ylim([-7.5, 17.5])
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.set_xticks([0, 0.5e-4, 1e-4])
    ax.set_xticklabels([0, 0.5, 1])

    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.set_yticklabels(['5', '0', '5', '10', '15'])
    ax.xaxis.major.formatter._useMathText = True

    ax.set_ylabel("# transfers / Mb")
    ax.set_xlabel("Clonal divergence ($\\times 10^{-4}$)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_distributions(fig, ax, within_lens, between_lens, inset_location=[0.7, 0.18, 0.15, 0.15]):
    bins = np.arange(max(max(within_lens), max(between_lens)))
    _ = ax.hist(within_lens, density=True, bins=bins, cumulative=-1, histtype="step",
                color=within_color)
    _ = ax.hist(between_lens, density=True, bins=bins, cumulative=-1, histtype="step",
                color=between_color)
    ax.plot([],[], color=within_color, linestyle='-', label="within clade")
    ax.plot([],[], color=between_color, linestyle='-', label="between clade")

    ax.set_xlim([0, 150e3])

    # left, bottom, width, height = inset_location
    # ax2 = fig.add_axes([left, bottom, width, height])
    # _ = ax2.hist(within_lens, density=True, bins=bins, cumulative=-1, histtype="step",
    #              label="within clade", color=within_color)
    # _ = ax2.hist(between_lens, density=True, bins=bins, cumulative=-1, histtype="step",
    #              label="between clade", color=between_color)
    # ax2.set_yscale('log')
    # ax2.set_xlim([0, 40000])

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 0.78))
    ax.set_ylim([0,1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0','','','','1'])
    ax.set_xticks([0, 50e3, 100e3, 150e3])
    ax.set_xticklabels([0, 50, 100, 150])
    ax.set_xlabel(r'Transfer length, $\ell$ (kb)')
    ax.set_ylabel(r'Fraction $\geq \ell$')
    ax.yaxis.set_label_coords(-.1, .5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



######################################################################
# preparing data
######################################################################
if __name__ == "__main__":
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.fontsize'] = 'small'

    species_name = 'Bacteroides_vulgatus_57955'
    # dh = parallel_utils.DataHoarder(species_name, mode="QP")
    dh = None

    save_path = config.B_vulgatus_data_path
    BLOCK_SIZE = config.second_pass_block_size
    clonal_divs, within_counts, between_counts, filtered_full_df, cf_mask = close_pair_utils.prepare_HMM_results_for_B_vulgatus(
        save_path, config.clonal_fraction_cutoff, cache_intermediate=True, merge_threshold=0)
    cd_mask = clonal_divs <= config.Bv_clonal_div_cutoff

    print(np.sum((within_counts>0) | (between_counts>0)))

    transfer_df_path = os.path.join(config.analysis_directory, "closely_related", 'third_pass',
                 'Bacteroides_vulgatus_57955' + '_all_transfers.pickle')
    if not os.path.exists(transfer_df_path):
        raise RuntimeError("Please run stage 3a scripts to obtain full genome length and divergences")
    transfer_df = pd.read_pickle(transfer_df_path)
    transfer_df = transfer_df[transfer_df['clonal fraction']>=config.clonal_fraction_cutoff]  # TODO!
    transfer_df = transfer_df[transfer_df['clonal divergence']<=config.Bv_clonal_div_cutoff]  # TODO!

    # within_lens = full_df[full_df['types']==0]['lengths'].to_numpy().astype(int) * BLOCK_SIZE
    within_lens = transfer_df[transfer_df['types']==0]['transfer lengths (core genome)'].to_numpy()
    # between_lens = full_df[full_df['types']==1]['lengths'].to_numpy().astype(int) * BLOCK_SIZE
    between_lens = transfer_df[transfer_df['types']==1]['transfer lengths (core genome)'].to_numpy()

    print("Median within transfer length: {}; mean within transfer length: {}".format(np.median(within_lens), np.mean(within_lens)))
    print("Median between transfer length: {}; mean between transfer length: {}".format(np.median(between_lens), np.mean(between_lens)))
    print("Total number of close pairs: %d, detected transfers: %d, within transfers: %d, between transfers: %d, mean length: %f" % (
        len(clonal_divs[cf_mask & cd_mask]), transfer_df.shape[0],
        np.sum(transfer_df['types'] == 0), np.sum(transfer_df['types'] == 1), np.mean(np.concatenate([within_lens, between_lens]))))

    # mapping out grid
    fig = plt.figure(figsize=(7, 1.5))
    gs_scatter = gridspec.GridSpec(1, 1)
    gs_len = gridspec.GridSpec(1, 1)
    gs_example = gridspec.GridSpec(2,1)

    gs_example.update(left=0.2, right=0.48, top=0.95, bottom=0.24, hspace=0.4)
    gs_scatter.update(left=0.56, right=0.78, top=0.95, bottom=0.24)
    gs_len.update(left=0.83, right=0.98, top=0.95, bottom=0.24)

    # adding axes
    ct_ax = fig.add_subplot(gs_scatter[0,0])
    len_dist_ax = fig.add_subplot(gs_len[0,0])
    ex1_ax = fig.add_subplot(gs_example[0,0])
    ex2_ax = fig.add_subplot(gs_example[1,0])

    ######################################################################
    # example species
    ######################################################################
    fig.text(0.155, 0.55, "Divergence (%)", size=7, rotation=90.,verticalalignment ='center'
             )
    # plot_typical_pair(ex0_ax, dh, (0, 128))
    # plot_example_pair(ex1_ax, dh, (128, 170), full_df, if_legend=False)
    plot_example_pair(ex1_ax, dh, (54, 238), transfer_df, if_legend=False, patch_ymin=-1.2e-2)
    plot_example_pair(ex2_ax, dh, (39, 74), transfer_df, if_legend=False, patch_ymin=8.e-2)
    # ex0_ax.set_xticklabels([])
    ex1_ax.set_xticklabels([])
    # ex0_ax.set_xlabel('')
    ex1_ax.set_xlabel('')
    ex1_ax.set_xticks([])


    ######################################################################
    # plot transfer counts
    ######################################################################

    plt_prop_cycle = plt.rcParams['axes.prop_cycle']
    plt_colors = plt_prop_cycle.by_key()['color']
    # fig, axes = plt.subplots(2, 1, sharex=True)
    # within_ct_ax.get_shared_x_axes().join(within_ct_ax, between_ct_ax)
    # s1 = within_ct_ax.scatter(T_approxs, within_counts, s=1, c=plt_colors[0])
    # s2 = between_ct_ax.scatter(T_approxs, between_counts, s=1, c=plt_colors[1])

    plot_scatter(ct_ax, clonal_divs, y1=within_counts, y2=between_counts, mask=cf_mask, if_trend_line=True)

    ######################################################################
    # plot transfer length distribution
    ######################################################################
    plot_distributions(fig, len_dist_ax, within_lens, between_lens)
    fig.patch.set_alpha(0.0)

    ex1_ax.text(-0.165, 1., "B", transform=ex1_ax.transAxes,
                    fontsize=9, fontweight='bold', va='top', ha='left')
    ct_ax.text(0.05, 1., "C", transform=ct_ax.transAxes,
                     fontsize=9, fontweight='bold', va='top', ha='left')
    len_dist_ax.text(0.15, 1., "D", transform=len_dist_ax.transAxes,
                     fontsize=9, fontweight='bold', va='top', ha='left')
    ex1_ax.text(1, 1, "Pair 1", transform=ex1_ax.transAxes,
                fontsize=6, va='top', ha='right',
                bbox=dict(fc='w',boxstyle='square,pad=0.15', ec='white',))
    ex2_ax.text(1, 0.8, "Pair 2", transform=ex2_ax.transAxes,
                fontsize=6, backgroundcolor='white', va='top', ha='right',
                bbox=dict(fc='w',boxstyle='square,pad=0.15', ec='white'))

    fig.savefig(os.path.join(config.figure_directory, 'final_fig', 'fig2.pdf'))
