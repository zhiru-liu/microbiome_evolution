import os
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
sys.path.append("..")
import config
from utils import close_pair_utils

# plotting functions
pi_color = 'tab:grey'
# within_color = 'tab:blue'
within_color = '#2b83ba'
# between_color = 'tab:orange'
between_color = '#fdae61'
snp_color = 'tab:red'

def plot_typical_pair(ax, dh, pair):
    cache_file = "cached_close_pair_{}.csv".format(pair)
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
    ax.set_xlim([0, 260000])
    ax.set_xlabel('Synonymous core genome location')
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))


def plot_example_pair(ax, dh, pair, full_df, if_legend=True):
    cache_file = "cached_close_pair_{}.csv".format(pair)
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
    for _, row in sub_df.iterrows():
        start = row['starts'] * block_size
        end = (row['ends'] + 1) * block_size
        if row['types'] == 0:
            within_ys[start:end] = 1
        else:
            between_ys[start:end] = 1

    ax.plot(xs, -between_ys * 0.03, color=between_color, linewidth=1)
    ax.plot(xs, -within_ys * 0.03, color=within_color, linewidth=1)

    ax.plot(local_pi, label='local heterozygosity', color=pi_color, linewidth=1)
    ax.plot(shown_snp_locs, np.zeros(len(shown_snp_locs)), '|', color=snp_color, label='individual snps', markersize=2)

    ax.set_yticks((-0.03, 0.0, 0.04, 0.08))
    labels = ['transfer', '0.0', '0.04', '0.08']
    if if_legend:
        ax.legend()
    ax.set_yticklabels(labels)
    # ax.set_title(pair)
    ax.set_ylim([-0.035, 0.10])

    ax.set_xlim([0, 260000])
    ax.set_xlabel('Synonymous core genome location')


def plot_scatter(ax, x, y1, y2, if_trend_line=True):
    s1 = ax.scatter(x, y1, s=2, c=within_color, label='Within clade transfers')
    s2 = ax.scatter(x, -y2, s=2, c=between_color, label='Between clade transfers')
    trend_directory = os.path.join(config.plotting_intermediate_directory, "B_vulgatus_trend_line.csv")
    if os.path.exists(trend_directory) and if_trend_line:
        # add trend line
        trend_data = pd.read_csv(trend_directory)
        ax.plot(trend_data['within_x'], trend_data['within_y'])
        ax.plot(trend_data['between_x'], -trend_data['between_y'])
        ax.fill_between(trend_data['within_x'], trend_data['within_y'] - trend_data['within_sigma'],
                           trend_data['within_y'] + trend_data['within_sigma'], alpha=0.25)
        ax.fill_between(trend_data['between_x'], - trend_data['between_y'] - trend_data['between_sigma'],
                           - trend_data['between_y'] + trend_data['between_sigma'], alpha=0.25)

    ax.plot(x, np.zeros(x.shape), 'k-')

    ax.set_xlim([0, 2e-4])
    ax.set_ylim([-20, 40])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    ax.set_yticks([-20, -10, 0, 10, 20, 30])
    ax.set_yticklabels(['20', '10', '0', '10', '20', '30'])
    ax.xaxis.major.formatter._useMathText = True

    ax.set_ylabel("Number of transfers")
    ax.set_xlabel("Clonal divergence")


def plot_distributions(fig, ax, within_lens, between_lens, inset_location=[0.7, 0.18, 0.15, 0.15]):
    bins = np.arange(max(max(within_lens), max(between_lens)))
    _ = ax.hist(within_lens, density=True, bins=bins, cumulative=-1, histtype="step",
                label="within clade", color=within_color)
    _ = ax.hist(between_lens, density=True, bins=bins, cumulative=-1, histtype="step",
                label="between clade", color=between_color)

    ax.set_xlim([0, 20000])

    left, bottom, width, height = inset_location
    ax2 = fig.add_axes([left, bottom, width, height])
    _ = ax2.hist(within_lens, density=True, bins=bins, cumulative=-1, histtype="step",
                 label="within clade", color=within_color)
    _ = ax2.hist(between_lens, density=True, bins=bins, cumulative=-1, histtype="step",
                 label="between clade", color=between_color)
    ax2.set_yscale('log')
    ax2.set_xlim([0, 40000])

    ax.legend()
    ax.set_xlabel('Transfer length / bps')
    ax.set_ylabel('Survival prob')



######################################################################
# preparing data
######################################################################
if __name__ == "__main__":
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['lines.linewidth'] = 0.5

    species_name = 'Bacteroides_vulgatus_57955'
    # dh = parallel_utils.DataHoarder(species_name, mode="QP")
    dh = None

    save_path = config.B_vulgatus_data_path
    BLOCK_SIZE = config.second_pass_block_size
    clonal_divs, within_counts, between_counts, full_df = close_pair_utils.prepare_HMM_results_for_B_vulgatus(
        save_path, 0.75, cache_intermediate=True)
    within_lens = full_df[full_df['types']==0]['lengths'].to_numpy().astype(int) * BLOCK_SIZE
    between_lens = full_df[full_df['types']==1]['lengths'].to_numpy().astype(int) * BLOCK_SIZE

    print("Mean within transfer length: {}".format(np.mean(within_lens)))
    print("Mean between transfer length: {}".format(np.mean(between_lens)))

    # mapping out grid
    fig = plt.figure(figsize=(7, 5.5))
    outer_grid = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 2.5], hspace=0.3, figure=fig)

    top_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[2,1.5],wspace=0,subplot_spec=outer_grid[0])

    top_right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.37, 2],hspace=0.4,subplot_spec=top_grid[1])
    example_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1,1],hspace=0.2,subplot_spec=top_right_grid[1])

    bottom_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 1], wspace=0.2, subplot_spec=outer_grid[1])

    # adding axes
    ct_ax = fig.add_subplot(bottom_grid[0])

    len_dist_ax = fig.add_subplot(bottom_grid[1])

    ex0_ax = fig.add_subplot(top_right_grid[0])

    ex1_ax = fig.add_subplot(example_grid[0])

    ex2_ax = fig.add_subplot(example_grid[1])

    ######################################################################
    # example species
    ######################################################################

    plot_typical_pair(ex0_ax, dh, (0, 128))
    plot_example_pair(ex1_ax, dh, (128, 170), full_df, if_legend=False)
    plot_example_pair(ex2_ax, dh, (39, 74), full_df, if_legend=False)
    ex0_ax.set_xticklabels([])
    ex1_ax.set_xticklabels([])
    ex0_ax.set_xlabel('')
    ex1_ax.set_xlabel('')


    ######################################################################
    # plot transfer counts
    ######################################################################

    plt_prop_cycle = plt.rcParams['axes.prop_cycle']
    plt_colors = plt_prop_cycle.by_key()['color']
    # fig, axes = plt.subplots(2, 1, sharex=True)
    # within_ct_ax.get_shared_x_axes().join(within_ct_ax, between_ct_ax)
    # s1 = within_ct_ax.scatter(T_approxs, within_counts, s=1, c=plt_colors[0])
    # s2 = between_ct_ax.scatter(T_approxs, between_counts, s=1, c=plt_colors[1])

    plot_scatter(ct_ax, clonal_divs, y1=within_counts, y2=between_counts, if_trend_line=True)

    ######################################################################
    # plot transfer length distribution
    ######################################################################
    plot_distributions(fig, len_dist_ax, within_lens, between_lens)
    fig.patch.set_alpha(0.0)
    fig.savefig('test_close_pair_cf_0.75.pdf', bbox_inches="tight")
