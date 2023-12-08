"""
Use python3 with package statsmodels for this script in order to add trend line
"""
import os
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.gridspec as gridspec
import pickle
import itertools
sys.path.append("..")
import config
from utils import close_pair_utils
from scripts.close_pair_stage4_plot_trendline import prepare_trend_line

def preprocess_data(data):
    within_ct, between_ct, full_df = close_pair_utils.merge_and_filter_transfers(
        data, separate_clade=True, merge_threshold=0, filter_threshold=0, ignore_pairs=True)
    within_ct = np.array(within_ct)
    between_ct = np.array(between_ct)
    total_ct = np.array(within_ct) + np.array(between_ct)
    return total_ct, within_ct, between_ct, full_df

def plot_between_within_length_distributions(ax, full_df):
    within_lens = full_df.loc[full_df['types']==0, 'lengths'].astype(float)
    between_lens = full_df.loc[full_df['types']==1, 'lengths'].astype(float)
    print("Median within transfer length: {}; mean within transfer length: {}".format(np.median(within_lens), np.mean(within_lens)))
    print("Median between transfer length: {}; mean between transfer length: {}".format(np.median(between_lens), np.mean(between_lens)))
    conversion_factor = 7.14  # 4D to full genome length conversion
    _ = ax.hist(within_lens * config.second_pass_block_size * conversion_factor, cumulative=-1,
                histtype='step', bins=100, density=True, label='Within-clade')
    _ = ax.hist(between_lens * config.second_pass_block_size * conversion_factor, cumulative=-1,
                histtype='step', bins=100, density=True, label='Between-clade')
    ax.set_xticks([0, 50e3, 100e3, 150e3])
    ax.set_xticklabels([0, 50, 100, 150])
    ax.set_xlabel('Transfer length / kbps')
    ax.set_ylabel('Prob greater')
    ax.legend()
    return

def plot_between_within_clade(ax, est_Ts, within_counts, between_counts):
    core_genome_len = 2057681
    within_counts = within_counts * 1e6 / core_genome_len
    between_counts = between_counts * 1e6 / core_genome_len
    print("Estimated transfer/divergence is {:e}".format(np.mean(within_counts[est_Ts>0]/est_Ts[est_Ts>0])))

    x = est_Ts
    y1 = within_counts
    y2 = between_counts
    within_data = prepare_trend_line(x, y1)
    between_data = prepare_trend_line(x, y2)


    s1 = ax.scatter(est_Ts, within_counts, s=1, label='Within-clade')
    s2 = ax.scatter(est_Ts, -np.array(between_counts), s=1, label='Between-clade')
    ax.plot(est_Ts, np.zeros(est_Ts.shape), 'k-')

    print("Mean at 1e-4: {:.2f}".format(within_data[1][np.abs(within_data[0]-1e-4).argmin()]))
    ax.plot(within_data[0], within_data[1])
    ax.plot(between_data[0], -between_data[1])
    ax.fill_between(within_data[0], within_data[1] - within_data[2],
                    within_data[1] + within_data[2], alpha=0.25, color='tab:blue')
    ax.fill_between(between_data[0], - between_data[1] - between_data[2],
                    -between_data[1] + between_data[2], alpha=0.25, color='tab:orange')

    ax.set_xlim([0, 2e-4])
    # ax.set_ylim([-40, 40])
    ax.set_ylim([-17.5, 17.5])
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_yticklabels(['15', '10', '5', '0', '5', '10', '15'])
    ax.set_xticks([0, 1e-4, 2e-4])
    ax.set_xticklabels([0, 1, 2])
    # ax.set_yticks([-40, -20, 0, 20, 40])
    # ax.set_yticklabels(['40', '20', '0', '20', '40'])
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.set_ylabel("Detected transfers")
    ax.set_xlabel(r"Clonal divergence ($\times 10^{-4}$)")
    l1 = ax.legend([s1], ['within-clade'], loc='upper left', bbox_to_anchor=(0, 1))
    ax.legend([s2], ['between-clade'], loc='lower left', bbox_to_anchor=(0,0))
    ax.add_artist(l1)
    return


# set up figure
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize']  = 'small'

fig = plt.figure(figsize=(4.5, 2.))
# fig = plt.figure(figsize=(6, 2.8))
gs_lf = gridspec.GridSpec(1, 1)
gs_rt = gridspec.GridSpec(1, 1)
between_within_count_ax = fig.add_subplot(gs_lf[0,0])
between_within_len_ax = fig.add_subplot(gs_rt[0,0])

# now the plots are on top of each other, we'll have to adjust their edges so that they won't overlap
gs_lf.update(right=0.48, top=0.76, bottom=0.24)
gs_rt.update(left=0.63, top=0.76, bottom=0.24)

path = os.path.join(config.analysis_directory, 'HMM_validation', 'Bacteroides_vulgatus_57955.pickle')
genome_len = 2.8e5
data = pickle.load(open(path, 'rb'), encoding='latin1')

true_counts = np.array(data['true counts'])
true_between_counts = np.array(data['true between clade counts'])
true_within_counts = true_counts - true_between_counts
true_Ts = np.array(data['true T'])
true_divs = np.array(data['true div'])
true_lens = np.concatenate(data['true lengths'])
true_total_lens = np.array([np.sum(x) for x in data['true lengths']])
est_Ts = np.array(data['T est'])
total_counts, within_counts, between_counts, full_df = preprocess_data(data)
plot_between_within_clade(between_within_count_ax, est_Ts, within_counts, between_counts)
# plot_between_within_clade(between_within_count_ax, true_divs, true_within_counts, true_between_counts)
plot_between_within_length_distributions(between_within_len_ax, full_df)

fig.savefig(os.path.join(config.figure_directory, 'supp', "S10_supp_simulated_Bv.pdf"), bbox_inches="tight")
