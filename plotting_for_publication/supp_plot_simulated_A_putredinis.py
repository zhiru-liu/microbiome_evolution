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
from HGT_scripts.close_pair_stage4_plot_trendline import prepare_trend_line

def preprocess_data(data):
    counts, full_df = close_pair_utils.merge_and_filter_transfers(
        data, merge_threshold=0, filter_threshold=0, ignore_pairs=True)
    counts = np.array(counts)
    return counts, full_df

def plot_between_within_clade(ax, est_Ts, counts):
    core_genome_len = 1457870
    counts = counts * 1e6 / core_genome_len
    # between_counts = between_counts * 1e6 / core_genome_len
    print("Estimated transfer/divergence is {:e}".format(np.mean(counts[est_Ts>0]/est_Ts[est_Ts>0])))

    x = est_Ts
    y = counts
    trend_data = prepare_trend_line(x, y)

    s = ax.scatter(est_Ts, counts, s=1)

    ax.plot(trend_data[0], trend_data[1])
    ax.fill_between(trend_data[0], trend_data[1] - trend_data[2],
                    trend_data[1] + trend_data[2], alpha=0.25, color='tab:blue')

    ax.set_xlim([0, 2e-4])
    # ax.set_ylim([-40, 40])
    # ax.set_ylim([0, 40])
    # ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    # ax.set_yticklabels(['15', '10', '5', '0', '5', '10', '15'])
    ax.set_xticks([0, 1e-4, 2e-4])
    ax.set_xticklabels([0, 1, 2])
    # ax.set_yticks([-40, -20, 0, 20, 40])
    # ax.set_yticklabels(['40', '20', '0', '20', '40'])
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.set_ylabel("Detected transfers")
    ax.set_xlabel(r"Clonal divergence ($\times 10^{-4}$)")
    # l1 = ax.legend([s1], ['within-clade'], loc='upper left', bbox_to_anchor=(0, 1))
    # ax.legend([s2], ['between-clade'], loc='lower left', bbox_to_anchor=(0,0))
    # ax.add_artist(l1)
    return


# set up figure
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize']  = 'small'

fig, ax = plt.subplots(1, 1, figsize=(2., 1.5))
# gs_lf = gridspec.GridSpec(1, 1)
# gs_rt = gridspec.GridSpec(1, 1)
# between_within_count_ax = fig.add_subplot(gs_lf[0,0])
# between_within_len_ax = fig.add_subplot(gs_rt[0,0])

# # now the plots are on top of each other, we'll have to adjust their edges so that they won't overlap
# gs_lf.update(right=0.48, top=0.8, bottom=0.2)
# gs_rt.update(left=0.63, top=0.76, bottom=0.24)

path = os.path.join(config.analysis_directory, 'HMM_validation', 'Alistipes_putredinis_61533.pickle')
genome_len = 2.5e5
data = pickle.load(open(path, 'rb'), encoding='latin1')

true_counts = np.array(data['true counts'])
true_between_counts = np.array(data['true between clade counts'])
true_within_counts = true_counts - true_between_counts
true_Ts = np.array(data['true T'])
true_divs = np.array(data['true div'])
true_lens = np.concatenate(data['true lengths'])
true_total_lens = np.array([np.sum(x) for x in data['true lengths']])
est_Ts = np.array(data['T est'])
total_counts, full_df = preprocess_data(data)
plot_between_within_clade(ax, true_divs, total_counts)
# plot_between_within_clade(ax, true_divs, true_counts)

fig.savefig(os.path.join(config.figure_directory, "supp_simulated_Ap_.pdf"), bbox_inches="tight")
