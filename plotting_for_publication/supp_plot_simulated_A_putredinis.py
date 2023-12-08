"""
Use python3 with package statsmodels for this script in order to add trend line
"""
import os
import numpy as np
import sys
import pandas as pd
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
    counts, full_df = close_pair_utils.merge_and_filter_transfers(
        data, merge_threshold=0, filter_threshold=0, ignore_pairs=True)
    counts = np.array(counts)
    return counts, full_df

def plot_scatter(ax, est_Ts, counts):
    core_genome_len = 1457870
    counts = counts * 1e6 / core_genome_len
    # between_counts = between_counts * 1e6 / core_genome_len
    print("Estimated transfer/divergence is {:e}".format(np.mean(counts[est_Ts>0]/est_Ts[est_Ts>0])))

    x = est_Ts
    y = counts
    trend_data = prepare_trend_line(x, y)

    s = ax.scatter(est_Ts, counts, s=1)

    print("Mean at 2.5e-5: {:.2f}".format(trend_data[1][np.abs(trend_data[0]-2.5e-5).argmin()]))
    print("Mean at 5e-5: {:.2f}".format(trend_data[1][np.abs(trend_data[0]-5e-5).argmin()]))
    print("Mean at 1e-4: {:.2f}".format(trend_data[1][np.abs(trend_data[0]-1e-4).argmin()]))
    ax.plot(trend_data[0], trend_data[1], color='tab:orange')
    ax.fill_between(trend_data[0], trend_data[1] - trend_data[2],
                    trend_data[1] + trend_data[2], alpha=0.25, color='tab:orange')

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
    # l1 = ax.legend([s1], ['within-clade'], loc='upper left', bbox_to_anchor=(0, 1))
    # ax.legend([s2], ['between-clade'], loc='lower left', bbox_to_anchor=(0,0))
    # ax.add_artist(l1)
    return


# set up figure
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize']  = 'small'

fig, axes = plt.subplots(2, 4, figsize=(6.5, 2.8))
plt.subplots_adjust(wspace=0.4, hspace=0.5)
# gs_lf = gridspec.GridSpec(1, 1)
# gs_rt = gridspec.GridSpec(1, 1)
# between_within_count_ax = fig.add_subplot(gs_lf[0,0])
# between_within_len_ax = fig.add_subplot(gs_rt[0,0])

# # now the plots are on top of each other, we'll have to adjust their edges so that they won't overlap
# gs_lf.update(right=0.48, top=0.8, bottom=0.2)
# gs_rt.update(left=0.63, top=0.76, bottom=0.24)
files = ['Alistipes_putredinis_61533_EM_025.pickle', 'Alistipes_putredinis_61533_EM_050.pickle', 'Alistipes_putredinis_61533_EM_100.pickle']
# rates = [r'2.5\times 10^{-5}', r'5\times 10^{-5}', r'1\times 10^{-4}']
rates = [r'$r/\mu=5.2$', r'$r/\mu=3.4$', r'$r/\mu=2.1$']
for i in range(3):
    path = os.path.join(config.analysis_directory, 'HMM_validation', files[i])
    genome_len = 2.5e5
    data = pickle.load(open(path, 'rb'), encoding='latin1')

    true_counts = np.array(data['true counts'])
    true_between_counts = np.array(data['true between clade counts'])
    true_within_counts = true_counts - true_between_counts
    true_Ts = np.array(data['true T'])
    true_divs = np.array(data['true div'])
    true_lens = np.concatenate(data['true lengths'])
    true_total_lens = np.array([np.sum(x) for x in data['true lengths']])
    cfs = np.array(data['clonal fraction'])
    mask = cfs > config.clonal_fraction_cutoff
    est_Ts = np.array(data['T est'])[mask]
    total_counts, full_df = preprocess_data(data)
    plot_scatter(axes[0, i], true_divs, true_counts)
    plot_scatter(axes[1, i], est_Ts, total_counts[mask])

    # axes[0, i].set_title("Rate @ ${}$".format(rates[i]))
    axes[0, i].set_title(rates[i])
    axes[1, i].set_xlabel(r"Clonal divergence ($\times 10^{-4}$)")

axes[1, 0].set_ylabel('Detected transfers')
axes[0, 0].set_ylabel('True transfers')

# next plot the real data
df = pd.read_pickle(os.path.join(config.analysis_directory, 'closely_related', 'third_pass', 'Alistipes_putredinis_61533.pickle'))
cf_cutoff = config.clonal_fraction_cutoff
cf = df['clonal fractions']
x = df['clonal divs'].to_numpy()[cf >= cf_cutoff]
y = df['transfer counts'].to_numpy()[cf >= cf_cutoff]
plot_scatter(axes[1, 3], x, y)
axes[1, 3].set_title("Real data")
axes[1, 3].set_xlabel(r"Clonal divergence ($\times 10^{-4}$)")

fig.delaxes(axes[0, 3])

for i in range(1, 4):
    axes[0, i].set_ylim(axes[0, 0].get_ylim())
    axes[1, i].set_ylim(axes[1, 0].get_ylim())

fig.savefig(os.path.join(config.figure_directory, "supp", "supp_simulated_Ap.pdf"), bbox_inches="tight")
