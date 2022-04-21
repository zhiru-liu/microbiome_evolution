import os
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import itertools
sys.path.append("..")
import config
from utils import close_pair_utils


def preprocess_data(data):
    within_ct, between_ct, full_df = close_pair_utils.merge_and_filter_transfers(
        data, separate_clade=True, merge_threshold=0, filter_threshold=0, ignore_pairs=True)
    within_ct = np.array(within_ct)
    between_ct = np.array(between_ct)
    total_ct = np.array(within_ct) + np.array(between_ct)
    return total_ct, within_ct, between_ct, full_df


def plot_count_correlation(ax, true_counts, true_Ts, total_ct):
    # preparing x y values
    ax.set_aspect('equal')
    markers = itertools.cycle(('+', '.', 'x', '^'))
    for T in np.unique(true_Ts):
        mask = true_Ts == T
        ax.plot(true_counts[mask], total_ct[mask], marker=markers.next(), markersize=2, linestyle='none', label="pi=%.e" % T)
    xs = np.linspace(0, ax.get_xlim()[1])
    ax.plot(xs, xs, '--', label='y=x')
    ax.legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("True")
    ax.set_ylabel("Detected")
    return


def plot_clonal_T_estimation(ax, true_Ts, est_Ts):
    plotting_dat = []
    Ts = np.unique(true_Ts)
    for T in Ts:
        plotting_dat.append(est_Ts[T == true_Ts])

    _ = ax.boxplot(plotting_dat, medianprops={'color':'orange'}, flierprops={'markersize': 1})
    xs = np.linspace(1, 10, 10)
    ys = np.linspace(1e-5, 1e-4, 10)
    ax.plot(xs, ys, '--', label='y=x')
    ax.plot([],'-', color='orange',label='median')
    ax.set_xlabel(r"True $2\mu T$ $(* 10^{-5})$")
    ax.set_ylabel(r"$\pi$ in clonal region")
    ax.legend()
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    return


def plot_length_distributions(ax, true_lens, detected_lens):
    _ = ax.hist(detected_lens * config.second_pass_block_size, histtype='step', bins=100, density=True, label='Detected dist')
    _ = ax.hist(true_lens, histtype='step', bins=100, density=True, label='True dist')
    ax.set_xlabel('Transfer length / bps')
    ax.set_ylabel('Density')
    ax.legend()
    return


def plot_between_within_length_distributions(ax, full_df):
    within_lens = full_df.loc[full_df['types']==0, 'lengths'].astype(float)
    between_lens = full_df.loc[full_df['types']==1, 'lengths'].astype(float)
    _ = ax.hist(within_lens * config.second_pass_block_size, cumulative=-1,
                histtype='step', bins=100, density=True, label='Within-clade')
    _ = ax.hist(between_lens * config.second_pass_block_size, cumulative=-1,
                histtype='step', bins=100, density=True, label='Between-clade')
    ax.set_xlabel('Transfer length / bps')
    ax.set_ylabel('Prob greater')
    ax.legend()
    return


def plot_between_within_clade(ax, est_Ts, within_counts, between_counts):
    _ = ax.scatter(est_Ts, within_counts, s=1, label='Within-clade')
    _ = ax.scatter(est_Ts, -np.array(between_counts), s=1, label='Between-clade')
    ax.plot(est_Ts, np.zeros(est_Ts.shape), 'k-')
    ax.set_xlim([0, 2e-4])
    ax.set_ylim([-40, 40])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.set_yticklabels(['40', '20', '0', '20', '40'])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.set_ylabel("Detected transfers")
    ax.set_xlabel("Clonal divergence")
    ax.legend(ncol=2)
    return

# set up figure
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

fig = plt.figure(figsize=(4.5, 5.))
outer_grid = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[1, 1, 1], hspace=0.6, figure=fig)
upper_grid = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, width_ratios=[1.1, 1], wspace=0.6, subplot_spec=outer_grid[0])
mid_grid = gridspec.GridSpecFromSubplotSpec(ncols=3, nrows=1, width_ratios=[0.0, 0.4, 0.3], wspace=0.1, subplot_spec=outer_grid[1])
lower_grid = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, width_ratios=[1, 1], wspace=0.6, subplot_spec=outer_grid[2])

count_ax = fig.add_subplot(mid_grid[1])
true_len_ax = fig.add_subplot(upper_grid[1])
T_est_ax = fig.add_subplot(upper_grid[0])
between_within_len_ax = fig.add_subplot(lower_grid[1])
between_within_count_ax = fig.add_subplot(lower_grid[0])

# load up data
path = os.path.join(config.analysis_directory, 'HMM_validation', 'Bacteroides_vulgatus_57955.pickle')
data = pickle.load(open(path, 'rb'))
true_counts = np.array(data['true counts'])
true_Ts = np.array(data['true T'])
true_lens = np.concatenate(data['true lengths'])
est_Ts = np.array(data['T est'])
total_counts, within_counts, between_counts, full_df = preprocess_data(data)

plot_count_correlation(count_ax, true_counts, true_Ts, total_counts)
plot_clonal_T_estimation(T_est_ax, true_Ts, est_Ts)
plot_length_distributions(true_len_ax, true_lens, full_df['lengths'].astype(float))
# plot_between_within_clade(between_within_count_ax, est_Ts, within_counts, between_counts)
plot_between_within_clade(between_within_count_ax, est_Ts, within_counts, between_counts)
plot_between_within_length_distributions(between_within_len_ax, full_df)

fig.savefig("test_validation.pdf", bbox_inches="tight")
