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


def preprocess_data(data):
    within_ct, between_ct, full_df = close_pair_utils.merge_and_filter_transfers(
        data, separate_clade=True, merge_threshold=0, filter_threshold=0, ignore_pairs=True)
    within_ct = np.array(within_ct)
    between_ct = np.array(between_ct)
    total_ct = np.array(within_ct) + np.array(between_ct)
    return total_ct, within_ct, between_ct, full_df


def plot_count_correlation(fig, ax, true_counts, true_Ts, total_ct, true_recombined_fractions):
    # preparing x y values
    markers = itertools.cycle(('+', '.', 'x', '^'))
    # for T in np.unique(true_Ts):
    #     mask = true_Ts == T
    #     ax.plot(true_counts[mask], total_ct[mask], marker=markers.next(), markersize=2, linestyle='none', label="pi=%.e" % T)
    im = ax.scatter(true_counts, total_ct, c=true_recombined_fractions, s=2)
    xs = np.linspace(0, ax.get_xlim()[1])
    ax.plot(xs, xs, '--', label='y=x', color='tab:orange')
    ax.set_aspect('equal')
    ax.legend(ncol=2, loc='upper left')
    ax.set_xlabel("True transfers")
    ax.set_ylabel("Detected transfers")
    ax.set_ylim([0, 40])
    ax.set_xlim([0, 40])
    ax.set_xticks([0, 20, 40])
    ax.set_yticks([0, 20, 40])
    # axins1 = inset_axes(ax, width='20%', height='5%', loc='lower right')
    # cmin, cmax = im.get_clim()
    # below = 0.25 * (cmax - cmin) + cmin
    # above = 0.75 * (cmax - cmin) + cmin
    # cb = fig.colorbar(im, cax=axins1, orientation='horizontal', ticks=[below, above])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # magical parameter to make it similar size to ax
    cb.set_label(r"$f_r$")
    # axins1.xaxis.set_ticks_position('top')
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
    ax.set_ylabel("Clonal divergence")
    ax.legend()
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    return

def plot_clonal_T_est_correlation(ax, true_Ts, est_Ts):
    # markers = itertools.cycle(('+', '.', 'x', '^'))
    ax.plot(true_Ts, est_Ts, '.', markersize=2)
    xs = np.linspace(0, ax.get_xlim()[1])
    ax.plot(xs, xs, '--', label='y=x')
    # ax.legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal')
    ax.set_xlabel("True clonal div " r"($\times 10^{-4}$)")
    ax.set_ylabel("Inferred clonal div " r"($\times 10^{-4}$)")
    ax.set_ylim([0, 2.2e-4])
    ax.set_xlim([0, 2.2e-4])
    ax.set_xticks([0, 1e-4, 2e-4])
    ax.set_xticklabels([0, 1, 2])
    ax.set_yticks([0, 1e-4, 2e-4])
    ax.set_yticklabels([0, 1, 2])
    ax.legend(loc='upper left')
    return


def plot_clonal_frac_correlation(ax, true_cfs, est_cfs):
    # markers = itertools.cycle(('+', '.', 'x', '^'))
    ax.plot(true_cfs, est_cfs, '.', markersize=2)
    xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
    ax.plot(xs, xs, '--', label='y=x')
    # ax.legend(ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal')
    ax.set_xlabel("True clonal fraction")
    ax.set_ylabel("Inferred clonal fraction")
    # ax.set_ylim([0, 2.2e-4])
    # ax.set_xlim([0, 2.2e-4])
    # ax.set_xticks([0, 1e-4, 2e-4])
    # ax.set_xticklabels([0, 1, 2])
    # ax.set_yticks([0, 1e-4, 2e-4])
    # ax.set_yticklabels([0, 1, 2])
    ax.legend(loc='upper left')
    return

def plot_length_distributions(ax, true_lens, detected_lens):
    _ = ax.hist(detected_lens * config.second_pass_block_size, histtype='step', bins=100, density=True, label='Detected dist')
    _ = ax.hist(true_lens, histtype='step', bins=100, density=True, label='True dist')
    ax.set_xlabel('Transfer length / bps')
    ax.set_ylabel(r'Density ($\times 10^{-4})$')
    ax.set_yticklabels([0, 1, 2, 3, 4])
    ax.legend()
    return


# set up figure
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

# fig = plt.figure(figsize=(4.5, 5.))
# outer_grid = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[1, 1, 1], hspace=0.6, figure=fig)
# upper_grid = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, width_ratios=[1.1, 1], wspace=0.6, subplot_spec=outer_grid[0])
# mid_grid = gridspec.GridSpecFromSubplotSpec(ncols=3, nrows=1, width_ratios=[0.0, 0.4, 0.3], wspace=0.1, subplot_spec=outer_grid[1])
# lower_grid = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, width_ratios=[1, 1], wspace=0.6, subplot_spec=outer_grid[2])
#
# count_ax = fig.add_subplot(mid_grid[1])
# true_len_ax = fig.add_subplot(upper_grid[1])
# T_est_ax = fig.add_subplot(upper_grid[0])
# between_within_len_ax = fig.add_subplot(lower_grid[1])
# between_within_count_ax = fig.add_subplot(lower_grid[0])
# fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(6.5, 2.), gridspec_kw={'width_ratios': [1,1.1,0.7], 'wspace':0.6})
fig = plt.figure(figsize=(4.5, 3.5))
# create a 1-row 3-column container as the left container
gs_top = gridspec.GridSpec(1, 2, width_ratios=[1, 1.1])
# create a 1-row 1-column grid as the right container
gs_bottom = gridspec.GridSpec(1, 2)
# add plots to the nested structure
T_est_ax = fig.add_subplot(gs_top[0,0])
count_ax = fig.add_subplot(gs_top[0,1])
true_len_ax = fig.add_subplot(gs_bottom[0,0])
cf_ax = fig.add_subplot(gs_bottom[0,1])

gs_top.update(bottom=0.55)
# gs_right.update(left=0.72, top=0.75, bottom=0.25)
# also, we want to get rid of the horizontal spacing in the left gridspec
# gs_left.update(wspace=0.4)
gs_bottom.update(top=0.45)

# fig2, axes2 = plt.subplots(ncols=2, nrows=1, figsize=(3, 2.5), gridspec_kw={'width_ratios': [1,1]})
# load up data
path = os.path.join(config.analysis_directory, 'HMM_validation', 'Bacteroides_vulgatus_57955.pickle')
genome_len = 2.8e5
# data = pickle.load(open(path, 'rb'), encoding='latin1')
data = pickle.load(open(path, 'rb'))
true_counts = np.array(data['true counts'])
true_between_counts = np.array(data['true between clade counts'])
true_within_counts = true_counts - true_between_counts
true_Ts = np.array(data['true T'])
true_divs = np.array(data['true div'])
true_lens = np.concatenate(data['true lengths'])
true_total_lens = np.array([np.sum(x) for x in data['true lengths']])
est_Ts = np.array(data['T est'])
true_cf = np.array(data['true clonal fraction'])
inferred_cf = np.array(data['clonal fraction'])
total_counts, within_counts, between_counts, full_df = preprocess_data(data)

plot_count_correlation(fig, count_ax, true_counts, true_Ts, total_counts, true_total_lens / genome_len)
# plot_clonal_T_estimation(T_est_ax, true_Ts, est_Ts)
plot_clonal_T_est_correlation(T_est_ax, true_divs, est_Ts)
plot_length_distributions(true_len_ax, true_lens, full_df['lengths'].astype(float))
plot_clonal_frac_correlation(cf_ax, true_cf, inferred_cf)

fig.savefig(os.path.join(config.figure_directory, "supp_HMM_validation.pdf"), bbox_inches="tight")
