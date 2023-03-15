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
from utils import close_pair_utils, figure_utils


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
    ax.plot(xs, xs, '--', label=r'$y=x$', color='tab:orange')
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
    ax.plot(xs, ys, '--', label=r'$y=x$')
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
    ax.plot(xs, xs, '--', label=r'$y=x$')
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
    ax.plot(xs, xs, '--', label=r'$y=x$')
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

def plot_length_distributions(ax, true_lens, detected_lens, density=True):
    print("Num detected: {}; num true: {}".format(len(detected_lens), len(true_lens)))
    res = ax.hist(detected_lens * config.second_pass_block_size, histtype='step', bins=50, density=density, label='Detected dist')
    _ = ax.hist(true_lens, histtype='step', bins=res[1], density=density, label='True dist')
    ax.set_xlabel('Transfer length / bps')
    if density:
        ax.set_ylabel(r'Density ($\times 10^{-4})$')
        ax.set_yticklabels([0, 1, 2, 3, 4])
    else:
        ax.set_ylabel('Counts')
    ax.legend()
    return

def plot_TcTm_dist(ax, est_T, true_div, est_clonal_frac, true_clonal_frac):
    Tc = 0.0094  # can be found using typical_pair_utils._compute_theta
    est_recomb_frac = 1 - est_clonal_frac
    true_recomb_frac = 1 - true_clonal_frac

    figure_utils.plot_jitters(ax, 2, est_recomb_frac / est_T * Tc, 0.1, alpha=0.35)
    figure_utils.plot_jitters(ax, 1, true_recomb_frac / true_div * Tc, 0.1, alpha=0.35)
    ax.axhline(Tc * 2600 * 0.65, linestyle='--', alpha=0.35, color='k', label=r'$r/m$')  # simulation parameter
    ax.set_ylabel(r'$T_{mrca}/T_{mosaic}$')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['True', 'Inferred'])
    ax.legend()
    return

def plot_TcTm_corr(ax, est_T, true_div, est_clonal_frac, true_clonal_frac):
    ax.set_aspect('equal')
    Tc = 0.0094  # can be found using typical_pair_utils._compute_theta
    est_recomb_frac = 1 - est_clonal_frac
    true_recomb_frac = 1 - true_clonal_frac
    ax.plot(true_recomb_frac / true_div * Tc, est_recomb_frac / est_T * Tc, '.', markersize=2)
    xs = np.linspace(0, 75)
    ax.plot(xs, xs, label=r'$y=x$', linestyle='--')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.set_xlabel(r'True pairwise $T_{mrca}/T_{mosaic}$')
    ax.set_ylabel(r'Inferred pairwise $T_{mrca}/T_{mosaic}$')
    ax.legend()


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
fig = plt.figure(figsize=(6, 3.7))
# create a 1-row 3-column container as the left container
gs_tl = gridspec.GridSpec(1, 1)
gs_tm = gridspec.GridSpec(1, 1)
gs_tr = gridspec.GridSpec(1, 1)
gs_bl = gridspec.GridSpec(1, 1)
gs_bm = gridspec.GridSpec(1, 1)
gs_br = gridspec.GridSpec(1, 1)

# add plots to the nested structure
T_est_ax = fig.add_subplot(gs_tl[0,0])
count_ax = fig.add_subplot(gs_tr[0,0])
true_len_ax = fig.add_subplot(gs_br[0,0])
cf_ax = fig.add_subplot(gs_tm[0,0])
TcTm_corr_ax = fig.add_subplot(gs_bl[0,0])
TcTm_dist_ax = fig.add_subplot(gs_bm[0,0])

# gs_tl.update(left=0.1, right=0.4, top=0.95, bottom=0.60)
# gs_tr.update(left=0.4, right=0.85, top=0.95, bottom=0.60)
# gs_bl.update(left=0.1, right=0.4, top=0.45, bottom=0.10)
# gs_br.update(left=0.55, right=0.9, top=0.45, bottom=0.10)
gs_tl.update(left=0.1, right=0.3, top=0.95, bottom=0.60)
gs_tm.update(left=0.39, right=0.59, top=0.95, bottom=0.60)
gs_tr.update(left=0.63, right=0.91, top=0.95, bottom=0.60)
gs_bl.update(left=0.1, right=0.3, top=0.45, bottom=0.10)
gs_bm.update(left=0.42, right=0.54, top=0.45, bottom=0.10)
gs_br.update(left=0.65, right=0.95, top=0.45, bottom=0.10)

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
plot_length_distributions(true_len_ax, true_lens, full_df['lengths'].astype(float), density=False)
plot_clonal_frac_correlation(cf_ax, true_cf, inferred_cf)

plot_TcTm_dist(TcTm_dist_ax, est_Ts, true_divs, inferred_cf, true_cf)
plot_TcTm_corr(TcTm_corr_ax, est_Ts, true_divs, inferred_cf, true_cf)


T_est_ax.text(-0.08, 1.12, "A", transform=T_est_ax.transAxes,
        fontsize=7, fontweight='bold', va='top', ha='left')
cf_ax.text(-0.08, 1.12, "B", transform=cf_ax.transAxes,
        fontsize=7, fontweight='bold', va='top', ha='left')
count_ax.text(-0.08, 1.12, "C", transform=count_ax.transAxes,
              fontsize=7, fontweight='bold', va='top', ha='left')
true_len_ax.text(-0.08, 1.12, "F", transform=true_len_ax.transAxes,
        fontsize=7, fontweight='bold', va='top', ha='left')
TcTm_corr_ax.text(-0.08, 1.12, "D", transform=TcTm_corr_ax.transAxes,
                 fontsize=7, fontweight='bold', va='top', ha='left')
TcTm_dist_ax.text(-0.08, 1.12, "E", transform=TcTm_dist_ax.transAxes,
                 fontsize=7, fontweight='bold', va='top', ha='left')

fig.savefig(os.path.join(config.figure_directory, "supp", "supp_HMM_validation.pdf"), bbox_inches='tight')
