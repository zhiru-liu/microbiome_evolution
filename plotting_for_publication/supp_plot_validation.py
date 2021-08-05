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
        ax.plot(true_counts[mask], total_ct[mask], marker=markers.next(), linestyle='none', label="pi=%.e" % T)
    xs = np.linspace(0, ax.get_xlim()[1])
    ax.legend(bbox_to_anchor=(1, 1))
    ax.plot(xs, xs, '--')
    ax.set_xlabel("True")
    ax.set_ylabel("Detected")
    ax.set_title("Number of transfers")
    return


def plot_clonal_T_estimation(ax, true_Ts, est_Ts):
    plotting_dat = []
    Ts = np.unique(true_Ts)
    for T in Ts:
        plotting_dat.append(est_Ts[T == true_Ts])

    _ = ax.boxplot(plotting_dat, meanprops={'color': 'k'})
    ax.set_xticklabels(Ts)
    ax.set_xlabel("True T")
    ax.set_ylabel("pi in clonal region")
    ax.set_title("Wall clock estimation")
    return


def plot_length_distributions(ax, true_lens, detected_lens):
    _ = ax.hist(detected_lens * config.second_pass_block_size, histtype='step', bins=100, density=True)
    _ = ax.hist(true_lens, histtype='step', bins=100, density=True)
    return


def plot_between_within_clade(ax, est_Ts, within_counts, between_counts):
    _ = ax.scatter(est_Ts, within_counts, s=2, label='Within clade transfers')
    _ = ax.scatter(est_Ts, -np.array(between_counts), s=2, label='Between clade transfers')
    ax.plot(est_Ts, np.zeros(est_Ts.shape), 'k-')
    ax.set_xlim([0, 3e-4])
    ax.set_ylim([-40, 40])
    return


path = os.path.join(config.analysis_directory, 'HMM_validation', 'Bacteroides_vulgatus_57955.pickle')
data = pickle.load(open(path, 'rb'))
true_counts = np.array(data['true counts'])
true_Ts = np.array(data['true T'])
true_lens = np.concatenate(data['true lengths'])
est_Ts = np.array(data['T est'])
total_counts, within_counts, between_counts, full_df = preprocess_data(data)

fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
plot_count_correlation(axes[0, 0], true_counts, true_Ts, total_counts)
plot_clonal_T_estimation(axes[0, 1], true_Ts, est_Ts)
plot_length_distributions(axes[1, 0], true_lens, full_df['lengths'].astype(float))
plot_between_within_clade(axes[1, 1], est_Ts, within_counts, between_counts)

fig.savefig("test_validation.pdf", bbox_inches="tight")
