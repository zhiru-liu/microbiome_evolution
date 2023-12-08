import sys
import os
import json
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
sys.path.append("..")
import config


def load_data_and_plot_mirror(between_host_path, within_host_path, ax, ind_to_plot=0, ylim=0.35, colors=['tab:blue', 'tab:orange']):
    between_cumu_runs = np.loadtxt(between_host_path)
    within_cumu_runs = np.loadtxt(within_host_path)
    plot_mirror_single_threshold(between_cumu_runs, within_cumu_runs, ax, ind_to_plot=ind_to_plot, ylim=ylim, colors=colors)
    return between_cumu_runs, within_cumu_runs


def plot_mirror(between_cumu_runs, within_cumu_runs, ax, threshold_lens, ind_to_plot=None, ylim=0.35, normalized=False):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = 0
    # decide which of the cumu_runs to plot
    if ind_to_plot is None:
        to_plot = range(between_cumu_runs.shape[1])
    else:
        to_plot = ind_to_plot

    for i in to_plot:
        if normalized:
            scale_between = between_cumu_runs[:, i].mean()
            scale_within = within_cumu_runs[:, i].mean()
            print("normalize by mean homozygosity (top&bottom) for threshold %d : %f & %f \n" % (threshold_lens[0][i], scale_between, scale_within))
            ax.plot(between_cumu_runs[:, i] / scale_between, linewidth=1, color=colors[color_idx],
                    label="%d / %d" % (threshold_lens[0][i], threshold_lens[1][i]))
            ax.plot(-within_cumu_runs[:, i] / scale_within, linewidth=1, color=colors[color_idx])
        else:
            ax.plot(between_cumu_runs[:, i], linewidth=1, color=colors[color_idx],
                    label="%d / %d" % (threshold_lens[0][i], threshold_lens[1][i]))
            ax.plot(-within_cumu_runs[:, i], linewidth=1, color=colors[color_idx])
        color_idx += 1
    ax.hlines(0, 0, between_cumu_runs.shape[0], 'black', linewidth=1)
    ax.set_xlim([0, between_cumu_runs.shape[0]])
    ax.set_ylim([-ylim, ylim])
    ax.set_xlabel("4D core genome location")
    ax.set_ylabel("Sharing probability")
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_yticklabels(np.around(map(np.abs, ax.get_yticks()), decimals=1))


def plot_mirror_single_threshold(between_cumu_runs, within_cumu_runs, ax, ind_to_plot, ylim=0.35, colors=['tab:blue', 'tab:orange']):
    # decide which of the cumu_runs to plot
    ax.plot(between_cumu_runs[:, ind_to_plot], linewidth=1, color=colors[0])
    ax.plot(-within_cumu_runs[:, ind_to_plot], linewidth=1, color=colors[1])

    xs = np.arange(between_cumu_runs.shape[0])
    zeros = np.zeros(between_cumu_runs.shape[0])
    ax.fill_between(xs, zeros, between_cumu_runs[:, ind_to_plot], color=colors[0], alpha=0.5, rasterized=True)
    ax.fill_between(xs, -within_cumu_runs[:, ind_to_plot], zeros, color=colors[1], alpha=0.5, rasterized=True)

    ax.hlines(0, 0, between_cumu_runs.shape[0], 'black', linewidth=1)
    ax.set_xlim([0, between_cumu_runs.shape[0]])
    ax.set_ylim([-ylim, ylim])
    ax.set_xlabel("Location along core genome")
    ax.set_ylabel("Sharing probability")
    # ax.legend(bbox_to_anchor=(1, 1))
    ax.set_yticklabels(np.around(map(np.abs, ax.get_yticks()), decimals=1))


def plot_delta(between_host_path, within_host_path, ax, threshold_lens, ind_to_plot=None, annotate_regions=[]):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    between_cumu_runs = np.loadtxt(between_host_path)
    within_cumu_runs = np.loadtxt(within_host_path)
    color_idx = 0
    # decide which of the cumu_runs to plot
    if ind_to_plot is None:
        to_plot = range(between_cumu_runs.shape[1])
    else:
        to_plot = ind_to_plot

    for i in to_plot:
        ax.plot(within_cumu_runs[:, i] - between_cumu_runs[:, i], linewidth=1, color=colors[color_idx],
                label="{}".format(int(threshold_lens[i])))
        color_idx += 1
    ax.hlines(0, 0, between_cumu_runs.shape[0], 'black', linewidth=1)
    for x, y in annotate_regions:
        ax.axvspan(x, y, alpha=0.5, color='red')
    ax.set_xlim([0, between_cumu_runs.shape[0]])
    ax.set_xlabel("4D core genome location")
    ax.set_ylabel("sharing probability")
    ax.legend(bbox_to_anchor=(1, 1))


if __name__ == '__main__':
    base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955_between')
    # between_path = os.path.join(base_path, 'cutoff_0.001.csv')
    # thresholds = np.loadtxt(os.path.join(base_path, 'thresholds.txt'))
    within_path = os.path.join(base_path, 'within_host.csv')
    between_path = os.path.join(base_path, 'between_host.csv')
    thresholds = np.loadtxt(os.path.join(base_path, 'between_host_thresholds.txt'))

    fig, ax = plt.subplots(figsize=(7, 2.5))
    # plot_mirror(between_path, within_path, ax, threshold_lens=thresholds, ind_to_plot=[3, 5, 7], ylim=0.35)
    plot_mirror(between_path, within_path, ax, threshold_lens=thresholds, ind_to_plot=[0, 2, 3], ylim=0.35)
    fig.savefig('test_mirror_between_clade.pdf', bbox_inches='tight')

    # plot_delta(between_path, within_path, ax, threshold_lens=thresholds, ind_to_plot=[3, 5, 7],
    #            annotate_regions=[[117500, 121800]])
    # fig.savefig('test_within_pileup_delta.pdf', bbox_inches='tight')
