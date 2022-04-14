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
from utils import pileup_utils


def plot_single_side(ckpt_path, threshold_lens, pileup_ax,
                     ind_to_plot=None, legend=True, xlabel=True):
    # also returns bins in histogram plot + cumu_runs
    cumu_runs = np.loadtxt(ckpt_path)

    # decide which of the cumu_runs to plot
    if ind_to_plot is None:
        to_plot = range(cumu_runs.shape[1])
    else:
        to_plot = ind_to_plot

    for i in to_plot:
        dat = cumu_runs[:, i]
        pileup_ax.plot(dat, linewidth=1, label="{}".format(int(threshold_lens[i])))
    pileup_ax.set_ylim([0, 0.35])
    pileup_ax.set_xlim([0, cumu_runs.shape[0]])
    if xlabel:
        pileup_ax.set_xlabel('4D core genome location')
    else:
        pileup_ax.set_xticklabels([])
    if legend:
        pileup_ax.legend()
    pileup_ax.set_ylabel('sharing fraction')
    return cumu_runs


def plot_haplotype_spectrum(axes, all_spectra, spectra_titles):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, dat in enumerate(all_spectra):
        cum_size = sum(dat)
        color_idx = 0
        for size in dat[::-1]:
            cum_size -= size
            if size > 1:
                color = colors[color_idx % len(colors)]
                color_idx += 1
                edgecolor = None
            else:
                color = 'grey'
                edgecolor = "white"
            axes[i].bar(0, size, bottom=cum_size, color=color, edgecolor=edgecolor, linewidth=0.1)
        axes[i].set_xlim([-0.4, 0.4])
        axes[i].set_ylim([0, sum(dat)])
        axes[i].axis('off')
        axes[i].set_title(spectra_titles[i])


def plot_cvs_comparison(ax, real_thresholds, sim_thresholds, real_cvs, sim_cvs):
    # currently sim params are hard coded
    rbymus = [0.1, 0.5, 1, 1.5, 2]
    lambdas = [500, 1000, 2000, 5000, 10000]
    num_reps = 4  # number of replicates
    markers = itertools.cycle(('^', 'x', '+', 'o', 'v'))

    for i in range(len(rbymus)):
        rbymu = rbymus[i]
        c = plt.get_cmap("tab10")(i)
        for j in range(len(lambdas)):
            l = lambdas[j]
            idx = i * len(lambdas) + j
            mean_cv = np.mean(sim_cvs[idx:idx + num_reps, :], axis=0)
            ax.plot(sim_thresholds+np.random.normal(scale=50), mean_cv, marker=next(markers), markersize=1, linewidth=0.1, color=c)
        ax.plot(-5000, 0.5, '-', color=c, label=r'$r/\mu$=%.1f' % rbymu)

    ax.plot(-5000, 0.5, marker='^', markersize=1, color='grey', label=r'$l_r=500$')
    ax.plot(-5000, 0.5, marker='x', markersize=1, color='grey', label=r'$l_r=1000$')
    ax.plot(-5000, 0.5, marker='+', markersize=1, color='grey', label=r'$l_r=2000$')
    ax.plot(-5000, 0.5, marker='o', markersize=1, color='grey', label=r'$l_r=5000$')
    ax.plot(-5000, 0.5, marker='v', markersize=1, color='grey', label=r'$l_r=10000$')
    ax.plot(real_thresholds[:-1], real_cvs[:-1], 'r.--', label="B. vulgatus")

    ax.set_xlim([1000, 5000])
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))
    ax.set_xlabel("Sharing threshold (4D syn sites)")
    ax.set_ylabel("coefficient of variation")


if __name__ == "__main__":
    # set up figure
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['legend.frameon']  = False
    mpl.rcParams['legend.fontsize']  = 'small'


