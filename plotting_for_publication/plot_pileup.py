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
    lambdas = [5000, 10000]
    num_reps = 4  # number of replicates
    markers = itertools.cycle(('^', 'x'))

    for i in range(len(rbymus)):
        rbymu = rbymus[i]
        c = plt.get_cmap("tab10")(i)
        for j in range(len(lambdas)):
            l = lambdas[j]
            idx = i * len(lambdas) + j
            mean_cv = np.mean(sim_cvs[idx:idx + num_reps, :], axis=0)
            ax.plot(sim_thresholds, mean_cv, marker=next(markers), color=c)
        ax.plot(-5000, 0.5, '-', color=c, label='r/mu=%.1f' % rbymu)

    ax.plot(-5000, 0.5, marker='^', color='grey', label='l=5000')
    ax.plot(-5000, 0.5, marker='x', color='grey', label='l=10000')
    ax.plot(real_thresholds, real_cvs, 'r.--', label="B. vulgatus")

    ax.set_xlim([1000, 5000])
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xlabel("Sharing threshold (4D syn sites)")
    ax.set_ylabel("coefficient of variation")


if __name__ == "__main__":
    # set up figure
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['legend.frameon']  = False
    mpl.rcParams['legend.fontsize']  = 'small'

    # setting up grids
    fig = plt.figure(figsize=(7, 5))
    outer_grid = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3.5, 2], hspace=0.4, figure=fig)
    pileup_grid = gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=2, height_ratios=[1, 1], hspace=0.4, subplot_spec=outer_grid[0])
    top_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], wspace=0.1, subplot_spec=pileup_grid[0])
    mid_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], wspace=0.1, subplot_spec=pileup_grid[1])
    low_grid = gridspec.GridSpecFromSubplotSpec(1, 3, width_ratios=[1, 1, 0.5], wspace=0.7, subplot_spec=outer_grid[1])
    spectrum_grid = gridspec.GridSpecFromSubplotSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.4, subplot_spec=low_grid[0])

    # setting up axes
    pileup_ax_sim = fig.add_subplot(mid_grid[0])
    pileup_ax_real = fig.add_subplot(top_grid[0])
    histo_ax_sim = fig.add_subplot(top_grid[1])
    histo_ax_real = fig.add_subplot(mid_grid[1])
    spectrum_axes = [fig.add_subplot(x) for x in spectrum_grid]
    cv_ax = fig.add_subplot(low_grid[1])

    # loading necessary data
    real_data_save_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')
    thresholds = np.loadtxt(os.path.join(real_data_save_path, 'between_host_thresholds.txt'))

    spectra_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'local_hap', 'local_haplotype_spectra.json')
    all_hap_spectra = json.load(open(spectra_path, 'r'))

    cvs_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'b_vulgatus', 'cv.csv')
    sim_cvs = np.loadtxt(cvs_path)
    threshold_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'b_vulgatus', 'thresholds.txt')
    sim_thresholds = np.loadtxt(os.path.join(threshold_path))

    # plot one species
    ind_to_plot = [1, 4]  # only showing three thresholds
    real_cumu_runs = plot_single_side(
        os.path.join(real_data_save_path, 'between_host.csv'),
        thresholds, pileup_ax_real, ind_to_plot=ind_to_plot, xlabel=False)

    # plot one simulation
    ind_to_plot = [0, 3]  # only showing three thresholds
    save_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'b_vulgatus')
    plot_single_side(os.path.join(save_path, '4.txt'), sim_thresholds, pileup_ax_sim,
                     ind_to_plot=ind_to_plot)

    # plot haplotype spectra
    plot_haplotype_spectrum(spectrum_axes, all_hap_spectra, ['a', 'b', 'c', 'neutral'])

    # plot CV comparison
    real_cv = np.std(real_cumu_runs, axis=0) / np.mean(real_cumu_runs, axis=0)
    plot_cvs_comparison(cv_ax, thresholds, sim_thresholds, real_cv, sim_cvs)

    # annotating regions
    regions = [[61000, 63000], [143000, 145000], [223000, 225000]]
    for x,y in regions:
        pileup_ax_real.axvspan(x, y, alpha=0.3, color='red')
        pileup_ax_sim.axvspan(x, y, alpha=0.3, color='red')

    # final adjustment of axes limits, labels, ...
    pileup_ax_real.set_title("Empirical")
    pileup_ax_sim.set_title("Simulated")
    pileup_ax_sim.set_xlim(pileup_ax_real.get_xlim())

    histo_ax_real.set_yticklabels([])
    histo_ax_sim.set_yticklabels([])

    # finally saving figure
    fig.savefig('test_pileup.pdf', bbox_inches='tight')
