import sys
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.append("..")
import config
from utils import snp_data_utils

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

def plot_example_run_length_dist(dist_ax, snp_vec_axes, between_pair1, between_pair2):
    cache_file_wt1 = os.path.join(config.plotting_intermediate_directory, "fig5_within_snp1.csv")
    cache_file_wt2 = os.path.join(config.plotting_intermediate_directory, "fig5_within_snp2.csv")
    cache_file_bt1 = os.path.join(config.plotting_intermediate_directory, "fig5_between_snp1.csv")
    cache_file_bt2 = os.path.join(config.plotting_intermediate_directory, "fig5_between_snp2.csv")

    if os.path.exists(cache_file_bt1):
        within_snp_vec1 = np.loadtxt(cache_file_wt1)
        within_snp_vec2 = np.loadtxt(cache_file_wt2)
        between_snp_vec1 = np.loadtxt(cache_file_bt1)
        between_snp_vec2 = np.loadtxt(cache_file_bt2)
    else:
        # loading neccessary data
        species_name = 'Bacteroides_vulgatus_57955'
        within_dh = snp_data_utils.DataHoarder(species_name, mode='within')
        between_dh = snp_data_utils.DataHoarder(species_name, mode='QP')

        within_idx1 = np.where(within_dh.good_samples=='700114218')[0]
        within_idx2 = np.where(within_dh.good_samples=='700171115')[0]
        within_snp_vec1, snp_mask1 = within_dh.get_snp_vector(within_idx1)
        within_snp_vec2, snp_mask2 = within_dh.get_snp_vector(within_idx2)
        both_true = snp_mask1 & snp_mask2
        new_within_vec1 = within_snp_vec1[both_true[snp_mask1]]
        new_within_vec2 = within_snp_vec2[both_true[snp_mask2]]

        between_snp_vec2, snp_mask = between_dh.get_snp_vector(between_pair2)
        between_snp_vec1, snp_mask = between_dh.get_snp_vector(between_pair1)

        np.savetxt(cache_file_wt1, new_within_vec1)
        np.savetxt(cache_file_wt2, new_within_vec2)
        np.savetxt(cache_file_bt1, between_snp_vec1)
        np.savetxt(cache_file_bt2, between_snp_vec2)

    within_example_runs = snp_data_utils._compute_runs_single_chromosome(within_snp_vec2)
    within_example_runs0 = snp_data_utils._compute_runs_single_chromosome(within_snp_vec1)
    between_example_runs1 = snp_data_utils._compute_runs_single_chromosome(between_snp_vec1)
    between_example_runs2 = snp_data_utils._compute_runs_single_chromosome(between_snp_vec2)

    within_div = (np.sum(within_snp_vec2) / float(len(within_snp_vec2)))
    between_div1 = (np.sum(between_snp_vec1) / float(len(between_snp_vec1)))
    between_div2 = (np.sum(between_snp_vec2) / float(len(between_snp_vec2)))

    print("Within div %f" % within_div)
    sort_idx = np.argsort(within_example_runs)
    de_novo_len = within_example_runs[sort_idx[-3]]
    print("Within de novo event length %d" % de_novo_len)
    print("Independent null p-val {0:e}".format(np.exp(-1. * de_novo_len * within_div)))

    # plot the histograms
    _ = dist_ax.hist(within_example_runs * within_div, density=True, cumulative=-1, bins=1000,
                histtype='step', label='within host', color=config.within_host_color)
    _ = dist_ax.hist(between_example_runs1 * between_div1, density=True, cumulative=-1, bins=1000,
                histtype='step', label='between host 1', linestyle='dotted', color=config.between_host_color)
    _ = dist_ax.hist(between_example_runs2 * between_div2, density=True, cumulative=-1, bins=1000,
                histtype='step', label='between host 2', color=config.between_host_color)
    dist_ax.axvline(within_example_runs[np.argsort(within_example_runs)[-3]] * within_div, label='within-host sweep event', linestyle='--', color='k')
    xs = np.linspace(0, 10)
    dist_ax.plot(xs, np.exp(-xs), label='Random mutations', color='tab:red')
    dist_ax.set_yscale('log')
    dist_ax.legend()
    dist_ax.set_xlim([0, dist_ax.get_xlim()[1]])
    ymin = 0.8 / max(len(within_example_runs),
                     len(between_example_runs1), len(between_example_runs2))
    dist_ax.set_ylim([ymin, dist_ax.get_ylim()[1]])
    dist_ax.set_xlabel("Normalized run length ($l\cdot d$)")
    dist_ax.set_ylabel("Prob longer than $l$")

    # plot the snp_vectors
    snp_vec_axes[0].plot(within_snp_vec2, linewidth=0.5)
    snp_vec_axes[1].plot(between_snp_vec1, linewidth=0.5)
    snp_vec_axes[2].plot(between_snp_vec2, linewidth=0.5)
    snp_vec_axes[0].set_xticklabels([])
    snp_vec_axes[1].set_xticklabels([])

    xmax = min(len(within_snp_vec2), len(between_snp_vec1), len(between_snp_vec2))
    for ax in snp_vec_axes:
        ax.set_xlim([0, xmax])
        ax.set_yticklabels([])
        ax.set_yticks([])


# setting up axes
fig1, ax = plt.subplots(figsize=(4, 3))
fig2, axes = plt.subplots(3, 1, figsize=(4, 2))

# plot
plot_example_run_length_dist(ax, axes,
                             between_pair1=(152, 224), between_pair2=(42, 69))

fig1.savefig(os.path.join(config.figure_directory, 'supp', "S26_supp_example_run_length_distributions.pdf"), bbox_inches='tight')
# fig2.savefig("example_genomes.pdf", bbox_inches='tight')
