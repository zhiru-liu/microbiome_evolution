import sys
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.append("..")
import config
from utils import parallel_utils


def plot_example_run_length_dist(dist_ax, snp_vec_axes, within_dh, between_dh, within_idx, between_pair1, between_pair2):
    good_chromo = within_dh.chromosomes[within_dh.general_mask]
    within_snp_vec, snp_mask = within_dh.get_snp_vector(within_idx)
    within_example_runs = parallel_utils.compute_runs_all_chromosomes(within_snp_vec, good_chromo[snp_mask])
    within_div = (np.sum(within_snp_vec) / float(len(within_snp_vec)))

    between_snp_vec1, snp_mask = between_dh.get_snp_vector(between_pair1)
    between_example_runs1 = parallel_utils.compute_runs_all_chromosomes(between_snp_vec1, good_chromo[snp_mask])

    between_snp_vec2, snp_mask = between_dh.get_snp_vector(between_pair2)
    between_example_runs2 = parallel_utils.compute_runs_all_chromosomes(between_snp_vec2, good_chromo[snp_mask])

    between_div1 = (np.sum(between_snp_vec1) / float(len(between_snp_vec1)))
    between_div2 = (np.sum(between_snp_vec2) / float(len(between_snp_vec2)))

    # plot the histograms
    _ = dist_ax.hist(within_example_runs * within_div, density=True, cumulative=-1, bins=1000,
                histtype='step', label='within host')
    _ = dist_ax.hist(between_example_runs1 * between_div1, density=True, cumulative=-1, bins=1000,
                histtype='step', label='between host 1')
    _ = dist_ax.hist(between_example_runs2 * between_div2, density=True, cumulative=-1, bins=1000,
                histtype='step', label='between host 2')
    xs = np.linspace(0, 10)
    dist_ax.plot(xs, np.exp(-xs), label='Null')
    dist_ax.set_yscale('log')
    dist_ax.legend()
    dist_ax.set_xlim([0, dist_ax.get_xlim()[1]])
    ymin = 0.8 / max(len(within_example_runs),
                     len(between_example_runs1), len(between_example_runs2))
    dist_ax.set_ylim([ymin, dist_ax.get_ylim()[1]])

    # plot the snp_vectors
    snp_vec_axes[0].plot(within_snp_vec, linewidth=0.5)
    snp_vec_axes[1].plot(between_snp_vec1, linewidth=0.5)
    snp_vec_axes[2].plot(between_snp_vec2, linewidth=0.5)
    snp_vec_axes[0].set_xticklabels([])
    snp_vec_axes[1].set_xticklabels([])

    xmax = min(len(within_snp_vec), len(between_snp_vec1), len(between_snp_vec2))
    for ax in snp_vec_axes:
        ax.set_xlim([0, xmax])
        ax.set_yticklabels([])
        ax.set_yticks([])


# setting up axes
fig1, ax = plt.subplots(figsize=(4, 3))
fig2, axes = plt.subplots(3, 1, figsize=(4, 2))

# loading neccessary data
species_name = 'Bacteroides_vulgatus_57955'
dh = parallel_utils.DataHoarder(species_name, mode='within')
dh2 = parallel_utils.DataHoarder(species_name, mode='QP')

# plot
plot_example_run_length_dist(ax, axes, dh, dh2,
                             within_idx=92, between_pair1=(152, 224), between_pair2=(42, 69))

fig1.savefig("example_run_length.pdf", bbox_inches='tight')
fig2.savefig("example_genomes.pdf", bbox_inches='tight')
