import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
import sys
import time
import seaborn as sns
sys.path.append("..")
from utils import core_gene_utils, diversity_utils, HGT_utils, parallel_utils
import config


def plot_for_one_species(ax, species_name, num_to_plot, normalization=True, mode='QP'):
    color_list = sns.color_palette(palette='colorblind')
    between_color = color_list[0]
    within_color = color_list[4]
    # mode can either be 'QP' or 'within'
    # loading data
    base_dir = os.path.join(config.data_directory, 'zarr_snps', species_name)
    if not os.path.exists(base_dir):
        print('No data found for {}'.format(species_name))
        return

    dh = parallel_utils.DataHoarder(species_name, mode)
    # filtering the arrays
    good_chromo = dh.chromosomes[dh.general_mask] # will be used in contig-wise run computation

    # load same clade snp cutoff
    # TODO eventually want to use only divergence
    cutoffs = json.load(open('./same_clade_snp_cutoffs.json', 'r'))
    if species_name not in cutoffs:
        lower_cutoff = 5
        upper_cutoff = 5e6
    else:
        lower_cutoff = cutoffs[species_name][0] or 0  # hacky way of assigning value to None
        upper_cutoff = cutoffs[species_name][1] or 5e6

    print("Finish loading for {}".format(species_name))

    ax.set_yscale('log')
    ax.set_xlabel('Normalized site counts')
    ax.set_ylabel('Survival Probability')
    if normalization:
        ax.set_xlim((0, 25))

    # prepare the list of sample pairs/samples to plot
    if mode == 'within':
        num_to_plot = min(num_to_plot, dh.snp_arr.shape[1])
        idxs = random.sample(range(dh.snp_arr.shape[1]), num_to_plot)
        if num_to_plot > 100:
            alpha = 0.1
        else:
            alpha = 0.5
        color = within_color
    elif mode == 'QP':
        idxs = []
        for i in range(num_to_plot):
            idxs.append(random.sample(range(dh.snp_arr.shape[1]), 2))
        alpha = 0.1
        color = between_color
    else:
        raise ValueError("Mode has to be either QP or within")

    for idx in idxs:
        snp_vec, snp_mask = dh.get_snp_vector(idx)
        snp_count = np.sum(snp_vec)
        if (snp_count < lower_cutoff) or (snp_count > upper_cutoff):
            continue
        if normalization:
            div = snp_count / float(len(snp_vec))
        else:
            div = 1
        runs = parallel_utils.compute_runs_all_chromosomes(snp_vec, good_chromo[snp_mask])
        if len(runs) == 0:
            print("Divergence is %f, for a total of %d snps" % (div, snp_count))
            continue
        # normalize by multiplying div
        data = runs * div
        plot_range = (0, max(data))
        _ = ax.hist(data, range=plot_range, normed=True, cumulative=-1, bins=100,
                    histtype='step', color=color, alpha=alpha)
    ax.plot([], color=color, label=mode)
    return


def plot_null(ax):
    # the boring e(-x) null
    xs = np.arange(0, 10)
    ax.plot(xs, np.exp(-xs), color='r', label="Null")


def main():
    t0 = time.time()
    base_dir = 'zarr_snps'
    fig_base_path = os.path.join(
        config.analysis_directory, 'run_size_survival_distributions', 'empirical', 'full_sites', 'both')
    for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
        if species_name.startswith('.'):
            continue
        fig_path = os.path.join(fig_base_path, '{}.pdf'.format(species_name))
        if os.path.exists(fig_path):
            print('{} has already been processed'.format(species_name))
            continue
        print("plotting for {} at {} min".format(species_name, (time.time()-t0)/60))

        fig = plt.figure()
        fig.set_size_inches(8, 6)
        ax = fig.gca()

        plot_for_one_species(ax, species_name, 500, normalization=True, mode='QP')
        plot_for_one_species(ax, species_name, 500, normalization=True, mode='within')
        plot_null(ax)
        ax.legend()

        plt.savefig(fig_path)
        plt.close()


if __name__ == "__main__":
    main()
