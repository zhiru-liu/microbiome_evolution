import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
import sys
import time
import dask
import dask.array as da
from scipy import stats
sys.path.append("..")
from utils import core_gene_utils, diversity_utils, HGT_utils, parallel_utils
import config


def plot_for_one_species(ax, species_name, num_to_plot, normalization=True):
    # loading data
    base_dir = os.path.join(config.data_directory, 'zarr_snps', species_name)
    if not os.path.exists(base_dir):
        print('No data found for {}'.format(species_name))
        return

    dh = parallel_utils.DataHoarder(species_name)
    # filtering the arrays
    good_chromo = dh.chromosomes[dh.general_mask] # will be used in contig-wise run computation

    # load same clade snp cutoff
    # TODO eventually want to use only divergence
    cutoffs = json.load(open('./same_clade_snp_cutoffs.json', 'r'))
    lower_cutoff = cutoffs[species_name][0] or 0  # hacky way of assigning value to None
    upper_cutoff = cutoffs[species_name][1] or 5e6

    print("Finish loading for {}".format(species_name))

    ax = plt.gca()
    ax.set_yscale('log')

    ax.set_xlabel('Normalized site counts')
    ax.set_ylabel('Survival Probability')
    if normalization:
        ax.set_xlim((0, 25))

    i = 0
    while i < num_to_plot:
        pair = random.sample(range(dh.snp_arr.shape[1]), 2)
        snp_vec, snp_mask = parallel_utils.get_two_QP_sample_snp_vector(
                dh.snp_arr, dh.covered_arr, pair)
        snp_count = np.sum(snp_vec)
        if (snp_count < lower_cutoff) or (snp_count > upper_cutoff):
            continue
        if normalization:
            div = snp_count / float(len(snp_vec))
        else:
            div = 1
        runs = parallel_utils.compute_runs_all_chromosomes(snp_vec, good_chromo[snp_mask])
        # normalize by multiplying div
        data = runs * div
        plot_range = (0, max(data))
        _ = ax.hist(data, range=plot_range, normed=True, cumulative=-1, bins=100,
                    histtype='step', color='b', alpha=0.1)
        i += 1
    return


def plot_null(ax):
    # the boring e(-x) null
    xs = np.arange(0, 10)
    ax.plot(xs, np.exp(-xs), color='r')


def main():
    t0 = time.time()
    base_dir = 'zarr_snps'
    fig_base_path = os.path.join(
        config.analysis_directory, 'run_size_survival_distributions', 'no_normalization')
    for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
        if species_name.startswith('.'):
            continue

        print("plotting for {} at {} min".format(species_name, (time.time()-t0)/60))

        fig = plt.figure()
        fig.set_size_inches(8, 6)
        ax = fig.gca()

        plot_for_one_species(ax, species_name, 500, normalization=False)
        #plot_null(ax)

        fig_path = os.path.join(fig_base_path, '{}.pdf'.format(species_name))
        plt.savefig(fig_path)
        plt.close()


if __name__ == "__main__":
    main()
