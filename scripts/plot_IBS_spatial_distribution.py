import sys
import pickle
import os
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import core_gene_utils, HGT_utils
import config


def plot_one_species(species_name):
    print("Loading {}".format(species_name))
    all_runs_dict = pickle.load(open(os.path.join(config.analysis_directory, 'between_hosts_checkpoints', species_name, 'all_runs_map.pickle'), 'rb'))
    snp_counts_map = pickle.load(open(os.path.join(config.analysis_directory, 'between_hosts_checkpoints', species_name, 'snp_counts_map.pickle'), 'rb'))
    # Use cutoffs of SNP count to ignore cousins
    cutoff_dict = json.load(open('./same_clade_snp_cutoffs.json', 'r'))
    cutoffs = cutoff_dict[species_name]
    counts = snp_counts_map.values()
    lower = cutoffs[0] or min(counts)  # hacky way of assigning value to None
    upper = cutoffs[1] or max(counts)
    all_genes = core_gene_utils.get_sorted_core_genes(species_name)

    threshold_list = [0, 5, 10, 15]
    final_cumu_runs = HGT_utils.cumulate_runs_by_thresholds(
            all_runs_dict, snp_counts_map, (lower, upper), len(all_genes), threshold_list)

    fig = plt.figure()
    ax = plt.gca()
    for i in range(len(threshold_list)):
        ax.plot(final_cumu_runs[i], label="Threshold %d" % threshold_list[i])
    ax.legend()
    ax.figure.set_size_inches(12, 4)
    ax.set_ylim((0, len(all_runs_dict)))
    ax.set_title('Cumulative long runs per gene')
    ax.set_xlabel('Gene id')
    ax.set_ylabel('Num runs')
    fig.savefig(os.path.join(config.analysis_directory, 'IBS_locations', '{}.pdf'.format(species_name)), dpi=600)
    plt.close()

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim([0, 1])
    ax.set_xlabel('Ratio between thresholds')
    ax.set_ylabel('Survival proba')
    for i in range(1, 4):
        _ = ax.hist(np.nan_to_num(final_cumu_runs[i] / final_cumu_runs[0]), cumulative=-1,
                    bins=100, density=True, label="Threshold %d" % threshold_list[i])
    ax.legend()
    fig.savefig(os.path.join(config.analysis_directory, 'IBS_locations', 'empirical', 'ratios',
                '{}.png'.format(species_name)), dpi=600)
    plt.close()


def plot_all():
    base_dir = 'between_hosts_checkpoints'
    for species_name in os.listdir(os.path.join(config.analysis_directory, base_dir)):
        if species_name.startswith('.'):
            continue
        plot_one_species(species_name)

plot_all()
