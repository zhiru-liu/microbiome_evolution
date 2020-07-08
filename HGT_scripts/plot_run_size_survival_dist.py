import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle
import json
import sys
import time
sys.path.append("..")
from utils import core_gene_utils, diversity_utils
import config

'''
    Need to know the cutoffs for same clade snp counts
    Currently, manually saved to a json file
'''


def get_core_gene_vector(species_name):
    core_genes = core_gene_utils.parse_core_genes(species_name)
    # sort the genes
    all_genes = np.array(list(core_genes))
    gene_indices = np.array(
        list(map(lambda name: int(name.split('.')[-1]), all_genes)))
    all_genes = all_genes[np.argsort(gene_indices)]
    return all_genes


def get_total_passed_sites(all_genes, passed_site_map, id1, id2):
    all_num_sites = np.zeros(len(all_genes))
    for i in xrange(len(all_genes)):
        if all_genes[i] in passed_site_map:
            all_num_sites[i] = passed_site_map[all_genes[i]
                                               ]['4D']['sites'][id1, id2]
        else:
            all_num_sites[i] = 0
    return sum(all_num_sites)


def prepare_divergence_dict(species_name):
    # load lots of data
    all_genes = get_core_gene_vector(species_name)
    checkpoint_path = os.path.join(
        config.analysis_directory, 'between_hosts_checkpoints', species_name)
    passed_sites = pickle.load(
        open(os.path.join(checkpoint_path, 'passed_sites_map.pickle'), 'rb'))
    snp_counts = pickle.load(
        open(os.path.join(checkpoint_path, 'snp_counts_map.pickle'), 'rb'))
    found_samples = pickle.load(
        open(os.path.join(checkpoint_path, 'found_samples.pickle'), 'rb'))
    num_samples = len(found_samples)
    divergence_dict = dict()
    for i in xrange(num_samples):
        for j in xrange(i + 1, num_samples):
            divergence_dict[(i, j)] = snp_counts[(i, j)] / \
                get_total_passed_sites(all_genes, passed_sites, i, j)
    return divergence_dict


def plot_for_one_species(species_name, subset=True, normalization=True):
    # loading data
    checkpoint_path = os.path.join(
        config.analysis_directory, 'between_hosts_checkpoints', species_name)
    all_runs_dict = pickle.load(
        open(os.path.join(checkpoint_path, 'all_runs_map.pickle'), 'rb'))
    snp_counts = pickle.load(
        open(os.path.join(checkpoint_path, 'snp_counts_map.pickle'), 'rb'))
    divergence_dict = prepare_divergence_dict(species_name)
    # load same clade snp cutoff
    # TODO eventually want to use only divergence
    cutoffs = json.load(open('./same_clade_snp_cutoffs.json', 'r'))
    lower_cutoff = cutoffs[species_name][0] or min(
        snp_counts.values())  # hacky way of assigning value to None
    upper_cutoff = cutoffs[species_name][1] or max(snp_counts.values())
    print("Finish loading for {}".format(species_name))

    fig = plt.figure()
    fig.set_size_inches(8, 6)
    ax = plt.gca()
    ax.set_yscale('log')

    ax.set_xlabel('Normalized site counts')
    ax.set_ylabel('Survival proba')
    ax.set_xlim((0, 15))

    null = np.random.geometric(p=0.01, size=5000)
    _ = plt.hist(null * 0.01, normed=True, cumulative=-1, bins=50,
                 histtype='step', label='Null (geometric)', color='r')
    plt.legend()

    fig_base_path = os.path.join(
        config.analysis_directory, 'run_size_survival_distributions')
    if not normalization:
        fig_base_path = os.path.join(fig_base_path, 'wrong_normalization')
    if subset:
        pairs = random.sample(all_runs_dict.keys(), 1000)
        fig_path = os.path.join(
            fig_base_path, '{}_subset.pdf'.format(species_name))
    else:
        pairs = all_runs_dict.keys()
        fig_path = os.path.join(fig_base_path, '{}.pdf'.format(species_name))

    for pair in pairs:
        div = divergence_dict[pair]
        if (snp_counts[pair] < lower_cutoff) or (snp_counts[pair] > upper_cutoff):
            continue
        if not normalization:
            div = 0.01 # use wrong normalization
        # normalize by multiplying div
        _ = ax.hist(all_runs_dict[pair][3] * div, normed=True, cumulative=-1, bins=50, histtype='step', log=True,
                    color=(0., 0., 0.2, 0.02))
    plt.savefig(fig_path)
    plt.close()


t0 = time.time()
for species_name in os.listdir(os.path.join(config.analysis_directory, 'between_hosts_checkpoints')):
    if species_name.startswith('.'):
        continue
    print("plotting for {} at {} min".format(species_name, (time.time()-t0)/60))
    plot_for_one_species(species_name, subset=True, normalization=False)
