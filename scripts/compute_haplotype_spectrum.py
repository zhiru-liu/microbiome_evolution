import numpy as np
import os
import itertools
import sys
sys.path.append("..")
from utils import pileup_utils, BSMC_utils, close_pair_utils
import config


def compute_local_pd_for_species(species_name, regions):
    """

    :param species_name:
    :param regions: list of tuples (region start site, region end site)
    :return: List of local pairwise distance matrices
    """
    if 'vulgatus' in species_name:
        clade_cutoff = 0.03
    else:
        clade_cutoff = None
    ph = pileup_utils.Pileup_Helper(species_name, clade_cutoff=clade_cutoff)
    num_samples = ph.good_samples.shape[0]
    local_pd_mats = []
    for _ in regions:
        local_pd_mats.append(np.zeros((num_samples, num_samples)))

    num_finished = 0
    for i, j in itertools.combinations(range(num_samples), 2):
        idx1 = ph.good_samples[i]
        idx2 = ph.good_samples[j]

        snp_vec, covered = ph.dh.get_snp_vector((idx1, idx2))
        core_locs = np.where(covered)[0]

        for k, (region_start, region_end) in enumerate(regions):
            mask = (core_locs < region_end) & (core_locs >= region_start)
            sub_vec = snp_vec[mask]
            div = np.sum(sub_vec) / float(len(sub_vec))
            local_pd_mats[k][i, j] = div
        num_finished += 1
    return local_pd_mats


def compute_local_pd_for_BSMC(dat_path, genome_len, regions):
    sim_data = BSMC_utils.load_data(dat_path)
    num_samples = sim_data.shape[1] - 1
    local_pd_mats = []
    for _ in regions:
        local_pd_mats.append(np.zeros((num_samples, num_samples)))

    num_finished = 0
    for i, j in itertools.combinations(range(num_samples), 2):
        snp_vec = BSMC_utils.get_full_snp_vector(i, j, sim_data, genome_len)

        for k, (region_start, region_end) in enumerate(regions):
            sub_vec = snp_vec[region_start:region_end]
            div = np.sum(sub_vec) / float(len(sub_vec))
            local_pd_mats[k][i, j] = div
        num_finished += 1
    return local_pd_mats

def pd_mat_to_sorted_haplotypes(local_pd_mats):
    all_dat = []
    for pd_mat in local_pd_mats:
        clusters = close_pair_utils.get_clusters_from_pairwise_matrix(pd_mat, threshold=0)
        cluster_sizes = list(map(len, clusters.values()))
        dat = np.sort(cluster_sizes)
        all_dat.append(dat)
    return all_dat