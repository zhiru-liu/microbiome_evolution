import numpy as np
import itertools
import random
import os
from utils import close_pair_utils, parallel_utils
import config


def compute_runs(dh, good_idxs):
    """
    Computing all homozygous runs for a given set of pairs/samples
    :param dh: DataHoarder instance for a given species
    :param good_idxs: list of sample idxs to process
    :return:
    """
    good_chromo = dh.chromosomes[dh.general_mask]

    run_data = {}
    for pair in good_idxs:
        # get the snp data
        snp_vec, coverage_arr = dh.get_snp_vector(pair)
        runs = parallel_utils.compute_runs_all_chromosomes(snp_vec, good_chromo[coverage_arr])
        run_data[pair] = runs
    return run_data


def generate_within_sample_idxs(dh, typical_cutoff, clade_cutoff=None):
    """
    :param dh: DataHoarder instance
    :param typical_cutoff: divergence cutoff for keeping typically diverged pairs
    e.g. 0.005 syn div for B. vulgatus
    :param clade_cutoff: if provided, will be used to separate within-clade and between-clade cases
    :return: one or two lists of idxs of samples
    """
    species_name = dh.species_name
    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'within_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')
    single_subject_samples = dh.get_single_subject_idxs()
    if clade_cutoff is None:
        mask = div_mat[single_subject_samples] > typical_cutoff
        return single_subject_samples[mask]
    else:
        mask1 = (div_mat[single_subject_samples] > typical_cutoff) & \
                (div_mat[single_subject_samples] < clade_cutoff)
        mask2 = (div_mat[single_subject_samples] > typical_cutoff) & \
                (div_mat[single_subject_samples] >= clade_cutoff)
        return single_subject_samples[mask1], single_subject_samples[mask2]


def generate_between_sample_idxs(dh, typical_cutoff, num_pairs=100, clade_cutoff=None):
    species_name = dh.species_name
    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'between_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')
    if clade_cutoff is None:
        idx1, idx2 = np.where(div_mat > typical_cutoff)
        idx1, idx2 = idx1[idx1 > idx2], idx2[idx1 < idx2]  # dedup
        all_pairs = zip(idx1, idx2)
        if num_pairs >= len(all_pairs):
            return all_pairs
        else:
            return random.sample(all_pairs, num_pairs)
    else:
        # first perform hierarchical clustering into two clades
        cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(div_mat, clade_cutoff)
        clade1, clade2 = cluster_dict[1], cluster_dict[2]
        # next form all within-clade pairs that are typically diverged
        clade1_pairs = [x for x in itertools.combinations(clade1, 2) if div_mat[x] > typical_cutoff]
        clade2_pairs = [x for x in itertools.combinations(clade2, 2) if div_mat[x] > typical_cutoff]
        within_clade_pairs = clade1_pairs + clade2_pairs
        # next form all between-clade pairs that are typically diverged
        between_clade_pairs = [x for x in itertools.product(clade1, clade2) if div_mat[x] > typical_cutoff]
        # finally, sample desired number of pairs from two lists
        if num_pairs >= len(within_clade_pairs):
            within_clade_pairs = within_clade_pairs
        else:
            within_clade_pairs = random.sample(within_clade_pairs, num_pairs)
        if num_pairs >= len(between_clade_pairs):
            between_clade_pairs = between_clade_pairs
        else:
            between_clade_pairs = random.sample(between_clade_pairs, num_pairs)
        return within_clade_pairs, between_clade_pairs


def compute_max_runs(runs_data):
    return np.array(map(max, runs_data.values()))


def _filter_and_sum(vals, threshold):
        return np.sum(vals[vals > threshold])


def compute_cumu_runs(runs_data, threshold):
    return np.array(list(map(lambda x: _filter_and_sum(x, threshold), runs_data.values())))
