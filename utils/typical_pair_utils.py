import numpy as np
import itertools
import random
import os
from utils import close_pair_utils, parallel_utils
import config


def load_clonal_frac_mat(species_name, desired_samples=None):
    """
    Load precomputed clonal fraction matrix. If desired_samples is supplied, will sort
    matrix according to desired_samples
    :param species_name:
    :param desired_samples: np.array
    :return: np.array of clonal fractions between pairs of QP samples
    """
    clonal_frac_dir = os.path.join(config.analysis_directory, 'pairwise_clonal_fraction',
                               'between_hosts', '%s.csv' % species_name)
    if not os.path.exists(clonal_frac_dir):
        return None
    clonal_frac_mat = np.loadtxt(clonal_frac_dir, delimiter=',')
    if desired_samples is None:
        return clonal_frac_mat
    else:
        QP_samples = parallel_utils.get_QP_samples(species_name)
        name_to_idx = {x:i for i, x in enumerate(QP_samples)}
        desired_idxs = np.array([name_to_idx[x] for x in desired_samples])
        return clonal_frac_mat[desired_idxs, :][:, desired_idxs]


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


def generate_within_sample_idxs(dh, clonal_frac_cutoff=0.15, clade_cutoff=None):
    """
    :param dh: DataHoarder instance
    :param clonal_frac_cutoff: clonal fraction cutoff for keeping typically diverged pairs
    :param clade_cutoff: if provided, will be used to separate within-clade and between-clade cases
    :return: one or two lists of idxs of samples
    """
    species_name = dh.species_name
    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'within_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')

    clonal_frac_mat = load_clonal_frac_mat(species_name)

    single_subject_samples = dh.get_single_subject_idxs()
    if clade_cutoff is None:
        mask = clonal_frac_mat[single_subject_samples] < clonal_frac_cutoff
        return single_subject_samples[mask]
    else:
        mask1 = (clonal_frac_mat[single_subject_samples] < clonal_frac_cutoff) & \
                (div_mat[single_subject_samples] < clade_cutoff)
        mask2 = (clonal_frac_mat[single_subject_samples] < clonal_frac_cutoff) & \
                (div_mat[single_subject_samples] >= clade_cutoff)
        return single_subject_samples[mask1], single_subject_samples[mask2]


def generate_between_sample_idxs(dh, clonal_frac_cutoff=0.15, num_pairs=100, clade_cutoff=None):
    species_name = dh.species_name
    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'between_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')
    clonal_frac_mat = load_clonal_frac_mat(species_name)

    if clade_cutoff is None:
        idx1, idx2 = np.where(clonal_frac_mat < clonal_frac_cutoff)
        idx1, idx2 = idx1[idx1 > idx2], idx2[idx1 > idx2]  # dedup
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
        clade1_pairs = [x for x in itertools.combinations(clade1, 2) if clonal_frac_mat[x] < clonal_frac_cutoff]
        clade2_pairs = [x for x in itertools.combinations(clade2, 2) if clonal_frac_mat[x] < clonal_frac_cutoff]
        within_clade_pairs = clade1_pairs + clade2_pairs
        # next form all between-clade pairs that are typically diverged
        between_clade_pairs = [x for x in itertools.product(clade1, clade2) if clonal_frac_mat[x] < clonal_frac_cutoff]
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
