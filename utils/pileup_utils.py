import numpy as np
import pandas as pd
import pickle
import json
import itertools
import os
from collections import Counter
from utils import close_pair_utils, parallel_utils, BSMC_utils, typical_pair_utils
import config


def compute_pileup_for_clusters(cluster_dict, get_run_start_end, genome_len, thresholds, cache_start_end=None):
    """
    General function for computing pileup curves; close genomes will be clustered according to cluster_dict
    :param cluster_dict: precomputed clusters using pairwise divergence matrix (using function in close_pair_utils.py)
    :param get_run_start_end: function that compute the list of event starts and ends for each threshold.
    Use get_event_start_end_BSMC for BSMC simulated data. Use Pileup_Helper.get_event_start_end for real species data
    :param genome_len: rough length of the genome. Used in determining the threshold lengths
    :param thresholds: the threshold lengths for determining if accumulating a run
    :param cache_start_end: directory; if provided, will cache all runs to this directory in json format
    :return: np array of shape (genome length, number of thresholds)
    """
    genome_len = int(genome_len)
    cumu_runs = np.zeros([genome_len, len(thresholds)])
    num_clusters = len(cluster_dict)

    num_comparisons = 0
    for i, j in itertools.combinations(range(1, num_clusters+1), 2):
        # i, j are cluster ids
        num_pairs = 0.
        tmp_runs = np.zeros(cumu_runs.shape)
        for l, m in itertools.product(cluster_dict[i], cluster_dict[j]):
            # l, m are sample ids
            all_start_end = get_run_start_end(l, m, thresholds)
            if all_start_end is None:
                continue
            if cache_start_end is not None:
                # pickle.dump(all_start_end, open(os.path.join(cache_start_end, '%s_%s.pkl' % (l, m)), 'wb'))
                json.dump(all_start_end, open(os.path.join(cache_start_end, '%s_%s.json' % (l, m)), 'w'))
            num_pairs += 1
            for start_end in all_start_end:
                # iterating over contigs
                for k, dat in enumerate(start_end):
                    # k represent which threshold
                    for start, end in dat:
                        tmp_runs[start:end+1, k] += 1  # in new version (as of 08.18.21) end is inclusive
        if num_pairs > 0:
            # not couting the cluster pair if no typical pair between them
            tmp_runs /= num_pairs
            num_comparisons += 1
        cumu_runs += tmp_runs
    cumu_runs /= num_comparisons
    print("Performed in total %d pairwise comparisons" % num_comparisons)
    return cumu_runs


def compute_pileup_for_pairs(pairs, get_run_start_end, genome_len, thresholds, cache_start_end=None):
    """
    Compute pileup for a given set of pairs
    :param pairs:
    :param get_run_start_end:
    :param genome_len:
    :param thresholds:
    :return:
    """
    genome_len = int(genome_len)
    cumu_runs = np.zeros([genome_len, len(thresholds)])
    num_pairs = 0

    for l, m in pairs:
        # l, m are sample ids
        all_start_end = get_run_start_end(l, m, thresholds, actual_index=True)
        if all_start_end is None:
            continue
        if cache_start_end is not None:
            # pickle.dump(all_start_end, open(os.path.join(cache_start_end, '%s_%s.pkl' % (l, m)), 'wb'))
            json.dump(all_start_end, open(os.path.join(cache_start_end, '%s_%s.json' % (l, m)), 'w'))
        num_pairs += 1
        for start_end in all_start_end:
            # iterating over contigs
            for k, dat in enumerate(start_end):
                # k represent which threshold
                for start, end in dat:
                    cumu_runs[start:end+1, k] += 1  # in new version (as of 08.18.21) end is inclusive
    if num_pairs > 0:
        # not couting the cluster pair if no typical pair between them
        cumu_runs /= num_pairs
    return cumu_runs


def compute_pileup_from_cache(files, genome_len, allowed_threshold=1):
    """
    For performance consideration, only allow one threshold each time
    :param files:
    :param genome_len:
    :param allowed_threshold:
    :return:
    """
    cumu_runs = np.zeros(genome_len)
    k = allowed_threshold
    for f in files:
        # all_start_end = pickle.load(open(f, 'rb'))
        all_start_end = json.load(open(f, 'r'))
        for start_end in all_start_end:
            # iterating over contigs
            # k represent which threshold
            for start, end in start_end[k]:
                cumu_runs[start:end+1] += 1  # in new version (as of 08.18.21) end is inclusive
    return cumu_runs / float(len(files))


def compute_pileup_for_cluster_between_clades(cluster1_dict, cluster2_dict, get_run_start_end, genome_len, thresholds, cache_start_end=None):
    genome_len = int(genome_len)
    cumu_runs = np.zeros([genome_len, len(thresholds)])

    num_comparisons = 0
    print("Cluster 1 size: %d; cluster 2 size: %d" % (len(cluster1_dict), len(cluster2_dict)))
    for i, j in itertools.product(cluster1_dict.keys(), cluster2_dict.keys()):
        # i, j are cluster ids
        num_pairs = 0.
        tmp_runs = np.zeros(cumu_runs.shape)
        for l, m in itertools.product(cluster1_dict[i], cluster2_dict[j]):
            # l, m are sample ids
            all_start_end = get_run_start_end(l, m, thresholds, minor_cluster=True)
            if all_start_end is None:
                continue
            if cache_start_end is not None:
                # pickle.dump(all_start_end, open(os.path.join(cache_start_end, '%s_%s.pkl' % (l, m)), 'wb'))
                json.dump(all_start_end, open(os.path.join(cache_start_end, '%s_%s.json' % (l, m)), 'w'))
            num_pairs += 1
            for start_end in all_start_end:
                # iterating over contigs
                for k, dat in enumerate(start_end):
                    # k represent which threshold
                    for start, end in dat:
                        tmp_runs[start:end+1, k] += 1  # in new version (as of 08.18.21) end is inclusive
        if num_pairs > 0:
            # not couting the cluster pair if no typical pair between them
            tmp_runs /= num_pairs
            num_comparisons += 1
        cumu_runs += tmp_runs
    cumu_runs /= num_comparisons
    print("Performed in total %d pairwise comparisons" % num_comparisons)
    return cumu_runs


def compute_pileup_for_within_host(dh, thresholds, clade_cutoff=None, within_clade=True, cache_start_end=None):
    """
    Computing same pileup dist for within host data
    :param dh: DataHoarder instance that loads within host data for a given species
    :param thresholds: list of threshold lengths for filtering out sharing events
    :return: np array of shape (genome length, number of thresholds)
    """
    num_reps = 0
    single_subject_samples = dh.get_single_subject_idxs()
    good_chromo = dh.chromosomes[dh.general_mask]

    within_cumu_runs = np.zeros([np.sum(dh.general_mask), len(thresholds)])
    if len(single_subject_samples) == 0:
        return None
    for pair in single_subject_samples:
        # get the snp data
        snp_vec, coverage_arr = dh.get_snp_vector(pair)
        if snp_vec is None:
            # sample did not pass coverage filter
            continue
        if (clade_cutoff is not None) and within_clade:
            if np.sum(snp_vec) / float(np.sum(coverage_arr)) > clade_cutoff:  # removing two clade pairs
                continue
        elif (clade_cutoff is not None) and (not within_clade):
            if np.sum(snp_vec) / float(np.sum(coverage_arr)) <= clade_cutoff:  # removing same clade pairs
                continue

        clonal_frac = close_pair_utils.compute_clonal_fraction(snp_vec, config.first_pass_block_size)
        if clonal_frac > config.typical_clonal_fraction_cutoff:
            # consider only typically diverged pairs
            continue
        num_reps += 1
        # get the location in the full array
        snp_to_core = np.nonzero(coverage_arr)[0]
        chromosomes = good_chromo[coverage_arr]

        all_start_end = compute_passed_starts_ends(snp_vec, chromosomes, snp_to_core, thresholds)
        if cache_start_end is not None:
            # pickle.dump(all_start_end, open(os.path.join(cache_start_end, '%s.pkl' % pair), 'wb'))
            json.dump(all_start_end, open(os.path.join(cache_start_end, '%s.json' % pair), 'w'))

        for start_end in all_start_end:
            for k, dat in enumerate(start_end):
                # k represent which threshold
                for start, end in dat:
                    within_cumu_runs[start:end+1, k] += 1
    within_cumu_runs /= float(num_reps)
    print("Performed in total %d pairwise comparisons" % num_reps)
    return within_cumu_runs


def get_event_start_end_BSMC(sim_data, genome_len, idx1, idx2, thresholds):
    """
    Function for obtaining all sharing events over thresholds for genome pairs (idx1, idx2)
    :param sim_data: np array loaded with relevant BSMC_util function
    :param genome_len: Same length that supplied to BSMC program
    :param idx1: index of the first genome (0 indexed)
    :param idx2: index of the second genome
    :param thresholds: List of threshold lengths to filter events. In number of sites
    :return: A list containing a list of (start loc, end loc) for each threshold
    """
    site_locations, runs = BSMC_utils.compare_two_samples(idx1, idx2, sim_data, genome_len)
    site_locations = np.array(site_locations * genome_len).astype(int)
    runs = site_locations[1:] - site_locations[:-1]
    all_dat = []
    for k in range(len(thresholds)):
        threshold = thresholds[k]
        event_starts = site_locations[:-1][runs > threshold]
        event_ends = site_locations[1:][runs > threshold]
        all_dat.append(zip(event_starts, event_ends))
    return [all_dat]  # the outer [] is for compatibility with multi-chromosome data


def compute_passed_starts_ends(snp_vec, chromosomes, locations, thresholds):
    """
    For a bool vector, potentially concat of multiple contigs, find all the starts and ends (as given by location)
    of runs longer than thresholds.
    :param snp_vec:
    :param chromosomes:
    :param locations:
    :param thresholds:
    :return: List of len # chromosomes, each element of which is a list of len # thresholds, each element of which
    is a list of # of passed events. The inner most elements are a tuple of (start, end)
    """
    index_offset = 0
    all_dats = []
    for chromo in pd.unique(chromosomes):
        # loop over contigs
        subvec = snp_vec[chromosomes==chromo]
        runs, starts, ends = parallel_utils._compute_runs_single_chromosome(subvec, return_locs=True)

        all_dat = []
        for i in range(len(thresholds)):
            # saving all the start,end pairs in core genome coordinates for all runs passing threshold
            threshold = thresholds[i]
            subvec_starts = starts[runs > threshold]
            subvec_ends = subvec_starts + runs[runs > threshold] - 1
            event_starts = locations[subvec_starts + index_offset]
            event_ends = locations[subvec_ends + index_offset]
            all_dat.append(zip(event_starts, event_ends))
        index_offset += len(subvec)
        all_dats.append(all_dat)
    return all_dats


def enrichment_test(gene_vector, site_mask, pass_func, shuffle_size=1, shuffle_reps=1e4):
    """
    Perform permutation of the genome to study whether certain regions are enriched for some criteria
    Genes are grouped in clusters and then permuted together to preserve local clustering of functions
    :param gene_vector: original vector of genes of size genome length
    :param site_mask: for filtering interested regions
    :param pass_func: a function that takes a gene name and return T/F
    :param shuffle_size: rough number of genes to be grouped together (but not exactly this number, will use array_split
    to find an approximate size_
    :param shuffle_reps: number of repetitions
    :return: a list of permutation results containing the number of genes in selected regions that pass the pass_func
    """
    unique_genes = pd.unique(gene_vector)
    gene_lengths = Counter(gene_vector)
    gene_clusters = np.array_split(unique_genes, len(unique_genes)//shuffle_size)
    passed_counts = []
    for i in range(int(shuffle_reps)):
        shuffled_clusters = np.random.permutation(gene_clusters)
        shuffled_genes = np.hstack(shuffled_clusters)
        lengths = [gene_lengths[x] for x in shuffled_genes]
        new_gene_vector = np.repeat(shuffled_genes, lengths)

        passed_genes = pd.unique(new_gene_vector[site_mask])
        passed_count = np.sum([pass_func(x) for x in passed_genes])
        passed_counts.append(passed_count)
    return passed_counts


class Pileup_Helper:
    def __init__(self, species_name, allowed_variants=["4D"], clade_cutoff=None, close_pair_cutoff=0.95):
        """
        Wrapper over DataHoarder to provide pileup specific functions
        :param species_name:
        :param allowed_variants:
        :param clade_cutoff: Only for B vulgatus to select the major clade
        :param close_pair_cutoff: For reducing overcounting of sharing
        """
        self.dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=allowed_variants)
        self.good_chromo = self.dh.chromosomes[self.dh.general_mask]

        div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence', 'between_hosts',
                               '%s.csv' % species_name)
        self.div_mat = np.loadtxt(div_dir, delimiter=',')
        self.cf_mat = typical_pair_utils.load_clonal_frac_mat(species_name)

        clade_cutoff = clade_cutoff if clade_cutoff else 1
        # form first order clusters using clade divergence cutoff
        d = close_pair_utils.get_clusters_from_pairwise_matrix(self.div_mat, threshold=clade_cutoff)
        cluster_ids = np.argsort(map(len, d.values()))
        clade_cluster = 1 + cluster_ids[-1]  # keep the largest clade
        clade_samples = d[clade_cluster]

        single_subject_samples = self.dh.get_single_subject_idxs()
        self.good_samples = np.intersect1d(single_subject_samples, clade_samples)

        # here only cluster highly similar strains to avoid overcounting too much
        self.close_pair_cutoff = close_pair_cutoff
        self.cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(
            1 - self.cf_mat[self.good_samples, :][:, self.good_samples],
            threshold=1-close_pair_cutoff)
        print("%s has %d closely-related clusters" % (species_name, len(self.cluster_dict)))

        if len(cluster_ids) > 1:
            clade_cluster = 1 + cluster_ids[-2]  # keep the second largest clade
            clade_samples = d[clade_cluster]
            self.minor_clade_samples = np.intersect1d(single_subject_samples, clade_samples)
            self.minor_cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(
                1 - self.cf_mat[self.minor_clade_samples, :][:, self.minor_clade_samples],
                threshold=1-close_pair_cutoff)

    def update_close_pair_cutoff(self, new_cutoff):
        self.close_pair_cutoff = new_cutoff
        self.cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(
            self.div_mat[self.good_samples, :][:, self.good_samples], threshold=new_cutoff)

    def cluster_id_to_sample_id(self, cluster_id):
        # sample id is used by self.dh, indexes all samples
        # cluster id indexes only the good samples
        return self.good_samples[cluster_id]

    def cluster_id_to_minor_sample_id(self, cluster_id):
        return self.minor_clade_samples[cluster_id]

    def get_event_start_end(self, idx1, idx2, thresholds, cf_cutoff=None, minor_cluster=False, actual_index=False):
        """
        :param idx1: cluster index of the first sample
        :param idx2: cluster index of the second sample
        :param thresholds: list of run length thresholds
        :param cf_cutoff: if not supplied, will use default value supplied in config. use this cutoff to skip
        pairs with cf too high
        :param actual_index: if True, then idx1,idx2 will be passed directly to dh.get_snp_vector
        :param minor_cluster: if true, idx2 will be interpreted as the index for the minor clade
        :return:
        """
        if actual_index:
            pair = (idx1, idx2)
        else:
            i1 = self.cluster_id_to_sample_id(idx1)
            if minor_cluster:
                i2 = self.cluster_id_to_minor_sample_id(idx2)
            else:
                i2 = self.cluster_id_to_sample_id(idx2)
            pair = (i1, i2)
        # get the snp data
        snp_vec, coverage_arr = self.dh.get_snp_vector(pair)
        # check whether pair has lots of clonal regions
        clonal_frac = close_pair_utils.compute_clonal_fraction(snp_vec, config.first_pass_block_size)
        cutoff = cf_cutoff if cf_cutoff else config.typical_clonal_fraction_cutoff
        if clonal_frac > cutoff:
            # consider only typically diverged pairs
            return None

        # get the location in the full array
        snp_to_core = np.nonzero(coverage_arr)[0]

        chromosomes = self.good_chromo[coverage_arr]
        return compute_passed_starts_ends(snp_vec, chromosomes, snp_to_core, thresholds)
