import numpy as np
import itertools
import os
from utils import close_pair_utils, parallel_utils, BSMC_utils
import config


def compute_pileup_for_clusters(cluster_dict, get_run_start_end, genome_len, mean_div, surprise_index):
    thresholds = surprise_index / mean_div
    genome_len = int(genome_len)
    cumu_runs = np.zeros([genome_len, len(thresholds)])
    num_clusters = len(cluster_dict)

    num_comparisons = 0
    for i, j in itertools.combinations(range(1, num_clusters+1), 2):
        # i, j are cluster ids
        num_comparisons += 1
        num_pairs = 0.
        tmp_runs = np.zeros(cumu_runs.shape)
        for l, m in itertools.product(cluster_dict[i], cluster_dict[j]):
            # l, m are sample ids
            num_pairs += 1
            all_start_end = get_run_start_end(l, m, thresholds)
            for k, dat in enumerate(all_start_end):
                # k represent which threshold
                for start, end in dat:
                    tmp_runs[start:end, k] += 1
        tmp_runs /= num_pairs
        cumu_runs += tmp_runs
    cumu_runs /= num_comparisons
    return cumu_runs

class Pileup_Helper:
    def __init__(self, species_name, allowed_variants=["4D"], clade_cutoff=None, close_pair_cutoff=1e-3):
        self.dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=allowed_variants)
        self.good_chromo = self.dh.chromosomes[self.dh.general_mask]

        div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence', 'between_hosts',
                               '%s.csv' % species_name)
        self.div_mat = np.loadtxt(div_dir, delimiter=',')

        clade_cutoff = clade_cutoff if clade_cutoff else 1
        d = close_pair_utils.get_clusters_from_pairwise_matrix(self.div_mat, threshold=clade_cutoff)
        clade_cluster = 1 + np.argmax(map(len, d.values()))  # keep the largest clade
        clade_samples = d[clade_cluster]

        single_subject_samples = self.dh.get_single_subject_idxs()
        self.good_samples = np.intersect1d(single_subject_samples, clade_samples)

        self.close_pair_cutoff = close_pair_cutoff
        self.cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(
            self.div_mat[self.good_samples, :][:, self.good_samples], threshold=close_pair_cutoff)

    def update_close_pair_cutoff(self, new_cutoff):
        self.close_pair_cutoff = new_cutoff
        self.cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(
            self.div_mat[self.good_samples, :][:, self.good_samples], threshold=new_cutoff)

    def cluster_id_to_sample_id(self, cluster_id):
        # sample id is used by self.dh, indexes all samples
        # cluster id indexes only the good samples
        return self.good_samples[cluster_id]

    def get_event_start_end(self, idx1, idx2, thresholds):
        i1 = self.cluster_id_to_sample_id(idx1)
        i2 = self.cluster_id_to_sample_id(idx2)
        pair = (i1, i2)
        # get the snp data
        snp_vec, coverage_arr = self.dh.get_snp_vector(pair)
        # get the location in the full array
        snp_to_core = np.nonzero(coverage_arr)[0]
        snp_genome_locs = snp_to_core[np.nonzero(snp_vec)[0]]

        runs = parallel_utils.compute_runs_all_chromosomes(snp_vec, self.good_chromo[coverage_arr])

        all_dat = []
        for i in range(len(thresholds)):
            threshold = thresholds[i]
            event_starts = snp_genome_locs[:-1][runs > threshold]
            event_ends = snp_genome_locs[1:][runs > threshold]
            all_dat.append(zip(event_starts, event_ends))
        return all_dat


def get_event_start_end_BSMC(sim_data, genome_len, idx1, idx2, thresholds):
    site_locations, runs = BSMC_utils.compare_two_samples(idx1, idx2, sim_data, genome_len)
    site_locations = np.array(site_locations * genome_len).astype(int)
    runs = site_locations[1:] - site_locations[:-1]
    all_dat = []
    for k in range(len(thresholds)):
        threshold = thresholds[k]
        event_starts = site_locations[:-1][runs > threshold]
        event_ends = site_locations[1:][runs > threshold]
        all_dat.append(zip(event_starts, event_ends))
    return all_dat