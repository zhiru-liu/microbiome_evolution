import sys
import os
import logging
import json
import numpy as np
sys.path.append("..")
import config
from utils import parallel_utils, close_pair_utils, hmm

BLOCK_SIZE = 1000
CLONAL_FRAC_CUTOFF = 0.5

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')


def process_one_species(species_name, div_cutoff, hmm_init_means=[0.5, 10]):
    dh = parallel_utils.DataHoarder(species_name, mode="QP")
    good_chromo = dh.chromosomes[dh.general_mask] # will be used in contig-wise transfer computation

    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'between_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')
    pairs = close_pair_utils.find_close_pairs(div_cutoff, div_mat, dh.get_single_subject_idxs())
    logging.info("After divergence cutoff, {} has {} pairs".format(species_name, len(pairs)))
    if len(pairs) < 5:
        logging.info("Too few pairs, skipping")
        return None

    logging.info("Coarse-graining the genome into blocks of size {}".format(BLOCK_SIZE))
    first_pass_stats = close_pair_utils.process_close_pairs_first_pass(dh, pairs, BLOCK_SIZE)
    mean_total_blocks = first_pass_stats['num_total_blocks'].mean()
    # use num of snp block as an estimate for clonal fraction
    # throw away pairs with too many blocks covered
    snp_block_cutoff = (1 - CLONAL_FRAC_CUTOFF) * mean_total_blocks
    second_pass_stats = first_pass_stats[
        first_pass_stats['snp_blocks'] < snp_block_cutoff].copy()
    good_pairs = second_pass_stats['pair_idxs']
    logging.info("After first pass, {} has {} pairs".format(species_name, len(good_pairs)))

    num_states = len(hmm_init_means)
    logging.info("Using a {} state HMM to detect transfers".format(num_states))
    phmm = hmm.PoissonHMM(init_means=hmm_init_means,
                          n_components=num_states, params='st')
    num_transfers = []
    transfer_lens = []
    T_approxs = []
    for pair in good_pairs:
        snp_vec, snp_mask = dh.get_snp_vector(pair)
        chromosomes = good_chromo[snp_mask]
        num_transfer, transfer_len, T_approx = close_pair_utils.fit_and_count_transfers_all_chromosomes(
            snp_vec, chromosomes, phmm, BLOCK_SIZE)
        num_transfers.append(num_transfer)
        transfer_lens.append(transfer_len)
        T_approxs.append(T_approx)
    second_pass_stats['num_transfers'] = num_transfers
    second_pass_stats['T_approx'] = T_approxs
    second_pass_stats['transfer_len'] = transfer_lens

    return first_pass_stats, second_pass_stats


cutoff_dict = json.load(open('./same_clade_div_cutoffs.json', 'r'))
base_dir = 'zarr_snps'
for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
    if species_name.startswith('.'):
        continue
    res = process_one_species(species_name, cutoff_dict[species_name][0])
    if res is not None:
        df1, df2 = res
        save_path = os.path.join(config.analysis_directory,
                                 "closely_related", "first_pass", "{}.csv".format(species_name))
        df1.to_csv(save_path)
        save_path = os.path.join(config.analysis_directory,
                                 "closely_related", "second_pass", "{}.csv".format(species_name))
        df2.to_csv(save_path)
