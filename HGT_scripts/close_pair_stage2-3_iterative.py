import sys
import os
import logging
import json
import pickle
import numpy as np
import traceback
import pandas as pd
sys.path.append("..")
import config
import cphmm.cphmm as hmm
from utils import parallel_utils, close_pair_utils, typical_pair_utils


def init_hmm(species_name, approx_genome_len, transfer_length, block_size=config.second_pass_block_size):
    # initialize the hmm with default params
    # clonal emission and transfer rate will be fitted per sequence later in the pipeline
    num_blocks = approx_genome_len / block_size
    transfer_counts = 20.  # will later iteratively update
    clonal_div = 5e-5  # same as above exact number does not matter

    transfer_rate = transfer_counts / num_blocks
    transfer_length = transfer_length / block_size
    clonal_emission = clonal_div * block_size
    cphmm = hmm.ClosePairHMM(species_name=species_name, block_size=block_size,
                             transfer_rate=transfer_rate, clonal_emission=clonal_emission,
                             transfer_length=transfer_length, n_iter=5)
    return cphmm


def decode_one_pass(dh, prev_second_pass_path, next_second_pass_path, separate_clades=False):
    """
    Takes the previous round of decoding, calculate the mean transfer length, and use that as the hmm parameter for another
    round
    """
    good_chromo = dh.chromosomes[dh.general_mask]  # will be used in contig-wise transfer computation

    # obtaining the mean genome length and pairs to process
    first_pass_save_path = os.path.join(config.analysis_directory,
                                        "closely_related", "first_pass", "{}.pickle".format(dh.species_name))
    first_pass_stats = pd.read_pickle(first_pass_save_path)
    mean_total_blocks = first_pass_stats['num_total_blocks'].mean()
    snp_block_cutoff = 0.5 * mean_total_blocks  # will process if half of the genome is empty
    good_pairs = first_pass_stats[first_pass_stats['snp_blocks'] < snp_block_cutoff]['pair_idxs']
    mean_genome_len = mean_total_blocks * config.first_pass_block_size

    if separate_clades:
        # good_pairs = good_pairs[:5]
        clade_cutoff_bin = config.empirical_histogram_bins  # for B vulgatus/ A. shahii separate clade
    else:
        clade_cutoff_bin = None

    data = pickle.load(open(os.path.join(prev_second_pass_path, '{}.pickle'.format(dh.species_name)), 'rb'))
    transfer_counts, all_transfer_df = close_pair_utils.merge_and_filter_transfers(data, separate_clade=False,
                                                                                   merge_threshold=0,
                                                                                   filter_threshold=5)
    mean_transfer_length = all_transfer_df['lengths'].to_numpy().astype(float).mean() * config.second_pass_block_size
    logging.info("Last round's mean transfer length is {}".format(mean_transfer_length))

    # init a new hmm and repeat second pass
    cphmm = init_hmm(dh.species_name, mean_genome_len, transfer_length=mean_transfer_length)

    dat = dict()
    dat['starts'] = []
    dat['ends'] = []
    # dat['clonal snps'] = []
    # dat['transfer snps'] = []
    dat['clonal divs'] = []
    dat['genome lengths'] = []
    dat['clonal lengths'] = []
    dat['pairs'] = list(good_pairs)
    processed_count = 0
    for pair in good_pairs:
        snp_vec, snp_mask = dh.get_snp_vector(pair)
        chromosomes = good_chromo[snp_mask]
        try:
            # starts, ends, T_approx = close_pair_utils.fit_and_count_transfers_all_chromosomes(
            # starts, ends, transfer_snp, clonal_snp, genome_len, clonal_len = \
            #     close_pair_utils.fit_and_count_transfers_all_chromosomes(
            #     snp_vec, chromosomes, cphmm, block_size, clade_cutoff_bin=clade_cutoff_bin)
            starts, ends, clonal_div, genome_len, clonal_len = \
                close_pair_utils.fit_and_count_transfers_all_chromosomes(
                    snp_vec, chromosomes, cphmm, config.second_pass_block_size, clade_cutoff_bin=clade_cutoff_bin)
        except:
            e = sys.exc_info()[0]
            tb = traceback.format_exc()
            print(pair)
            print(tb)
            raise e
        dat['starts'].append(starts)
        dat['ends'].append(ends)
        # dat['transfer snps'].append(transfer_snp)
        # dat['clonal snps'].append(clonal_snp)
        dat['clonal divs'].append(clonal_div)
        dat['genome lengths'].append(genome_len)
        dat['clonal lengths'].append(clonal_len)

        processed_count += 1
        if processed_count % 100 == 0:
            logging.info("Finished %d out of %d pairs" % (processed_count, len(good_pairs)))

    savepath = os.path.join(next_second_pass_path, '{}.pickle'.format(dh.species_name))
    pickle.dump(data, open(savepath, 'wb'))


def process_one_species(species_name):
    dh = parallel_utils.DataHoarder(species_name, mode='QP')
    for i in range(3):
        prev_path = os.path.join(config.analysis_directory, 'closely_related', 'iter_second_third_passes', 'iter{}'.format(i), 'second_pass')
        next_path = os.path.join(config.analysis_directory, 'closely_related', 'iter_second_third_passes', 'iter{}'.format(i+1), 'second_pass')
        if not os.path.exists(next_path):
            os.makedirs(next_path)
        decode_one_pass(dh, prev_path, next_path)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

process_one_species('Akkermansia_muciniphila_55290')
