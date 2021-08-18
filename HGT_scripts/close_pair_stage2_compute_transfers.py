import sys
import os
import logging
import json
import pickle
import numpy as np
import traceback
sys.path.append("..")
import config
import cphmm.cphmm as hmm
from utils import parallel_utils, close_pair_utils


def init_hmm(species_name, genome_len, block_size):
    # initialize the hmm with default params
    # clonal emission and transfer rate will be fitted per sequence later in the pipeline
    num_blocks = genome_len / block_size
    transfer_counts = 20.
    clonal_div = 5e-5
    transfer_length = 1000.

    transfer_rate = transfer_counts / num_blocks
    transfer_length = transfer_length / block_size
    clonal_emission = clonal_div * block_size
    cphmm = hmm.ClosePairHMM(species_name=species_name, block_size=block_size,
                             transfer_rate=transfer_rate, clonal_emission=clonal_emission,
                             transfer_length=transfer_length, n_iter=5)
    return cphmm


def process_one_species(species_name, div_cutoff, block_size, debug=False):
    """
    :param species_name:
    :param div_cutoff: Hand annotated cutoff for first filtering of pairs
    :param block_size: Coarse-graining length scale for the genome
    :param debug: Flag to determine whether running the debug version
    :return: a DataFrame for first pass statistics, and a dict for second pass statistics (including clonal snps etc)
    """
    dh = parallel_utils.DataHoarder(species_name, mode="QP")
    good_chromo = dh.chromosomes[dh.general_mask]  # will be used in contig-wise transfer computation

    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'between_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')
    pairs = close_pair_utils.find_close_pairs(div_cutoff, div_mat, dh.get_single_subject_idxs())
    logging.info("After divergence cutoff, {} has {} pairs".format(species_name, len(pairs)))
    if len(pairs) < 5:
        logging.info("Too few pairs, skipping")
        return None

    FIRST_PASS_BLOCK_SIZE = config.first_pass_block_size
    logging.info("Coarse-graining the genome into blocks of size {}".format(FIRST_PASS_BLOCK_SIZE))
    first_pass_stats = close_pair_utils.process_close_pairs_first_pass(dh, pairs, FIRST_PASS_BLOCK_SIZE)
    mean_total_blocks = first_pass_stats['num_total_blocks'].mean()

    # use num of snp block as an estimate for clonal fraction
    # throw away pairs with too many blocks covered
    snp_block_cutoff = (1 - CLONAL_FRAC_CUTOFF) * mean_total_blocks
    second_pass_stats = first_pass_stats[
        first_pass_stats['snp_blocks'] < snp_block_cutoff].copy()
    good_pairs = second_pass_stats['pair_idxs']
    if debug:
        # good_pairs = good_pairs[:5]
        clade_cutoff_bin = config.empirical_histogram_bins  # for B vulgatus separate clade
    else:
        clade_cutoff_bin = None

    logging.info("After first pass, {} has {} pairs".format(species_name, len(good_pairs)))
    mean_genome_len = mean_total_blocks * FIRST_PASS_BLOCK_SIZE
    logging.info("Mean genome length is {} sites".format(mean_genome_len))

    logging.info("Using HMM to detect transfers")
    logging.info("Block size is {}".format(block_size))
    cphmm = init_hmm(species_name, mean_genome_len, block_size)

    dat = dict()
    dat['starts'] = []
    dat['ends'] = []
    dat['clonal snps'] = []
    dat['transfer snps'] = []
    dat['genome lengths'] = []
    dat['clonal lengths'] = []
    dat['pairs'] = list(good_pairs)
    processed_count = 0
    for pair in good_pairs:
        snp_vec, snp_mask = dh.get_snp_vector(pair)
        chromosomes = good_chromo[snp_mask]
        try:
            # starts, ends, T_approx = close_pair_utils.fit_and_count_transfers_all_chromosomes(
            starts, ends, transfer_snp, clonal_snp, genome_len, clonal_len = \
                close_pair_utils.fit_and_count_transfers_all_chromosomes(
                snp_vec, chromosomes, cphmm, block_size, clade_cutoff_bin=clade_cutoff_bin)
        except:
            e = sys.exc_info()[0]
            tb = traceback.format_exc()
            print(pair)
            print(tb)
            raise e
        dat['starts'].append(starts)
        dat['ends'].append(ends)
        dat['transfer snps'].append(transfer_snp)
        dat['clonal snps'].append(clonal_snp)
        dat['genome lengths'].append(genome_len)
        dat['clonal lengths'].append(clonal_len)

        processed_count += 1
        if processed_count % 100 == 0:
            logging.info("Finished %d out of %d pairs" % (processed_count, len(good_pairs)))
    return first_pass_stats, dat


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

CLONAL_FRAC_CUTOFF = 0.5  # config.clonal_fraction_cutoff
BLOCK_SIZE = config.second_pass_block_size
DEBUG = False

black_list = ['Bacteroides_xylanisolvens_57185', # for having extremely short contigs and short total core genome
              'Escherichia_coli_58110'] # for having extremely short contigs

cutoff_dict = json.load(open('./same_clade_div_cutoffs.json', 'r'))
base_dir = 'zarr_snps'
for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
    if species_name.startswith('.'):
        continue
    if DEBUG:
        species_name = 'Bacteroides_vulgatus_57955'
        # species_name = 'Bacteroides_massiliensis_44749'
        second_path_save_path = os.path.join(config.analysis_directory,
                                             "closely_related", "debug", "{}_two_clades.pickle".format(species_name))
    else:
        second_path_save_path = os.path.join(config.analysis_directory,
                                 "closely_related", "second_pass", "{}.pickle".format(species_name))
    if os.path.exists(second_path_save_path):
        logging.info("Skipping %s" % species_name)
        continue

    if species_name in black_list:
        logging.info("Skipping %s" % species_name)
        continue

    logging.info("Starting %s" % species_name)
    res = process_one_species(species_name, cutoff_dict[species_name][0], BLOCK_SIZE, debug=DEBUG)
    logging.info("Finished %s" % species_name)
    if res is not None:
        df, data = res
        first_pass_save_path = os.path.join(config.analysis_directory,
                                 "closely_related", "first_pass", "{}.pickle".format(species_name))
        df.to_pickle(first_pass_save_path)
        # data has a mixture of np array and lists and dict... save to pickle for simplicity
        pickle.dump(data, open(second_path_save_path, 'wb'))
    if DEBUG:
        break
