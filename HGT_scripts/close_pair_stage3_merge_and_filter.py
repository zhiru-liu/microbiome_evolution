import sys
import os
import pickle
import numpy as np
import pandas as pd
from itertools import compress
sys.path.append("..")
import config
from utils import close_pair_utils, parallel_utils


# ckpt_path = os.path.join(config.analysis_directory,
#                          "closely_related", "second_pass")
ckpt_path = os.path.join(config.analysis_directory, 'closely_related', 'iter_second_third_passes', 'converged_pass')
for filename in os.listdir(ckpt_path):
    if filename.startswith('.'):
        continue
    species_name = filename.split('.')[0]

    if ('vulgatus'in species_name) or ('shahii' in species_name):
        two_clades = True
    else:
        two_clades = False

    print("Processing {}".format(species_name))
    first_pass_save_path = os.path.join(config.analysis_directory,
                                        "closely_related", "first_pass", "{}.pickle".format(species_name))
    first_pass_df = pd.read_pickle(first_pass_save_path)

    third_pass_path = os.path.join(config.analysis_directory, "closely_related", 'third_pass', species_name + '.pickle')
    if os.path.exists(third_pass_path):
        print("{} already processed, skipping".format(species_name))
        continue

    if two_clades:
        # second_path_save_path = os.path.join(config.analysis_directory,
        #                                      "closely_related", "two_clades", "{}_two_clades.pickle".format(species_name))
        second_path_save_path = os.path.join(ckpt_path, filename)
        data = pickle.load(open(second_path_save_path, 'rb'))
        within_counts, between_counts, all_transfer_df = close_pair_utils.merge_and_filter_transfers(data, separate_clade=True, merge_threshold=0, filter_threshold=5)
        transfer_counts = within_counts + between_counts
    else:
        data = pickle.load(open(os.path.join(ckpt_path, filename), 'rb'))
        transfer_counts, all_transfer_df = close_pair_utils.merge_and_filter_transfers(data, separate_clade=False, merge_threshold=0, filter_threshold=5)
    if transfer_counts is None:
        print("{} has no data".format(species_name))
        continue
    pair_to_total_length = all_transfer_df.groupby('pairs')['lengths'].sum().to_dict()
    full_lengths = np.array([pair_to_total_length.get(x, 0) for x in data['pairs']]) * config.second_pass_block_size

    third_pass_df = pd.DataFrame()
    third_pass_df['pairs'] = data['pairs']
    # third_pass_df['clonal snps'] = data['clonal snps']
    # third_pass_df['transfer snps'] = data['transfer snps']
    third_pass_df['genome lengths'] = data['genome lengths'] # length of snp vector - taking missing data into account
    third_pass_df['clonal lengths'] = data['clonal lengths']
    # third_pass_df['clonal divs'] = third_pass_df['clonal snps'] / third_pass_df['clonal lengths'].astype(float)
    clonal_divs = np.array(data['clonal divs'])
    third_pass_df['clonal divs'] = clonal_divs[:, 1]
    third_pass_df['naive clonal divs'] = clonal_divs[:, 0]
    third_pass_df['expected clonal snps'] = third_pass_df['clonal divs'] * third_pass_df['genome lengths']
    third_pass_df['transfer counts'] = transfer_counts
    core_genome_len = parallel_utils.get_genome_length(species_name)
    third_pass_df['normalized transfer counts'] = transfer_counts * 1e6 / core_genome_len # simple normalization by total core genome len
    third_pass_df['total transfer lengths'] = full_lengths

    # total_blocks = first_pass_df[first_pass_df['pair_idxs'].isin(data['pairs'])]['num_total_blocks']
    # third_pass_df['genome lengths'] = total_blocks.to_numpy() * config.first_pass_block_size
    third_pass_df['clonal fractions'] = 1 - third_pass_df['total transfer lengths'] / \
                                       third_pass_df['genome lengths'].astype(float)
    # one more round of clonal fraction cutoff based on detected transfers
    # third_pass_df = third_pass_df[third_pass_df['clonal fractions'] >= config.clonal_fraction_cutoff]
    # filter pairs later according to cf cutoff
    print("Before additional clonal fraction filter, {} pairs".format(len(data['pairs'])))
    print("After filter by {} clonal fraction, {} pairs".format(config.clonal_fraction_cutoff,
                                                                np.sum(third_pass_df['clonal fractions']>=config.clonal_fraction_cutoff)))

    # save which transfers are in close pairs
    mask = third_pass_df['clonal fractions'] > config.clonal_fraction_cutoff
    cf_dict = dict(zip(third_pass_df['pairs'], third_pass_df['clonal fractions']))
    cd_dict = dict(zip(third_pass_df['pairs'], third_pass_df['clonal divs']))
    all_transfer_df['clonal divergence'] = all_transfer_df['pairs'].map(cd_dict)
    all_transfer_df['clonal fraction'] = all_transfer_df['pairs'].map(cf_dict)

    # passed_pairs = list(compress(data['pairs'], mask))  # compress is concatenating lists
    # all_transfer_df['clonal fraction >75%'] = all_transfer_df['pairs'].isin(passed_pairs)

    third_pass_df.to_pickle(third_pass_path)
    all_transfer_df.to_pickle(os.path.join(config.analysis_directory, "closely_related", 'third_pass', species_name + '_all_transfers.pickle'))
