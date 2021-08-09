import sys
import os
import pickle
import numpy as np
import pandas as pd
sys.path.append("..")
import config
from utils import close_pair_utils


ckpt_path = os.path.join(config.analysis_directory,
                         "closely_related", "second_pass")
for filename in os.listdir(ckpt_path):
    if filename.startswith('.'):
        continue
    species_name = filename.split('.')[0]

    print("Processing {}".format(species_name))
    first_pass_save_path = os.path.join(config.analysis_directory,
                                        "closely_related", "first_pass", "{}.pickle".format(species_name))
    first_pass_df = pd.read_pickle(first_pass_save_path)

    data = pickle.load(open(os.path.join(ckpt_path, filename), 'rb'))
    transfer_counts, all_transfer_df = close_pair_utils.merge_and_filter_transfers(data, separate_clade=False)
    if transfer_counts is None:
        print("{} has no data".format(species_name))
        continue
    pair_to_total_length = all_transfer_df.groupby('pairs')['lengths'].sum().to_dict()
    full_lengths = np.array([pair_to_total_length.get(x, 0) for x in data['pairs']]) * config.second_pass_block_size

    third_pass_df = pd.DataFrame()
    third_pass_df['pairs'] = data['pairs']
    third_pass_df['clonal divs'] = data['clonal divs']
    third_pass_df['transfer snps'] = data['transfer snps']
    third_pass_df['genome lengths'] = data['genome lengths']
    third_pass_df['clonal snps'] = third_pass_df['clonal divs'] * third_pass_df['genome lengths']
    third_pass_df['transfer counts'] = transfer_counts
    third_pass_df['total transfer lengths'] = full_lengths

    # total_blocks = first_pass_df[first_pass_df['pair_idxs'].isin(data['pairs'])]['num_total_blocks']
    # third_pass_df['genome lengths'] = total_blocks.to_numpy() * config.first_pass_block_size
    third_pass_df['clonal fractions'] = 1 - third_pass_df['total transfer lengths'] / \
                                       third_pass_df['genome lengths'].astype(float)
    # one more round of clonal fraction cutoff based on detected transfers
    third_pass_df = third_pass_df[third_pass_df['clonal fractions'] >= config.clonal_fraction_cutoff]
    print("Before additional clonal fraction filter, {} pairs".format(len(data['pairs'])))
    print("After filter, {} pairs".format(third_pass_df.shape[0]))

    third_pass_df.to_pickle(os.path.join(config.analysis_directory, "closely_related", 'third_pass', species_name + '.pickle'))
    all_transfer_df.to_pickle(os.path.join(config.analysis_directory, "closely_related", 'third_pass', species_name + '_all_transfers.pickle'))
