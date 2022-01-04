import numpy as np
import os
import pandas as pd
from utils import parallel_utils, close_pair_utils
import config


def compute_div_in_transfers(dh, transfer_df):
    divs = []
    for pair in pd.unique(transfer_df['pairs']):
        snp_vec, coverage_vec = dh.get_snp_vector(pair)
        # good_chromo = dh.core_chromosomes[coverage_vec]
        good_chromo = dh.chromosomes[dh.general_mask][coverage_vec]
        contig_lengths = parallel_utils.get_contig_lengths(good_chromo)
        block_size = config.second_pass_block_size
        for _, row in transfer_df[transfer_df['pairs'] == pair].iterrows():
            # translating HMM coordinates (because of contig+block) to snp_vec coordinates
            start = close_pair_utils.block_loc_to_genome_loc(row['starts'], contig_lengths, block_size, left=True)
            end = close_pair_utils.block_loc_to_genome_loc(row['ends'], contig_lengths, block_size, left=False)
            # compute divergence of this transfer
            div = np.sum(snp_vec[int(start):int(end)]) / float(end - start)
            divs.append(div)
    transfer_df['divergences'] = divs
    return transfer_df


if __name__ == "__main__":
    second_pass_dir = os.path.join(config.analysis_directory, "closely_related", "second_pass")
    data_dir = os.path.join(config.analysis_directory, "closely_related", "third_pass")

    for filename in os.listdir(second_pass_dir):
        if filename.startswith('.'):
            continue
        species_name = filename.split('.')[0]
        print("Processing {}".format(species_name))
        filepath = os.path.join(data_dir, "%s_all_transfers.pickle" % species_name)
        if not os.path.exists(filepath):
            print("Intermediate file not found for {}, skipping".format(species_name))
            continue
        transfer_df = pd.read_pickle(filepath)
        dh = parallel_utils.DataHoarder(species_name=species_name, mode='QP')
        transfer_df = compute_div_in_transfers(dh, transfer_df)
        transfer_df.to_pickle(filepath)
