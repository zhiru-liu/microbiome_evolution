import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random
import config
from utils import figure_utils, snp_data_utils, typical_pair_utils


def generate_fake_transfer_dist(species_name):
    dh = snp_data_utils.DataHoarder(
        species_name, mode='QP', allowed_variants=['4D'])
    good_idxs = dh.get_single_subject_idxs()

    def sample_strain():
        return random.sample(good_idxs, 1)[0]

    def get_random_transfer_div(focal_pair, start_frac, end_frac):
        focal_strain = random.sample(focal_pair, 1)[0]
        donor_strain = sample_strain()
        snp_vec, _ = dh.get_snp_vector((focal_strain, donor_strain))
        genome_len = len(snp_vec)
        start = int(start_frac * genome_len)
        end = int(end_frac * genome_len)
        return np.mean(snp_vec[start:end])

    # first loading all transfer location
    save_path = os.path.join(config.analysis_directory,
                             "closely_related", "third_pass", "{}_all_transfers.pickle".format(species_name))
    full_df = pd.read_pickle(save_path)
    print("Simulating {} random transfers".format(full_df.shape[0]))
    all_sim_divs = []
    for focal_pair, grouped in full_df.groupby('pairs'):
        snp_vec, _ = dh.get_snp_vector(focal_pair)
        focal_genome_len = float(len(snp_vec))
        for _, row in grouped.iterrows():
            start = row['starts']
            end = row['ends']
            start_frac = start*config.second_pass_block_size / focal_genome_len
            end_frac = end*config.second_pass_block_size / focal_genome_len
            for i in range(5):
                # better sample the distribution
                div = get_random_transfer_div(focal_pair, start_frac, end_frac)
                all_sim_divs.append(div)
    np.savetxt(os.path.join(config.analysis_directory, 'closely_related', 'simulated_transfers', '{}.csv'.format(species_name)), np.array(all_sim_divs))

if __name__ == "__main__":
    second_pass_dir = os.path.join(config.analysis_directory, "closely_related", "second_pass")
    for filename in os.listdir(second_pass_dir):
        if filename.startswith('.'):
            continue
        species_name = filename.split('.')[0]
        print("Processing {}".format(species_name))
        generate_fake_transfer_dist(species_name)
