import os
import numpy as np
import pandas as pd
import pickle
import sys

import config
from utils import parallel_utils, close_pair_utils, pileup_utils, BSMC_utils, typical_pair_utils, core_gene_utils

second_pass_dir = os.path.join(config.analysis_directory, 'closely_related', 'iter_second_third_passes',
                               'converged_pass')

# species = []
# for species_name in species:
for filename in os.listdir(second_pass_dir):
    if filename.startswith('.'):
        continue
    species_name = filename.split('.')[0]
    transfer_df_path = os.path.join(config.analysis_directory, "closely_related", 'third_pass',
                     species_name + '_all_transfers.pickle')

    df = pd.read_pickle(transfer_df_path)
    dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['1D', '2D', '3D', '4D'])

    count = 0
    dat = []
    for pair, grouped in df.groupby('pairs'):
        snp_vec, covered = dh.get_snp_vector(pair)
        covered_variants = dh.variants[dh.general_mask][covered]

        covered_locs = -np.ones(covered.shape)
        covered_locs[covered] = np.arange(len(snp_vec))
        covered_locs = covered_locs.astype(int)

        recomb_mask = np.zeros(snp_vec.shape).astype(bool)
        for idx, row in grouped.iterrows():
            start, end = covered_locs[row['core genome starts']], covered_locs[row['core genome ends']]
            recomb_mask[start:end + 1] = True
        clonal_mask = ~recomb_mask

        mask_4D = covered_variants == '4D'
        mask_1D = covered_variants == '1D'
        core_div = snp_vec.mean()
        core_div_4D = snp_vec[mask_4D].mean()
        core_div_1D = snp_vec[mask_1D].mean()
        recomb_div_4D = snp_vec[mask_4D & recomb_mask].mean()
        recomb_div_1D = snp_vec[mask_1D & recomb_mask].mean()
        clonal_div_4D = snp_vec[mask_4D & clonal_mask].mean()
        clonal_div_1D = snp_vec[mask_1D & clonal_mask].mean()

        dat.append((species_name, pair, core_div, core_div_4D, core_div_1D, recomb_div_4D, recomb_div_1D, clonal_div_4D,
                    clonal_div_1D))
        count += 1

    dnds_df = pd.DataFrame(dat)
    dnds_df.columns = ['speices_name', 'pair', 'core_div', 'core_div_4D', 'core_div_1D', 'recomb_div_4D',
                       'recomb_div_1D', 'clonal_div_4D', 'clonal_div_1D']
    dnds_df.to_pickle(os.path.join(config.analysis_directory, 'dNdS_recombination', '{}.pickle'.format(species_name)))