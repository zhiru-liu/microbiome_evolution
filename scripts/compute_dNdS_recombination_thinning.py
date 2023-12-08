import os
import numpy as np
import pandas as pd
import pickle
import sys

import config
from utils import snp_data_utils, close_pair_utils, pileup_utils, BSMC_utils, typical_pair_utils, core_gene_utils

second_pass_dir = os.path.join(config.analysis_directory, 'closely_related', 'iter_second_third_passes',
                               'converged_pass')
def calculate_spacing(snp_vec):
    if snp_vec.sum > 1:
        syn_mut_locs = np.where(snp_vec)[0]
        snp_spacings = syn_mut_locs[1:] - syn_mut_locs[:-1]
        median_spacing = np.median(snp_spacings)
        mean_spacing = np.mean(snp_spacings)
    else:
        median_spacing, mean_spacing = -1
    return median_spacing, mean_spacing

# species = []
# for species_name in species:
for filename in os.listdir(second_pass_dir):
    if filename.startswith('.'):
        continue
    species_name = filename.split('.')[0]
    transfer_df_path = os.path.join(config.analysis_directory, "closely_related", 'third_pass',
                     species_name + '_all_transfers.pickle')
    third_pass_df_path = os.path.join(config.analysis_directory, "closely_related", 'third_pass',
                                      species_name + '.pickle')

    df = pd.read_pickle(transfer_df_path)
    third_df = pd.read_pickle(third_pass_df_path)
    dh = snp_data_utils.DataHoarder(species_name, mode='QP', allowed_variants=['1D', '2D', '3D', '4D'])

    good_idxs = dh.get_single_subject_idxs()
    recomb_pairs = list(df.pairs.unique())
    close_pairs = list(third_df.pairs.unique())

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
        core_len = len(snp_vec)

        core_len_4D = len(snp_vec[mask_4D])
        core_len_1D = len(snp_vec[mask_1D])
        core_diff_4D = np.sum(snp_vec[mask_4D])
        core_diff_1D = np.sum(snp_vec[mask_1D])

        recomb_len_4D = len(snp_vec[mask_4D & recomb_mask])
        recomb_diff_4D = np.sum(snp_vec[mask_4D & recomb_mask])
        recomb_len_1D = len(snp_vec[mask_1D & recomb_mask])
        recomb_diff_1D = np.sum(snp_vec[mask_1D & recomb_mask])

        clonal_len_4D = len(snp_vec[mask_4D & clonal_mask])
        clonal_diff_4D = np.sum(snp_vec[mask_4D & clonal_mask])
        clonal_len_1D = len(snp_vec[mask_1D & clonal_mask])
        clonal_diff_1D = np.sum(snp_vec[mask_1D & clonal_mask])

        median_spacing_core, mean_spacing_core = calculate_spacing(snp_vec[mask_4D])
        median_spacing_recomb, mean_spacing_recomb = calculate_spacing(snp_vec[mask_4D & recomb_mask])
        median_spacing_clonal, mean_spacing_clonal = calculate_spacing(snp_vec[mask_4D & clonal_mask])

        dat.append((species_name, pair, core_div, core_len,
                    core_len_4D, core_len_1D, core_diff_4D, core_diff_1D,
                    recomb_len_4D, recomb_len_1D, recomb_diff_4D, recomb_diff_1D,
                    clonal_len_4D, clonal_len_1D, clonal_diff_4D, clonal_diff_1D,
                    median_spacing_core, mean_spacing_core,
                    median_spacing_recomb, mean_spacing_recomb,
                    median_spacing_clonal, mean_spacing_clonal))
        count += 1

    for pair in close_pairs:
        # processing pairs with only clonal regions
        if pair in recomb_pairs:
            continue
        snp_vec, covered = dh.get_snp_vector(pair)
        covered_variants = dh.variants[dh.general_mask][covered]
        mask_4D = covered_variants == '4D'
        mask_1D = covered_variants == '1D'

        core_div = snp_vec.mean()
        core_len = len(snp_vec)

        core_len_4D = len(snp_vec[mask_4D])
        core_len_1D = len(snp_vec[mask_1D])
        core_diff_4D = np.sum(snp_vec[mask_4D])
        core_diff_1D = np.sum(snp_vec[mask_1D])

        median_spacing, mean_spacing = calculate_spacing(snp_vec[mask_4D])  # this will ensure no nan in spacing

        dat.append((species_name, pair, core_div, core_len,
                    core_len_4D, core_len_1D, core_diff_4D, core_diff_1D,
                    0, 0, 0, 0,
                    core_len_4D, core_len_1D, core_diff_4D, core_diff_1D,
                    median_spacing, mean_spacing,
                    -1, -1,
                    median_spacing, mean_spacing))

    dnds_df = pd.DataFrame(dat)
    dnds_df.columns = ['species_name', 'pair', 'core_div', 'core_len',
                    'core_len_4D', 'core_len_1D', 'core_diff_4D', 'core_diff_1D',
                    'recomb_len_4D', 'recomb_len_1D', 'recomb_diff_4D', 'recomb_diff_1D',
                    'clonal_len_4D', 'clonal_len_1D', 'clonal_diff_4D', 'clonal_diff_1D',
                    'median_spacing_core', 'mean_spacing_core',
                    'median_spacing_recomb', 'mean_spacing_recomb',
                    'median_spacing_clonal', 'mean_spacing_clonal']
    dnds_df.to_pickle(os.path.join(config.analysis_directory, 'dNdS_recombination', '{}.pickle'.format(species_name)))