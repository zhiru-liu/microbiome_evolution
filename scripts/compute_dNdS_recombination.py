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

    half_genome_mask = np.ones(dh.general_mask.sum()).astype(bool)
    half_genome_mask[:len(half_genome_mask) // 2] = False

    count = 0
    dat = []
    for pair, grouped in df.groupby('pairs'):
        snp_vec, covered = dh.get_snp_vector(pair)
        covered_variants = dh.variants[dh.general_mask][covered]

        covered_locs = -np.ones(covered.shape)
        covered_locs[covered] = np.arange(len(snp_vec))
        covered_locs = covered_locs.astype(int)

        half_mask = half_genome_mask[covered]
        second_half_mask = ~half_mask

        recomb_mask = np.zeros(snp_vec.shape).astype(bool)
        for idx, row in grouped.iterrows():
            start, end = covered_locs[row['core genome starts']], covered_locs[row['core genome ends']]
            recomb_mask[start:end + 1] = True
        clonal_mask = ~recomb_mask

        mask_4D = covered_variants == '4D'
        mask_1D = covered_variants == '1D'
        core_div = snp_vec.mean()

        core_div_4D = snp_vec[mask_4D].mean()
        core_div_4D_1 = snp_vec[mask_4D & half_mask].mean()
        core_div_4D_2 = snp_vec[mask_4D & second_half_mask].mean()

        core_div_1D = snp_vec[mask_1D].mean()
        core_div_1D_1 = snp_vec[mask_1D & half_mask].mean()
        core_div_1D_2 = snp_vec[mask_1D & second_half_mask].mean()

        recomb_div_4D = snp_vec[mask_4D & recomb_mask].mean()
        recomb_div_4D_1 = snp_vec[mask_4D & recomb_mask & half_mask].mean()
        recomb_div_4D_2 = snp_vec[mask_4D & recomb_mask & second_half_mask].mean()

        recomb_div_1D = snp_vec[mask_1D & recomb_mask].mean()
        recomb_div_1D_1 = snp_vec[mask_1D & recomb_mask & half_mask].mean()
        recomb_div_1D_2 = snp_vec[mask_1D & recomb_mask & second_half_mask].mean()

        clonal_div_4D = snp_vec[mask_4D & clonal_mask].mean()
        clonal_div_4D_1 = snp_vec[mask_4D & clonal_mask & half_mask].mean()
        clonal_div_4D_2 = snp_vec[mask_4D & clonal_mask & second_half_mask].mean()

        clonal_div_1D = snp_vec[mask_1D & clonal_mask].mean()
        clonal_div_1D_1 = snp_vec[mask_1D & clonal_mask & half_mask].mean()
        clonal_div_1D_2 = snp_vec[mask_1D & clonal_mask & second_half_mask].mean()

        median_spacing_core, mean_spacing_core = calculate_spacing(snp_vec[mask_4D])
        median_spacing_recomb, mean_spacing_recomb = calculate_spacing(snp_vec[mask_4D & recomb_mask])
        median_spacing_clonal, mean_spacing_clonal = calculate_spacing(snp_vec[mask_4D & clonal_mask])

        dat.append((species_name, pair, core_div,
                    core_div_4D, core_div_4D_1, core_div_4D_2,
                    core_div_1D, core_div_1D_1, core_div_1D_2,
                    recomb_div_4D, recomb_div_4D_1, recomb_div_4D_2,
                    recomb_div_1D, recomb_div_1D_1, recomb_div_1D_2,
                    clonal_div_4D, clonal_div_4D_1, clonal_div_4D_2,
                    clonal_div_1D, clonal_div_1D_1, clonal_div_1D_2,
                    median_spacing_core, mean_spacing_core,
                    median_spacing_recomb, mean_spacing_recomb,
                    median_spacing_clonal, mean_spacing_clonal))
        # dat.append((species_name, pair, core_div, core_div_4D, core_div_1D, recomb_div_4D, recomb_div_1D, clonal_div_4D,
        #             clonal_div_1D))
        count += 1

    dnds_df = pd.DataFrame(dat)
    # dnds_df.columns = ['speices_name', 'pair', 'core_div', 'core_div_4D', 'core_div_1D', 'recomb_div_4D',
    #                    'recomb_div_1D', 'clonal_div_4D', 'clonal_div_1D']
    dnds_df.columns = ['speices_name', 'pair', 'core_div',
                       'core_div_4D', 'core_div_4D_1', 'core_div_4D_2',
                       'core_div_1D', 'core_div_1D_1', 'core_div_1D_2',
                       'recomb_div_4D', 'recomb_div_4D_1', 'recomb_div_4D_2',
                       'recomb_div_1D', 'recomb_div_1D_1', 'recomb_div_1D_2',
                       'clonal_div_4D', 'clonal_div_4D_1', 'clonal_div_4D_2',
                       'clonal_div_1D', 'clonal_div_1D_1', 'clonal_div_1D_2',
                       'median_spacing_core', 'mean_spacing_core',
                       'median_spacing_recomb', 'mean_spacing_recomb',
                       'median_spacing_clonal', 'mean_spacing_clonal']
    dnds_df.to_pickle(os.path.join(config.analysis_directory, 'dNdS_recombination', '{}.pickle'.format(species_name)))

    clonal_dat = []
    for pair in close_pairs:
        if pair in recomb_pairs:
            continue
        snp_vec, covered = dh.get_snp_vector(pair)
        covered_variants = dh.variants[dh.general_mask][covered]
        half_mask = half_genome_mask[covered]
        second_half_mask = ~half_mask
        mask_4D = covered_variants == '4D'
        mask_1D = covered_variants == '1D'
        core_div = snp_vec.mean()
        core_div_4D = snp_vec[mask_4D].mean()
        core_div_4D_1 = snp_vec[mask_4D & half_mask].mean()
        core_div_4D_2 = snp_vec[mask_4D & second_half_mask].mean()

        core_div_1D = snp_vec[mask_1D].mean()
        core_div_1D_1 = snp_vec[mask_1D & half_mask].mean()
        core_div_1D_2 = snp_vec[mask_1D & second_half_mask].mean()

        if core_div_4D != 0:
            syn_mut_locs = np.where(snp_vec[mask_4D])[0]
            snp_spacings = syn_mut_locs[1:] - syn_mut_locs[:-1]
            median_spacing = np.median(snp_spacings)
            mean_spacing = np.mean(snp_spacings)
        else:
            median_spacing = -1
            mean_spacing = -1

        clonal_dat.append((species_name, pair, core_div, core_div_4D, core_div_1D, core_div_4D_1, core_div_1D_1,
                           core_div_4D_2, core_div_1D_2, median_spacing, mean_spacing))
    if len(clonal_dat) == 0:
        continue
    clonal_df = pd.DataFrame(clonal_dat)
    clonal_df.columns = ['speices_name', 'pair', 'core_div', 'core_div_4D', 'core_div_1D', 'core_div_4D_1',
                         'core_div_1D_1', 'core_div_4D_2', 'core_div_1D_2', 'median_spacing', 'mean_spacing']
    clonal_df.to_pickle(os.path.join(config.analysis_directory, 'dNdS_recombination', '{}_clonal.pickle'.format(species_name)))
