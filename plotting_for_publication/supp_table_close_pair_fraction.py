import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import os
sys.path.append("..")
from utils import parallel_utils, typical_pair_utils
import config

sample_df = parallel_utils.compute_good_sample_stats()

close_pair_frac_20 = []
close_pair_frac_50 = []
close_pair_frac_80 = []
for species_name in sample_df['species_name']:
    clonal_frac_mat = typical_pair_utils.load_clonal_frac_mat(species_name)
    single_sub_idxs = typical_pair_utils.load_single_subject_sample_idxs(species_name)
    clonal_frac_mat = clonal_frac_mat[single_sub_idxs, :][:, single_sub_idxs]
    pd_mat = typical_pair_utils.load_pairwise_div_mat(species_name)
    pd_mat = pd_mat[single_sub_idxs, :][:, single_sub_idxs]

    cf_dist = clonal_frac_mat[np.triu_indices(clonal_frac_mat.shape[0], 1)]
    pd_dist = pd_mat[np.triu_indices(clonal_frac_mat.shape[0], 1)]
    close_pairs = np.sum(cf_dist > 0.2)
    # total_pairs = np.sum(pd_dist < Tc_cutoffs[species_name][1])
    total_pairs = float(len(cf_dist))
    close_pair_frac_20.append(np.sum(cf_dist > 0.2))
    close_pair_frac_50.append(np.sum(cf_dist > 0.5))
    close_pair_frac_80.append(np.sum(cf_dist > 0.8))

sample_df['num pairs >20% identical blocks'] = close_pair_frac_20
sample_df['num pairs >50% identical blocks'] = close_pair_frac_50
sample_df['num pairs >80% identical blocks'] = close_pair_frac_80
sample_df = sample_df.drop(columns=['num_high_coverage_samples', 'num_good_within_samples'])

sample_df.to_csv(os.path.join(config.figure_directory, 'supp_table', 'close_pair_fraction.csv'))
