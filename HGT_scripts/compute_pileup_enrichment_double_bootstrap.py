import os
import numpy as np
import random
from utils import pileup_utils
import config

"""
Performing double bootstrap analysis for E rectale within-host vs between-host enrichment
"""
num_reps = 100

save_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'enrichment', 'E_rectale')
within_cached_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'cached', 'E_rectale_within_json')
between_cached_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'cached', 'E_rectale_between_json')
base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Eubacterium_rectale_56927')
genome_len = 230159

threshold_choice = 0
between_data = np.loadtxt(os.path.join(base_path, 'between_host.csv'))
within_data = np.loadtxt(os.path.join(base_path, 'within_host.csv'))
real_diff = within_data[:, threshold_choice] - between_data[:, threshold_choice]

# preparing the intermediate cached files
within_files = [within_cached_path + '/' + x for x in os.listdir(within_cached_path)]
between_files = [between_cached_path + '/' + x for x in os.listdir(between_cached_path)]
all_files = between_files + within_files

# permuting within/between labels
fake_diffs = np.empty(shape=(num_reps + 1, genome_len))
between_sample_size = 3000
for i in xrange(num_reps + 1):
    random.shuffle(all_files)
    wt_cumu_runs = pileup_utils.compute_pileup_from_cache(all_files[:len(within_files)], genome_len, allowed_threshold=threshold_choice)
    bt_cumu_runs = pileup_utils.compute_pileup_from_cache(all_files[len(within_files):len(within_files) + between_sample_size], genome_len, allowed_threshold=threshold_choice)
    fake_diffs[i, :] = wt_cumu_runs - bt_cumu_runs  # save the enrichment

real_num_great = np.sum(fake_diffs[1:, :] >= real_diff[None, :], axis=0)  # first level p-val: "q-val"
test_num_great = np.sum(fake_diffs[1:, :] >= fake_diffs[0, :], axis=0)  # using the shuffled as a sanity check

# second level bootstraping: iterate over all the permutations and compute q-val for each
# the following two will be used to compute the actual p-val after second bootstrap level
real_num_passed = np.zeros(genome_len)
test_num_passed = np.zeros(genome_len)
significant_regions = []
significant_regions.append(np.sum(real_num_great < (num_reps * 5e-2)))  # record the number of sites that appear significant in q-val
significant_regions.append(np.sum(test_num_great < (num_reps * 5e-2)))  # record the number of sites that appear significant in q-val

for i in xrange(1, num_reps+1):
    mask = np.ones(num_reps+1).astype(bool)
    mask[0] = False  # the zeroth is the test rep
    mask[i] = False
    fake_num_great = np.sum(fake_diffs[mask, :] >= fake_diffs[i, :], axis=0)
#     fake_q = np.convolve(num_great, np.ones(500)/500., mode='same') / (num_reps-1)
    significant_regions.append(np.sum(fake_num_great < (num_reps * 5e-2)))  # record the number of sites that appear significant in q-val
    real_num_passed += fake_num_great <= real_num_great  # count reps that are more significant than real data
    test_num_passed += fake_num_great <= test_num_great  # count reps that are more significant than test rep

np.savetxt(os.path.join(save_path, 'real_num_passed.txt'), real_num_passed)
np.savetxt(os.path.join(save_path, 'test_num_passed.txt'), test_num_passed)
np.savetxt(os.path.join(save_path, 'sig_regions_dist.txt'), significant_regions)
