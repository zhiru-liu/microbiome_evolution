import random
import numpy as np
import os
import config
from utils import pileup_utils

############# setting up the parameters #############
save_path = os.path.join(config.plotting_intermediate_directory, 'B_vulgatus_within_clade_pileup_shuffle.txt')
# E rectale
# within_cached_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'cached', 'E_rectale_within_json')
# between_cached_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'cached', 'E_rectale_between_json')
# base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Eubacterium_rectale_56927')
# genome_len = 230159

# B vulgatus
within_cached_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'cached', 'B_vulgatus_within_host_within_clade')
between_cached_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'cached', 'B_vulgatus_between_host_within_clade')
base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')

# within_cached_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'cached', 'B_vulgatus_within_host_between_clade')
# between_cached_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'cached', 'B_vulgatus_between_host_between_clade')
# base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')

genome_len = 288079

########## loading the true pile up data ###########
threshold_choice = 0
between_data = np.loadtxt(os.path.join(base_path, 'between_host.csv'))
within_data = np.loadtxt(os.path.join(base_path, 'within_host.csv'))
real_diff = within_data[:, threshold_choice] - between_data[:, threshold_choice]

# preparing the intermediate cached files
within_files = [within_cached_path + '/' + x for x in os.listdir(within_cached_path)]
between_files = [between_cached_path + '/' + x for x in os.listdir(between_cached_path)]
all_files = between_files + within_files

# preparing permutation test
num_great = np.zeros(genome_len)
num_less = np.zeros(genome_len)

num_reps = 10000
between_sample_size = 3000

for i in xrange(num_reps):
    random.shuffle(all_files)
    wt_cumu_runs = pileup_utils.compute_pileup_from_cache(all_files[:len(within_files)], genome_len, allowed_threshold=threshold_choice)
    bt_cumu_runs = pileup_utils.compute_pileup_from_cache(all_files[len(within_files):len(within_files) + between_sample_size], genome_len, allowed_threshold=threshold_choice)
    num_great += (wt_cumu_runs - bt_cumu_runs) > real_diff
    num_less += (wt_cumu_runs - bt_cumu_runs) < real_diff

dat = np.stack([num_great, num_less])
np.savetxt(save_path, dat)
