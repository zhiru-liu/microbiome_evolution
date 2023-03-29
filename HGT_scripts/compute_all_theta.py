import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt

from utils import typical_pair_utils
import config

Tc_cutoffs = json.load(open(os.path.join(config.analysis_directory, 'misc', 'Tc_cutoffs.json'), 'r'))

theta_dict = {}
for species_name in os.listdir(os.path.join(config.data_directory, 'zarr_snps')):
    if species_name.startswith('.'):
        continue
    single_sub_idxs = typical_pair_utils.load_single_subject_sample_idxs(species_name)
    theta = typical_pair_utils._compute_theta(species_name, single_sub_idxs, clade_cutoff=Tc_cutoffs.get(species_name, [None, None]))
    print(species_name, theta)
    theta_dict[species_name] = theta
json.dump(theta_dict, open(os.path.join(config.analysis_directory, 'misc', 'all_thetas.json'), 'w'))