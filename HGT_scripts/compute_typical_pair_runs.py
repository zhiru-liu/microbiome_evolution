import numpy as np
import itertools
import random
import os
import pickle
import sys
sys.path.append("..")
from utils import close_pair_utils, parallel_utils, typical_pair_utils
import config


def process_species(species_name):
    within_dh = parallel_utils.DataHoarder(species_name, mode='within', allowed_variants=['4D'])
    between_dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])
    within_pairs = typical_pair_utils.generate_within_sample_idxs(within_dh, 0.005)
    between_pairs = typical_pair_utils.generate_between_sample_idxs(
        between_dh, 0.005, num_pairs=1000)

    within_runs_data = typical_pair_utils.compute_runs(within_dh, within_pairs)
    save_path = os.path.join(config.analysis_directory, 'runs_data', 'within_hosts', '{}.pickle'.format(species_name))
    pickle.dump(within_runs_data, open(save_path, 'wb'))

    between_runs_data = typical_pair_utils.compute_runs(between_dh, between_pairs)
    save_path = os.path.join(config.analysis_directory, 'runs_data', 'between_hosts', '{}.pickle'.format(species_name))
    pickle.dump(between_runs_data, open(save_path, 'wb'))


process_species("Eubacterium_rectale_56927")
