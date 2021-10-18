import os
import pickle
import sys
import numpy as np
sys.path.append("..")
import config
from utils import typical_pair_utils


run_data_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data')
max_run_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs')
for filename in os.listdir(os.path.join(run_data_dir, 'between_hosts')):
    if filename.startswith('.'):
        continue
    species_name = filename.split('.')[0]

    save_path = os.path.join(run_data_dir, 'within_hosts', '{}.pickle'.format(species_name))
    try:
        within_runs_data = pickle.load(open(save_path, 'rb'))
    except IOError:
        print("%s within-host has not been processed yet" % species_name)
        continue
    save_path = os.path.join(run_data_dir, 'between_hosts', '{}.pickle'.format(species_name))
    between_runs_data = pickle.load(open(save_path, 'rb'))

    within_host_max_runs = typical_pair_utils.compute_max_runs(within_runs_data)
    between_host_max_runs = typical_pair_utils.compute_max_runs(between_runs_data)
    np.savetxt(os.path.join(max_run_dir, species_name + '_within.txt'), within_host_max_runs)
    np.savetxt(os.path.join(max_run_dir, species_name + '_between.txt'), between_host_max_runs)
