import numpy as np
import itertools
import random
import os
import pickle
import sys
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
sys.path.append("..")
from utils import typical_pair_utils
import config


run_data_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data')
plot_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs')
for filename in os.listdir(os.path.join(run_data_dir, 'between_hosts')):
    if filename.startswith('.'):
        continue
    species_name = filename.split('.')[0]

    save_path = os.path.join(run_data_dir, 'within_hosts', '{}.pickle'.format(species_name))
    within_runs_data = pickle.load(open(save_path, 'rb'))
    save_path = os.path.join(run_data_dir, 'between_hosts', '{}.pickle'.format(species_name))
    between_runs_data = pickle.load(open(save_path, 'rb'))

    within_host_max_runs = typical_pair_utils.compute_max_runs(within_runs_data)
    between_host_max_runs = typical_pair_utils.compute_max_runs(between_runs_data)
    ks_dist, p_val = ks_2samp(within_host_max_runs, between_host_max_runs)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.hist([between_host_max_runs, within_host_max_runs], bins=100, density=True,
            cumulative=-1, histtype='step', label=['Between host', 'Within host'])
    ax.set_xlabel('Max homozygous run length (4D syn sites)')
    ax.set_ylabel('Fraction longer than')
    ax.legend()
    ax.set_title("p={:.1e}".format(p_val))

    fig.savefig(os.path.join(plot_dir, '{}.pdf'.format(species_name)), bbox_inches='tight')
