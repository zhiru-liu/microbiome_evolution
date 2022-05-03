import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
sys.path.append("..")
import config
from utils import typical_pair_utils

def process_species(species_name, len_threshold=None):
    save_path = os.path.join(run_data_dir, 'within_hosts', '{}.pickle'.format(species_name))
    try:
        within_runs_data = pickle.load(open(save_path, 'rb'))
    except IOError:
        print("%s within-host has not been processed yet" % species_name)
        return
    if len(within_runs_data)==0:
        print("{} has no within samples".format(species_name))
        return
    save_path = os.path.join(run_data_dir, 'between_hosts', '{}.pickle'.format(species_name))
    between_runs_data = pickle.load(open(save_path, 'rb'))

    if len_threshold is None:
        if 'vulgatus' in species_name:
            same_clade, diff_clade = typical_pair_utils.compute_theta('Bacteroides_vulgatus_57955', return_both=True)
            if 'same_clade' in species_name:
                theta = same_clade
            elif 'diff_clade' in species_name:
                theta = diff_clade
            else:
                theta = same_clade
        else:
            theta = typical_pair_utils.compute_theta(species_name)
        len_threshold = 50 / theta
        print(species_name, len_threshold)

    # within_long_runs_dict = {k:np.sum(within_runs_data[k]>len_threshold) for k in within_runs_data}
    # between_long_runs_dict = {k:np.sum(between_runs_data[k]>len_threshold) for k in between_runs_data}
    within_long_runs_dict = {k:np.sum(within_runs_data[k][within_runs_data[k]>len_threshold]) for k in within_runs_data}
    between_long_runs_dict = {k:np.sum(between_runs_data[k][between_runs_data[k]>len_threshold]) for k in between_runs_data}

    between_long_runs = between_long_runs_dict.values()
    within_long_runs = within_long_runs_dict.values()

    ks_dist, p_val = ks_2samp(within_long_runs, between_long_runs)

    bins = np.arange(max(max(between_long_runs), max(within_long_runs)))
    fig, ax = plt.subplots()
    _ = ax.hist(within_long_runs, cumulative=-1, density=True, bins=bins, histtype='step')
    _ = ax.hist(between_long_runs, cumulative=-1, density=True, bins=bins, histtype='step')
    ax.set_title("length threshold: {0}, p={1:.2e}".format(int(len_threshold), p_val))
    fig.savefig(os.path.join(config.analysis_directory, 'typical_pairs', 'long_run_sums', '{}.pdf'.format(species_name)))
    plt.close()


run_data_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data')
for filename in os.listdir(os.path.join(run_data_dir, 'between_hosts')):
    if filename.startswith('.'):
        continue
    species_name = filename.split('.')[0]
    process_species(species_name)

# process_species('Eubacterium_rectale_56927', len_threshold=1160)