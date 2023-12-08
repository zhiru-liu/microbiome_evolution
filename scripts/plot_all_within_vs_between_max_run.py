import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
sys.path.append("..")
import config


max_run_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs')
plot_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs_plots')
processed = set()
for filename in os.listdir(os.path.join(max_run_dir)):
    if filename.startswith('.'):
        continue
    species_name = '_'.join(filename.split('_')[:-1])
    if species_name in processed:
        continue
    else:
        processed.add(species_name)

    within_host_max_runs = np.loadtxt(os.path.join(max_run_dir, species_name + '_within.txt'), ndmin=1)
    between_host_max_runs = np.loadtxt(os.path.join(max_run_dir, species_name + '_between.txt'))
    # in order to use one sided ks test, need to use python3's scipy
    if len(within_host_max_runs) == 0:
        print("Skipping {}: no within host data".format(species_name))
        continue
    print("Species: {}; Samples: {}".format(species_name, len(within_host_max_runs)))
    ks_dist, p_val = ks_2samp(within_host_max_runs, between_host_max_runs, alternative='less')

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.hist([between_host_max_runs, within_host_max_runs], bins=100, density=True,
            cumulative=-1, histtype='step', label=['Between host', 'Within host'])
    ax.set_xlabel('Max homozygous run length (4D syn sites)')
    ax.set_ylabel('Fraction longer than')
    ax.legend()
    ax.set_title("$n_w={}, p={:.1e}$".format(len(within_host_max_runs), p_val))

    fig.savefig(os.path.join(plot_dir, '{}.pdf'.format(species_name)), bbox_inches='tight')
    plt.close()
