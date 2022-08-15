import os
import json
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import config
from utils import parallel_utils
from plotting_for_publication import plot_pileup_mirror

fig, axes = plt.subplots(2, 1, figsize=(6.5, 3), dpi=600)
fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'


# load double bootstrap data
n_reps = 5000.
save_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'enrichment', 'E_rectale_double_bootstrap')
real_ps = np.load(os.path.join(save_path, 'real_num_great.npy')) / n_reps
test_ps = np.load(os.path.join(save_path, 'control_num_great.npy')) / n_reps
# significant_regions = np.loadtxt(os.path.join(save_path, 'sig_regions_dist.txt'))
min_pvals = np.load(os.path.join(save_path, 'min_pval.npy')) / n_reps

# plot pileup data
species_name = 'Eubacterium_rectale_56927'
mirror_ax = axes[0]
base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
between_host_path = os.path.join(base_path, 'between_host.csv')
thresholds = np.loadtxt(os.path.join(base_path, 'between_host_thresholds.txt'))

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
within_host_path = os.path.join(base_path, 'within_host.csv')
within_thresholds = np.loadtxt(os.path.join(base_path, 'within_host_thresholds.txt'))

between_cumu_runs, within_cumu_runs = plot_pileup_mirror.load_data_and_plot_mirror(
    between_host_path, within_host_path, mirror_ax, ind_to_plot=0, ylim=0.5, colors=[config.between_host_color, config.within_host_color])

pval_ax = axes[1]
window_size = 1000.
average_p = np.convolve(real_ps, np.ones(int(window_size))/window_size, mode='same')
average_test_p = np.convolve(test_ps, np.ones(int(window_size))/window_size, mode='same')

pval_ax.plot(average_p, label='real data')
pval_ax.plot(average_test_p, color='grey', label='control')
pval_ax.set_xlim(mirror_ax.get_xlim())
pval_ax.set_ylabel('$p$ vals\n(uncorrected)')
pval_ax.set_yscale('log')

p_val_threshold = np.percentile(min_pvals, 5)
axes[1].axhline(p_val_threshold, linestyle='--', color='pink', label='significance threshold')
bool_vec = average_p < p_val_threshold
runs, starts, ends= parallel_utils._compute_runs_single_chromosome(~bool_vec, return_locs=True)
for start, end in zip(starts, ends):
    mirror_ax.axvspan(start, end, color='red', alpha=0.2, linewidth=1, zorder=3)
    pval_ax.axvspan(start, end, color='red', alpha=0.2, linewidth=1, zorder=3)

pval_ax.legend()
axes[0].set_xticklabels([])
axes[0].set_xlabel('')
axes[1].set_xlabel("Genome locations (synonymous sites on core genome)")

fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_pileup_double_bootstrap.pdf'), bbox_inches='tight')
