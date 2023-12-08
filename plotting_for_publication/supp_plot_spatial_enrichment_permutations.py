import os
import numpy as np
import matplotlib.pyplot as plt
import json

import config
import plotting_for_publication.default_fig_styles
from utils import snp_data_utils

# loading saved permutation results
dat = np.loadtxt(os.path.join(config.plotting_intermediate_directory, 'pileup_shuffle.txt'))
num_great = dat[0, :]  # number of permutations that real data is more enriched than fake data
num_reps = 10000

# smoothening the p value array
averaged_p = np.convolve(num_great, np.ones(500)/500., mode='same') / num_reps
fig, ax = plt.subplots(figsize=(6, 2))
plt.plot(averaged_p)
plt.axhline(1e-3, linestyle='--', color='tab:orange', label='$p=10^{-3}$')
# plt.plot(np.convolve(num_less, np.ones(100)/100., mode='same'))
plt.yscale('log')
plt.xlabel("Genome location")
plt.ylabel("$p$")
plt.legend()
fig.savefig(os.path.join(config.figure_directory, 'supp_E_rectale_pileup_p_values.pdf'), bbox_inches='tight')

# cache regions that are highly enriched
bool_vec = averaged_p < 1e-3
runs, starts, ends = snp_data_utils._compute_runs_single_chromosome(~bool_vec, return_locs=True)
json.dump(zip(starts, ends), open(os.path.join(config.plotting_intermediate_directory, 'E_rectale_p_values.json'), 'w'))
