import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random
import os
from utils import BSMC_utils, close_pair_utils
import config

genome_len = 2.8e5
fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'
fig, ax = plt.subplots(figsize=(4, 3), dpi=600)

data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'for_pileup')
exp_df = pd.read_csv(os.path.join(data_dir, 'b_vulgatus', 'experiments.txt'), sep=' ')
exp_df = exp_df.set_index(['rbymu', 'lambda', 'rep'])

rbymus = [0.1, 0.5, 2]
lambs = [500, 1000, 2000]
sim_id = int(exp_df.loc[(rbymus[-1], lambs[-1], 1), 'sim_id'])
print(sim_id)

filename = os.path.join(data_dir, 'b_vulgatus', '%d.txt' % sim_id)

data = BSMC_utils.load_data(filename)

sample_size = data.shape[1] - 1
num_curves = 500
pairs = [random.sample(range(sample_size), 2) for i in range(num_curves)]

# plot the simulation
num_runs = []
cutoff = 1e-3
idx = 0
for pair in pairs:
    snp_locs, run_lens = BSMC_utils.compare_two_samples(pair[0], pair[1], data, genome_len)
    snp_vec = BSMC_utils.get_full_snp_vector(pair[0], pair[1], data, genome_len)
    cf = close_pair_utils.compute_clonal_fraction(snp_vec, block_size=1000)
    snp_count = len(snp_locs)
    div = snp_count / genome_len
    if cf > config.typical_clonal_fraction_cutoff:
        continue
    if idx ==0:
        color = 'tab:blue'
        alpha = 1
        zorder = 2
        label = 'ex 1'
    elif idx==1:
        color = 'xkcd:sky blue'
        alpha = 1
        zorder = 2
        label = 'ex 2'
    else:
        color = 'tab:grey'
        alpha = 0.1
        zorder = 1
        label=None
    _ = ax.hist(run_lens * div, bins=100, cumulative=-1, histtype='step', color=color, density=True, alpha=alpha, zorder=zorder, label=label)
    num_runs.append(len(run_lens))
    idx += 1

xs = np.linspace(0, 10)
ax.plot(xs, np.exp(-xs), label='Random mutations', color='tab:red')
xs = np.linspace(0, 30)
ax.plot(xs, (xs+1)**(-2), label='$x^{-2}$', color='tab:green')
ax.plot(xs, (xs+1)**(-3), label='$x^{-3}$', linestyle='--', color='tab:green')

ymin = 0.8 / max(num_runs)
ax.set_ylim([ymin, ax.get_ylim()[1]])

# ax.set_xlim([0, 38])
ax.set_ylabel('Prob greater than $l$')
ax.set_xlabel("Normalized run length ($l\cdot d$)")
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()

plt.savefig(os.path.join(config.figure_directory, 'supp', 'S29_supp_BSMC_run_length_dist.pdf'), bbox_inches='tight')