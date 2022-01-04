import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import config
from utils import parallel_utils, close_pair_utils

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

dh = parallel_utils.DataHoarder('Bacteroides_vulgatus_57955', mode='QP', allowed_variants=['4D'])

# getting the pairs
clonal_frac_dir = os.path.join(config.analysis_directory, 'pairwise_clonal_fraction',
                               'between_hosts', 'Bacteroides_vulgatus_57955.csv')
clonal_frac_mat = np.loadtxt(clonal_frac_dir, delimiter=',')
single_sub_idxs = dh.single_subject_samples
clonal_frac_mat = clonal_frac_mat[single_sub_idxs, :][:, single_sub_idxs]

firsts, seconds = np.nonzero(clonal_frac_mat > config.clonal_fraction_cutoff)
close_pairs = zip(single_sub_idxs[firsts[firsts < seconds]], single_sub_idxs[seconds[firsts < seconds]])

firsts, seconds = np.nonzero((clonal_frac_mat > config.typical_clonal_fraction_cutoff)
                             & (clonal_frac_mat < config.clonal_fraction_cutoff))
intermediate_pairs = zip(single_sub_idxs[firsts[firsts < seconds]], single_sub_idxs[seconds[firsts < seconds]])


def process_pairs(pairs, block_size=500, clade_div_cutoff=0.03, clonal_block_cutoff=3):
    between_counts = []
    within_counts = []
    for pair in pairs:
        snp_vec, _ = dh.get_snp_vector(pair)
        blocks = close_pair_utils.to_block(snp_vec, block_size)
        between_count = np.sum(blocks > clade_div_cutoff * block_size)
        within_count = np.sum((blocks <= clade_div_cutoff * block_size) & (blocks > clonal_block_cutoff))
        between_counts.append(between_count)
        within_counts.append(within_count)
    return np.array(between_counts), np.array(within_counts)


fig, ax = plt.subplots(figsize=(4, 3))
block_size = 500
total_blocks = float(np.sum(dh.general_mask) // block_size)
print("Computing fractions for close pairs")
between_counts, within_counts = process_pairs(close_pairs, block_size=block_size)
# np.savetxt('close_extrapolation.txt', np.vstack([between_counts, within_counts]))
# adding some noise to show density
ax.scatter((within_counts + 0.2*np.random.normal(size=len(within_counts))) / total_blocks,
           (between_counts + 0.2*np.random.normal(size=len(within_counts))) / total_blocks, s=2, color='tab:green',
           label='close pairs')

print("Computing fractions for intermediate pairs")
between_counts, within_counts = process_pairs(intermediate_pairs)
# np.savetxt('intermediate_extrapolation.txt', np.vstack([between_counts, within_counts]))
ax.scatter((within_counts + 0.2*np.random.normal(size=len(within_counts))) / total_blocks,
           (between_counts + 0.2*np.random.normal(size=len(within_counts))) / total_blocks, s=2, color='tab:blue',
           label='intermediate pairs')

ax.legend()
ax.set_xlabel(r'$f_{r}$, within')
ax.set_ylabel(r'$f_{r}$, between')
ax.set_xlim(xmin=-0.01)
ax.set_ylim(ymin=-0.002)
fig.savefig(os.path.join(config.figure_directory, 'B_vulgatus_extrapolation.pdf'), bbox_inches='tight')
