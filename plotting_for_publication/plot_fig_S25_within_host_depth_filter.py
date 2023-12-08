import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import config
from utils import snp_data_utils

species_name = 'Bacteroides_vulgatus_57955'
dh = snp_data_utils.DataHoarder(species_name, mode='within', allowed_variants=['4D'])

# set up figure
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

idx = 8
major_freq = dh.major_freqs[idx]

fig, axes = plt.subplots(3, 1, figsize=(4, 4.5))

depth_df, median_coverage = dh._get_median_coverage_df(dh.depth_arr[:, idx])
ids = np.arange(median_coverage.shape[0])

axes[0].plot(ids, median_coverage['depths'].to_numpy(), linewidth=0.5)
axes[0].plot(ids, median_coverage['smoothed depths'].to_numpy(), label='Smoothed')
sample_id = dh.good_samples[idx]
axes[0].set_title("Sample id: {}".format(sample_id))
axes[0].set_ylabel('Depths')

axes[1].plot(median_coverage['relative copy number'].to_numpy(), linewidth=0.5)
axes[1].axhline(major_freq, linestyle='dashed', color='tab:pink', linewidth=1, label='Major strain freq')
axes[1].axhline(1-major_freq, linestyle='dotted', color='tab:pink', linewidth=1, label='Minor strain freq')
axes[1].set_ylabel('Rel copy number')

axes[2].plot(median_coverage['zscores'].to_numpy(), linewidth=0.5)
axes[2].axhline(2, linestyle='dotted', color='tab:grey', linewidth=1, label='Z score=2')
axes[2].axhline(-2, linestyle='dotted', color='tab:grey', linewidth=1)
axes[2].set_ylabel('Z scores')
axes[2].set_xlabel('Core gene id')

for i in range(3):
    axes[i].set_xlim([0, median_coverage.shape[0]])
    axes[i].legend(ncol=2, loc='upper right')
median_coverage.to_csv(os.path.join(config.figure_data_directory, 'figS25', 'depth_data.csv'))
plt.tight_layout()
fig.savefig(os.path.join(config.figure_directory, 'supp', 'S25_supp_within_host_depth_filter.pdf'))