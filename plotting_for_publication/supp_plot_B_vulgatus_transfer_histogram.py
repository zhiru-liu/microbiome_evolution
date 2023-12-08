import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
import config
from utils import snp_data_utils
from utils.close_pair_utils import sample_blocks

species_name = 'Bacteroides_vulgatus_57955'
block_size = 1000
num_samples = 50000
dh = parallel_utils.DataHoarder(species_name, mode="QP")
local_divs, genome_divs = sample_blocks(dh, num_samples=num_samples, block_size=block_size)

between_mean = np.mean(local_divs[genome_divs > 0.03])
within_mean = np.mean(local_divs[genome_divs <= 0.03])


bins = np.linspace(0, max(local_divs), 41)
fig, ax = plt.subplots(2, 1, figsize=(4, 3))
plt.subplots_adjust(hspace=0.5)
_ = ax[0].hist(local_divs[genome_divs > 0.03], bins=bins, alpha=0.5, density=True, color='tab:orange', label='Empirical between clade')
_ = ax[0].hist(local_divs[genome_divs <= 0.03], bins=bins, alpha=0.5, density=True, color='tab:blue', label='Empirical within clade')
_ = ax[1].hist(np.random.poisson(within_mean*block_size, num_samples) / float(block_size),
             bins=bins, alpha=0.5, density=True, color='tab:blue', label='Poisson within clade')
_ = ax[1].hist(np.random.poisson(between_mean*block_size, num_samples) / float(block_size),
             bins=bins, alpha=0.5, density=True, color='tab:orange', label='Poisson between clade')
ax[1].set_xlim(ax[0].get_xlim())
ax[0].legend()
ax[1].legend()
ax[1].set_xlabel('simulated transfer block divergence')
fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_B_vulgatus_empirical_histogram.pdf'), bbox_inches='tight')
