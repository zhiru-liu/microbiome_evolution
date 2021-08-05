import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
import config
from utils import parallel_utils
from utils.close_pair_utils import sample_blocks

species_name = 'Bacteroides_vulgatus_57955'
block_size = 1000
num_samples = 50000
dh = parallel_utils.DataHoarder(species_name, mode="QP")
local_divs, genome_divs = sample_blocks(dh, num_samples=num_samples, block_size=block_size)

within_mean = np.mean(local_divs[genome_divs > 0.03])
between_mean = np.mean(local_divs[genome_divs <= 0.03])


bins = np.linspace(0, max(local_divs), 41)
_ = plt.hist(local_divs[genome_divs > 0.03], bins=bins, histtype='step', density=True, label='Empirical between clade')
_ = plt.hist(local_divs[genome_divs <= 0.03], bins=bins, histtype='step', density=True, label='Empirical within clade')
_ = plt.hist(np.random.poisson(within_mean*block_size, num_samples) / float(block_size),
             bins=bins, histtype='step', density=True, label='Poisson within clade')
_ = plt.hist(np.random.poisson(between_mean*block_size, num_samples) / float(block_size),
             bins=bins, histtype='step', density=True, label='Poisson between clade')
plt.legend()
plt.xlabel('simulated transfer block divergence')
plt.savefig('B_vulgatus_empirical_histogram.pdf')
