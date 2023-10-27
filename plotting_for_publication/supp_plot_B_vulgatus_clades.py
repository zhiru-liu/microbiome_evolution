import matplotlib.pyplot as plt
import numpy as np
import os
import config
from utils import figure_utils, typical_pair_utils, parallel_utils

species_name = 'Bacteroides_vulgatus_57955'
pd_mat = typical_pair_utils.load_pairwise_div_mat(species_name)
major, minor = typical_pair_utils.compute_B_vulgatus_clades()
order = np.concatenate([major, minor])

sample_mask, sample_names = parallel_utils.get_QP_sample_mask(species_name)
good_samples = sample_names[sample_mask]
print(len(good_samples), len(major) + len(minor))

with open('Bv_clades.txt', 'w') as f:
    for i, sample in enumerate(good_samples):
        if i in major:
            f.write('{}\tmajor\n'.format(sample))
        elif i in minor:
            f.write('{}\tminor\n'.format(sample))
        else:
            raise ValueError('Sample not found')
f.close()

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
im = ax.imshow(pd_mat[order, :][:, order])
ax.set_xlabel("Samples")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.set_ylabel("Synonymous divergence")

fig.savefig(os.path.join(config.figure_directory, 'supp_Bv_clades.pdf'), bbox_inches='tight')
