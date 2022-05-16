import matplotlib.pyplot as plt
import numpy as np
import os
import config
from utils import figure_utils, typical_pair_utils

species_name = 'Bacteroides_vulgatus_57955'
pd_mat = typical_pair_utils.load_pairwise_div_mat(species_name)
major, minor = typical_pair_utils.compute_B_vulgatus_clades()
order = np.concatenate([major, minor])

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
im = ax.imshow(pd_mat[order, :][:, order])
ax.set_xlabel("Samples")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.set_ylabel("Synonymous divergence")

fig.savefig(os.path.join(config.figure_directory, 'supp_Bv_clades.pdf'), bbox_inches='tight')
