# correlation of sharing with polymorphism
from scipy.stats import spearmanr, pearsonr
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import config
from plotting_for_publication import default_fig_styles
from utils import snp_data_utils, typical_pair_utils

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')
between_data = np.loadtxt(os.path.join(base_path, 'between_host.csv'))
within_data = np.loadtxt(os.path.join(base_path, 'within_host.csv'))

polymorphism_dir = os.path.join(config.plotting_intermediate_directory, 'B_vulgatus_polymorphism.csv')
if os.path.exists(polymorphism_dir):
    pis = np.loadtxt(polymorphism_dir)
    pi = pis[:, 0]
    clade_pi = pis[:, 1]
else:
    dh = snp_data_utils.DataHoarder('Bacteroides_vulgatus_57955', mode='QP', allowed_variants=['4D'])
    pi, clade_pi = typical_pair_utils.get_sitewise_polymorphism(dh, clade_cutoff=0.03)
    np.savetxt(os.path.join(config.plotting_intermediate_directory, 'B_vulgatus_polymorphism.csv'),
               np.vstack([pi, clade_pi]).transpose())

kernel = np.ones(3000) / 3000.
smooth_pi = np.convolve(pi, kernel, mode='same')
smooth_clade_pi = np.convolve(clade_pi, kernel, mode='same')

# mask to remove one of the top peak
mask = np.ones(smooth_pi.shape)
mask[59600:63400] = 0
mask = mask.astype(bool)

print("Correlation with within-clade pi")
print(spearmanr(smooth_clade_pi[:], between_data[:, 1]))
print("Correlation with overall pi")
print(spearmanr(smooth_pi[:], between_data[:, 1]))

print("Correlation with within-clade pi without outlier")
print(spearmanr(smooth_clade_pi[mask], between_data[mask, 1]))
print("Correlation with overallpi without outlier")
print(spearmanr(smooth_pi[mask], between_data[mask, 1]))

# plotting the scatter plot
fig, axes = plt.subplots(1, 1, figsize=(3, 2), dpi=600)
# plt.subplots_adjust(wspace=0.5)
# axes[0].plot(smooth_clade_pi[mask], between_data[mask, 1], '.', markersize=1, rasterized=True)
# axes[0].plot(smooth_clade_pi[~mask], between_data[~mask, 1], '.', markersize=1, color='red', rasterized=True)
# axes[0].set_xlabel('within-clade $\pi$')
# axes[0].set_ylabel('Sharing fraction')

axes.plot(smooth_pi[mask], between_data[mask, 1], '.', markersize=1, rasterized=True)
axes.plot(smooth_pi[~mask], between_data[~mask, 1], '.', markersize=1, color='tab:orange', rasterized=True, label='enriched for\nribosomal proteins')
mpl.rcParams['legend.frameon']  = True
axes.legend()
axes.set_xlabel('Within-clade divergence')
axes.set_ylabel('Sharing probability')

plt.savefig(os.path.join(config.figure_directory, 'supp', 'S30_supp_pi_pileup_correlation.pdf'), bbox_inches='tight')
