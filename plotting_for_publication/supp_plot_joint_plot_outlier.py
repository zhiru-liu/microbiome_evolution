import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import config
from utils import typical_pair_utils

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'
fig, axes = plt.subplots(1, 2, figsize=(5, 1.5))
plt.subplots_adjust(wspace=0.3)

species_name = "Alistipes_shahii_62199"
x, y = typical_pair_utils.get_joint_plot_x_y(species_name)

outlier_xy = x[(x>0.75)&(y>0.001)][0], y[(x>0.75)&(y>0.001)][0]
typical_xy = x[(x>0.75)&(y<0.001)][0], y[(x>0.75)&(y<0.001)][0]
outlier_pair = (24, 75)
typical_pair = (21, 34)

axes[0].scatter(x, y, s=2, rasterized=True)
axes[0].plot(outlier_xy[0], outlier_xy[1], 'x', color='tab:orange', markersize=3, label='outlier')
axes[0].plot(typical_xy[0], typical_xy[1], '^', color='tab:green', markersize=3, label='typical')
axes[0].legend()
axes[0].set_xlabel('identical fraction')
axes[0].set_ylabel('pairwise divergence')
axes[0].set_ylim([-0.001, 0.018])

full_df = pd.read_pickle(os.path.join(config.analysis_directory, "closely_related", 'third_pass',
                         species_name + '_all_transfers.pickle'))
sub_df = full_df[full_df['pairs'] == outlier_pair]
typical_sub_df = full_df[full_df['pairs'] == typical_pair]

_ = axes[1].hist(full_df['divergences'].to_numpy(), bins=40)
axes[1].plot(sub_df['divergences'], 8 + np.zeros(sub_df['divergences'].shape), 'x', markersize=3, label='outlier')
axes[1].plot(typical_sub_df['divergences'], 2 + np.zeros(typical_sub_df['divergences'].shape), '^', markersize=3, label='regular')
axes[1].legend()
axes[1].set_xlabel('transfer divergence')
axes[1].set_ylabel('histogram counts')

fig.savefig(os.path.join(config.figure_directory, 'joint_plot_outlier.pdf'), bbox_inches='tight', dpi=600)
