import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import config
from utils import typical_pair_utils

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

fig, axes = plt.subplots(1, 2, figsize=(6, 2.))
plt.subplots_adjust(wspace=0.35)
within_color = '#2b83ba'
between_color = '#fdae61'

ax2 = fig.add_axes([0.4/2+0.5, 0.3, 0.35/2, 0.35])

species_name = 'Alistipes_shahii_62199'
save_path = os.path.join(config.plotting_intermediate_directory,
                         "{}_all_transfers_processed.pickle".format(species_name))
full_df = pd.read_pickle(save_path)
y1 = full_df[full_df['types'] == 1]['lengths'] * config.second_pass_block_size
y2 = full_df[full_df['types'] == 0]['lengths'] * config.second_pass_block_size
print("Within counts: {}; between counts: {}".format(len(y2), len(y1)))

dx = 50
bins = np.arange(0, max(y2) + dx, dx)
_ = axes[1].hist(y2, bins=bins, cumulative=-1, density=True, color=within_color, histtype='step', label='within clade')
_ = axes[1].hist(y1, bins=bins, cumulative=-1, density=True, color=between_color, histtype='step', label='between clade')

_ = ax2.hist(y2, bins=bins, cumulative=-1, density=True, color=within_color, histtype='step')
_ = ax2.hist(y1, bins=bins, cumulative=-1, density=True, color=between_color, histtype='step')

axes[1].set_xlim(left=-10)
ax2.set_xlim(left=0, right=1000)
axes[1].legend()
axes[1].set_xlabel("transfer length ($x$)")
axes[1].set_ylabel("probability longer than $x$")

_, div_dist = typical_pair_utils.get_joint_plot_x_y('Alistipes_shahii_62199')
axes[0].hist(div_dist, bins=100)
axes[0].set_yscale('log')
axes[0].set_xlabel('pairwise synonymous divergence')
axes[0].set_ylabel('counts')

fig.savefig(os.path.join(config.figure_directory, 'supp', "S22_supp_A_shahii_transfer_length.pdf"), bbox_inches='tight')
