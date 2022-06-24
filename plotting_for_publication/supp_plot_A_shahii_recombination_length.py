import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import config

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

fig, ax = plt.subplots(figsize=(4, 3))
within_color = '#2b83ba'
between_color = '#fdae61'

ax2 = fig.add_axes([0.45, 0.3, 0.35, 0.35])

species_name = 'Alistipes_shahii_62199'
save_path = os.path.join(config.analysis_directory,
                         "closely_related", "third_pass", "{}_all_transfers_processed.pickle".format(species_name))
full_df = pd.read_pickle(save_path)
y1 = full_df[full_df['types'] == 1]['lengths'] * config.second_pass_block_size
y2 = full_df[full_df['types'] == 0]['lengths'] * config.second_pass_block_size
print("Within counts: {}; between counts: {}".format(len(y2), len(y1)))

dx = 50
bins = np.arange(0, max(y2) + dx, dx)
_ = ax.hist(y2, bins=bins, cumulative=-1, density=True, color=within_color, histtype='step', label='within clade')
_ = ax.hist(y1, bins=bins, cumulative=-1, density=True, color=between_color, histtype='step', label='between clade')

_ = ax2.hist(y2, bins=bins, cumulative=-1, density=True, color=within_color, histtype='step')
_ = ax2.hist(y1, bins=bins, cumulative=-1, density=True, color=between_color, histtype='step')

ax.set_xlim(left=-10)
ax2.set_xlim(left=0, right=1000)
ax.legend()
ax.set_xlabel("transfer length")
ax.set_ylabel("probability of longer")
fig.savefig(os.path.join(config.figure_directory, 'supp', "supp_A_shahii_transfer_length.pdf"), bbox_inches='tight')
