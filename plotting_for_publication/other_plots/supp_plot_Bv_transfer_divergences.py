import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.style import context
import matplotlib as mpl
from matplotlib import cm
import pandas as pd
import config
from utils import typical_pair_utils

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5
species_name = 'Bacteroides_vulgatus_57955'
save_path = os.path.join(config.analysis_directory,
                             "closely_related", "third_pass", "{}_all_transfers_processed.pickle".format(species_name))
run_df = pd.read_pickle(save_path)
cf_cutoff=config.clonal_fraction_cutoff
_, div_dist = typical_pair_utils.get_joint_plot_x_y(species_name)

all_divergences = run_df['synonymous divergences']
fig, axes = plt.subplots(3, 1, figsize=(5,4))
plt.subplots_adjust(hspace=0.5)

histo = np.loadtxt(os.path.join(config.hmm_data_directory, species_name + '.csv'))
mids = histo[0, :40]
within_histo = histo[1, :40] / np.sum(histo[1, :])
between_histo = histo[1, 40:] / np.sum(histo[1, :])
axes[1].bar(mids, within_histo, width=mids[1] - mids[0], label='simulated within-clade', alpha=0.5)
axes[1].bar(mids, between_histo, width=mids[1] - mids[0], label='simulated between-clade', alpha=0.5)
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].set_xlabel('Synonymous divergence in transfer')

bins = np.arange(0, all_divergences.max() + mids[1]-mids[0], mids[1]-mids[0])
axes[2].hist(all_divergences, alpha=0.2, color='tab:grey', bins=bins, label='Total')
axes[2].hist(run_df[run_df['types']==0]['synonymous divergences'], histtype='step', bins=bins, label='Detected within-clade')
axes[2].hist(run_df[run_df['types']==1]['synonymous divergences'], histtype='step', bins=bins, label='Detected between-clade')
axes[2].set_xlim(axes[1].get_xlim())
axes[2].legend()
axes[2].set_ylabel('# transfers')
axes[2].set_xlabel('Synonymous divergence in transfer')

axes[0].hist(div_dist[div_dist<0.03], bins=50)
axes[0].hist(div_dist[div_dist>0.03], bins=50)
axes[0].set_xlim(axes[1].get_xlim())
axes[0].set_xlabel('Pairwise synonymous divergence')
axes[0].set_ylabel('Pairs')
plt.tight_layout()
fig.savefig(os.path.join(config.figure_directory, 'supp_Bv_transfer_divergences.pdf'))