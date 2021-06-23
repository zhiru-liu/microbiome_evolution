import os
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
sys.path.append("..")
import config
from utils import close_pair_utils

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5

######################################################################
# preparing data
######################################################################

species_name = 'Bacteroides_vulgatus_57955'
save_path = os.path.join(config.analysis_directory,
                         "closely_related", "second_pass", "{}_two_clades.pickle".format(species_name))
dat = pickle.load(open(save_path, 'rb'))

within_counts, between_counts, full_df = close_pair_utils.merge_and_filter_transfers(dat, separate_clade=True)
T_approxs = np.array(dat['T approxs'])
within_lens = full_df[full_df['types'] == 0]['lengths'].to_numpy().astype(int)
between_lens = full_df[full_df['types'] == 1]['lengths'].to_numpy().astype(int)
BLOCK_SIZE = 10
print("Mean within transfer length: {}".format(np.mean(within_lens) * BLOCK_SIZE))
print("Mean between transfer length: {}".format(np.mean(between_lens) * BLOCK_SIZE))


fig = plt.figure(figsize=(7, 3))
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
within_ct_ax = fig.add_subplot(spec[0, 0])
between_ct_ax = fig.add_subplot(spec[1, 0])
len_dist_ax = fig.add_subplot(spec[:, 1])
# grid = gridspec.GridSpecFromSubplotSpec(1,2, width_ratios=[1,1],wspace=1)
# axes = plt.Subplot(fig, grid[0])
# fig.add_subplot(axes)


######################################################################
# plot transfer counts
######################################################################

plt_prop_cycle = plt.rcParams['axes.prop_cycle']
plt_colors = plt_prop_cycle.by_key()['color']
# fig, axes = plt.subplots(2, 1, sharex=True)
within_ct_ax.get_shared_x_axes().join(within_ct_ax, between_ct_ax)
s1 = within_ct_ax.scatter(T_approxs, within_counts, s=1, c=plt_colors[0])
s2 = between_ct_ax.scatter(T_approxs, between_counts, s=1, c=plt_colors[1])

fig.text(0.04, 0.5, 'Number of transfers', va='center', rotation='vertical')

between_ct_ax.legend((s1, s2), ("Within clade transfers", "Between clade transfers"))
within_ct_ax.set_xlabel("heterozygosity in clonal region")
within_ct_ax.set_ylim([0, within_ct_ax.get_ylim()[1]])
between_ct_ax.set_ylim(within_ct_ax.get_ylim())
between_ct_ax.invert_yaxis()
between_ct_ax.set_xlim([0, 3e-4])
between_ct_ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
within_ct_ax.set_xticklabels([])
between_ct_ax.xaxis.major.formatter._useMathText = True
# within_ct_ax.set_title(' '.join(species_name.split('_')[:-1]))
within_ct_ax.set_title('Transfer count vs divergence')
fig.subplots_adjust(hspace=0)


######################################################################
# plot transfer length distribution
######################################################################
bins = np.arange(max(max(within_lens * BLOCK_SIZE), max(between_lens * BLOCK_SIZE)))
_ = len_dist_ax.hist(within_lens * BLOCK_SIZE, density=True, bins=bins, cumulative=-1, histtype="step",
             label="within clade")
_ = len_dist_ax.hist(between_lens * BLOCK_SIZE, density=True, bins=bins, cumulative=-1, histtype="step",
             label="between clade")

left, bottom, width, height = [0.7, 0.25, 0.15, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])
_ = ax2.hist(within_lens * BLOCK_SIZE, density=True, bins=bins, cumulative=-1, histtype="step",
             label="within clade")
_ = ax2.hist(between_lens * BLOCK_SIZE, density=True, bins=bins, cumulative=-1, histtype="step",
             label="between clade")
ax2.set_yscale('log')

len_dist_ax.legend()
len_dist_ax.set_xlabel('Transfer length / bps')
len_dist_ax.set_ylabel('Survival prob')
len_dist_ax.set_title('Real length distribution')

fig.savefig('test.pdf')
