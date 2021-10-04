import os
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
sys.path.append("..")
import config
from utils import close_pair_utils, parallel_utils

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5

######################################################################
# preparing data
######################################################################

species_name = 'Bacteroides_vulgatus_57955'
# dh = parallel_utils.DataHoarder(species_name, mode="QP")

save_path = os.path.join(config.analysis_directory,
                         "closely_related", "debug", "{}_two_clades.pickle".format(species_name))
dat = pickle.load(open(save_path, 'rb'))

within_counts, between_counts, full_df = close_pair_utils.merge_and_filter_transfers(dat, separate_clade=True)
full_df.to_pickle(
    os.path.join(config.analysis_directory, "closely_related", 'third_pass', species_name + '_all_transfers_two_clades.pickle'))

clonal_snps = np.array(dat['clonal snps'])
clonal_lens = np.array(dat['clonal lengths'])
clonal_divs = clonal_snps / clonal_lens.astype(float)

within_lens = full_df[full_df['types'] == 0]['lengths'].to_numpy().astype(int)
between_lens = full_df[full_df['types'] == 1]['lengths'].to_numpy().astype(int)
BLOCK_SIZE = 10
print("Mean within transfer length: {}".format(np.mean(within_lens) * BLOCK_SIZE))
print("Mean between transfer length: {}".format(np.mean(between_lens) * BLOCK_SIZE))

# mapping out grid
fig = plt.figure(figsize=(7, 5.5))
outer_grid = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 2.5], hspace=0.3, figure=fig)

top_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[2,1.5],wspace=0,subplot_spec=outer_grid[0])

top_right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.37, 2],hspace=0.4,subplot_spec=top_grid[1])
example_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1,1],hspace=0.2,subplot_spec=top_right_grid[1])

bottom_grid = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 1], wspace=0.2, subplot_spec=outer_grid[1])

# adding axes
ct_ax = fig.add_subplot(bottom_grid[0])

len_dist_ax = fig.add_subplot(bottom_grid[1])

ex0_ax = fig.add_subplot(top_right_grid[0])

ex1_ax = fig.add_subplot(example_grid[0])

ex2_ax = fig.add_subplot(example_grid[1])

######################################################################
# example species
######################################################################

pi_color = 'tab:grey'
# within_color = 'tab:blue'
within_color = '#2b83ba'
# between_color = 'tab:orange'
between_color = '#fdae61'
snp_color = 'tab:red'


def plot_typical_pair(ax, dh, pair):
    cache_file = "cached_close_pair_{}.csv".format(pair)
    if os.path.exists(cache_file):
        snp_vec = np.loadtxt(cache_file)
    else:
        snp_vec, _ = dh.get_snp_vector(pair)
        np.savetxt(cache_file, snp_vec)

    window_size = 1000
    local_pi = np.convolve(snp_vec, np.ones(window_size) / float(window_size), mode='same')
    snp_locs = np.nonzero(snp_vec)[0]
    shown_snp_locs = snp_locs

    ax.plot(local_pi, label='local heterozygosity', color=pi_color, linewidth=1)
    ax.plot(shown_snp_locs, np.zeros(shown_snp_locs.shape), '|', color=snp_color, label='individual snps', markersize=2)
    ax.set_ylim([-0.005, 0.05])
    ax.set_yticks((0.0, 0.04))
    labels = ['0.0', '0.04']
    ax.set_xlim([0, 260000])
    ax.set_xlabel('Synonymous core genome location')
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))


def plot_example_pair(ax, dh, pair, full_df, if_legend=True):
    cache_file = "cached_close_pair_{}.csv".format(pair)
    if os.path.exists(cache_file):
        snp_vec = np.loadtxt(cache_file)
    else:
        snp_vec, _ = dh.get_snp_vector(pair)
        np.savetxt(cache_file, snp_vec)

    window_size = 1000
    local_pi = np.convolve(snp_vec, np.ones(window_size) / float(window_size), mode='same')

    snp_locs = np.nonzero(snp_vec)[0]
    shown_snp_locs = snp_locs

    block_size = 10
    sub_df = full_df[full_df['pairs'] == pair]

    xs = np.arange(len(snp_vec))
    between_ys = np.zeros(xs.shape)
    within_ys = np.zeros(xs.shape)
    for _, row in sub_df.iterrows():
        start = row['starts'] * block_size
        end = (row['ends'] + 1) * block_size
        if row['types'] == 0:
            within_ys[start:end] = 1
        else:
            between_ys[start:end] = 1

    ax.plot(xs, -between_ys * 0.03, color=between_color, linewidth=1)
    ax.plot(xs, -within_ys * 0.03, color=within_color, linewidth=1)

    ax.plot(local_pi, label='local heterozygosity', color=pi_color, linewidth=1)
    ax.plot(shown_snp_locs, np.zeros(len(shown_snp_locs)), '|', color=snp_color, label='individual snps', markersize=2)

    ax.set_yticks((-0.03, 0.0, 0.04, 0.08))
    labels = ['transfer', '0.0', '0.04', '0.08']
    if if_legend:
        ax.legend()
    ax.set_yticklabels(labels)
    # ax.set_title(pair)
    ax.set_ylim([-0.035, 0.10])

    ax.set_xlim([0, 260000])
    ax.set_xlabel('Synonymous core genome location')


plot_typical_pair(ex0_ax, None, (0, 128))
plot_example_pair(ex1_ax, None, (128, 170), full_df, if_legend=False)
plot_example_pair(ex2_ax, None, (39, 74), full_df, if_legend=False)
ex0_ax.set_xticklabels([])
ex1_ax.set_xticklabels([])
ex0_ax.set_xlabel('')
ex1_ax.set_xlabel('')


######################################################################
# plot transfer counts
######################################################################

plt_prop_cycle = plt.rcParams['axes.prop_cycle']
plt_colors = plt_prop_cycle.by_key()['color']
# fig, axes = plt.subplots(2, 1, sharex=True)
# within_ct_ax.get_shared_x_axes().join(within_ct_ax, between_ct_ax)
# s1 = within_ct_ax.scatter(T_approxs, within_counts, s=1, c=plt_colors[0])
# s2 = between_ct_ax.scatter(T_approxs, between_counts, s=1, c=plt_colors[1])

s1 = ct_ax.scatter(clonal_divs, within_counts, s=2, c=within_color, label='Within clade transfers')
s2 = ct_ax.scatter(clonal_divs, -np.array(between_counts), s=2, c=between_color, label='Between clade transfers')
ct_ax.plot(clonal_divs, np.zeros(clonal_snps.shape), 'k-')

ct_ax.set_xlim([0, 2e-4])
ct_ax.set_ylim([-20, 40])
ct_ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

ct_ax.set_yticks([-20, -10, 0, 10, 20, 30])
ct_ax.set_yticklabels(['20', '10', '0', '10', '20', '30'])
# ct_ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ct_ax.xaxis.major.formatter._useMathText = True

ct_ax.set_ylabel("Number of transfers")
ct_ax.set_xlabel("Clonal snps")

# fig.text(0.04, 0.5, 'Number of transfers', va='center', rotation='vertical')

# between_ct_ax.legend((s1, s2), ("Within clade transfers", "Between clade transfers"))
# within_ct_ax.set_xlabel("heterozygosity in clonal region")
# within_ct_ax.set_ylim([0, within_ct_ax.get_ylim()[1]])
# between_ct_ax.set_ylim(within_ct_ax.get_ylim())
# between_ct_ax.invert_yaxis()
# between_ct_ax.set_xlim([0, 3e-4])
# between_ct_ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
# within_ct_ax.set_xticklabels([])
# between_ct_ax.xaxis.major.formatter._useMathText = True
# # within_ct_ax.set_title(' '.join(species_name.split('_')[:-1]))
# within_ct_ax.set_title('Transfer count vs divergence')
# fig.subplots_adjust(hspace=0)


######################################################################
# plot transfer length distribution
######################################################################
bins = np.arange(max(max(within_lens * BLOCK_SIZE), max(between_lens * BLOCK_SIZE)))
_ = len_dist_ax.hist(within_lens * BLOCK_SIZE, density=True, bins=bins, cumulative=-1, histtype="step",
             label="within clade", color=within_color)
_ = len_dist_ax.hist(between_lens * BLOCK_SIZE, density=True, bins=bins, cumulative=-1, histtype="step",
             label="between clade", color=between_color)

len_dist_ax.set_xlim([0, 20000])

left, bottom, width, height = [0.7, 0.18, 0.15, 0.15]
ax2 = fig.add_axes([left, bottom, width, height])
_ = ax2.hist(within_lens * BLOCK_SIZE, density=True, bins=bins, cumulative=-1, histtype="step",
             label="within clade", color=within_color)
_ = ax2.hist(between_lens * BLOCK_SIZE, density=True, bins=bins, cumulative=-1, histtype="step",
             label="between clade", color=between_color)
ax2.set_yscale('log')
ax2.set_xlim([0, 40000])

len_dist_ax.legend()
len_dist_ax.set_xlabel('Transfer length / bps')
len_dist_ax.set_ylabel('Survival prob')
# len_dist_ax.set_title('Real length distribution')

fig.patch.set_alpha(0.0)
fig.savefig('test_close_pair.pdf', bbox_inches="tight")
