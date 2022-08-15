import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import close_pair_utils
from plotting_for_publication import plot_B_vulgatus_close_pair
import config

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

save_path = config.B_vulgatus_data_path
cutoffs = np.arange(0.7, 0.96, 0.01)
# cutoffs = np.arange(0.7, 0.96, 0.1)
cutoffs_to_plot = [0.7, 0.8, 0.9]
BLOCK_SIZE = config.second_pass_block_size

number_events = []
mean_within = []
mean_between = []
median_within = []
median_between = []

fig, axes = plt.subplots(3, 2, figsize=(4.2, 4.5))
plt.subplots_adjust(wspace=0.4, hspace=0.5)
idx = 0
for i, cf_cutoff in enumerate(cutoffs):
    clonal_divs, within_counts, between_counts, full_df, cf_mask = close_pair_utils.prepare_HMM_results_for_B_vulgatus(
        save_path, cf_cutoff, cache_intermediate=False, filter_threshold=5)
    number_events.append(full_df.shape[0])
    clonal_divs = clonal_divs[cf_mask]
    within_counts = within_counts[cf_mask]
    between_counts = between_counts[cf_mask]
    within_lens = full_df[full_df['types']==0]['lengths'].to_numpy().astype(int) * BLOCK_SIZE
    between_lens = full_df[full_df['types']==1]['lengths'].to_numpy().astype(int) * BLOCK_SIZE

    mean_within.append(within_lens.mean())
    median_within.append(np.median(within_lens))
    mean_between.append(between_lens.mean())
    median_between.append(np.median(between_lens))
    if np.around(cf_cutoff, 2) in cutoffs_to_plot:
        # fig, axes = plt.subplots(1, 2, figsize=(6, 2.))
        plot_B_vulgatus_close_pair.plot_scatter(axes[idx, 0], clonal_divs, within_counts, between_counts, False, sci_format=False)
        plot_B_vulgatus_close_pair.plot_distributions(fig, axes[idx, 1], within_lens, between_lens,
                                                      inset_location=None)
        axes[idx, 1].set_xticks([0, 5000, 10000, 15000, 20000])
        axes[idx, 0].set_xticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'])
        idx += 1
for i in range(2):
    axes[i, 0].set_xlabel('')
    axes[i, 1].set_xlabel('')
for i in range(3):
    axes[i, 0].set_ylabel('# of transfers / 1Mbps')
    axes[i, 0].set_title('CF cutoff %.1f' % (cutoffs_to_plot[i]))
    axes[i, 1].set_title('CF cutoff %.1f' % (cutoffs_to_plot[i]))
axes[2, 0].set_xlabel(r'Clonal divergence ($\times 10^{-4}$)')

fig.savefig(os.path.join(config.figure_directory, 'supp_close_pair_robustness_example_cfs.pdf'),
            bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.3))
plt.subplots_adjust(wspace=0.4)
axes[0].plot(cutoffs, number_events)
axes[0].set_ylabel('num good transfers')
axes[0].set_xlabel('clonal fraction cutoff')

axes[1].plot(cutoffs, mean_within, label='within clade')
axes[1].plot(cutoffs, mean_between, label='between clade')
axes[1].legend()
axes[1].set_xlabel('clonal fraction cutoff')
axes[1].set_ylabel('mean transfer length')

axes[2].plot(cutoffs, median_within, label='within clade')
axes[2].plot(cutoffs, median_between, label='between clade')
axes[2].legend()
axes[2].set_xlabel('clonal fraction cutoff')
axes[2].set_ylabel('median transfer length')
fig.savefig(os.path.join(config.figure_directory, 'supp_close_pair_robustness_overall_trend.pdf'), bbox_inches='tight')
