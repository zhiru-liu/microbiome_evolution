import os
import numpy as np
import matplotlib.pyplot as plt
from utils import close_pair_utils
from plotting_for_publication import plot_B_vulgatus_close_pair
import config

save_path = config.B_vulgatus_data_path
cutoffs = np.arange(0.7, 0.96, 0.01)
cutoffs_to_plot = [0.7, 0.8, 0.9]
BLOCK_SIZE = config.second_pass_block_size

number_events = []
mean_within = []
mean_between = []
median_within = []
median_between = []

for cf_cutoff in cutoffs:
    clonal_divs, within_counts, between_counts, full_df = close_pair_utils.prepare_HMM_results_for_B_vulgatus(
        save_path, cf_cutoff, cache_intermediate=False)
    number_events.append(full_df.shape[0])
    within_lens = full_df[full_df['types']==0]['lengths'].to_numpy().astype(int) * BLOCK_SIZE
    between_lens = full_df[full_df['types']==1]['lengths'].to_numpy().astype(int) * BLOCK_SIZE

    mean_within.append(within_lens.mean())
    median_within.append(np.median(within_lens))
    mean_between.append(between_lens.mean())
    median_between.append(np.median(between_lens))
    if np.around(cf_cutoff, 2) in cutoffs_to_plot:
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.))
        plt.subplots_adjust(wspace=0.3)
        plot_B_vulgatus_close_pair.plot_scatter(axes[0], clonal_divs, within_counts, between_counts, False)
        plot_B_vulgatus_close_pair.plot_distributions(fig, axes[1], within_lens, between_lens,
                                                      inset_location=[0.7, 0.25, 0.15, 0.3])
        fig.savefig(os.path.join(config.analysis_directory, 'misc', 'B_vulgatus_varying_cf', 'cf_%.2f.pdf'%cf_cutoff),
                    bbox_inches='tight')
        plt.close()

fig, axes = plt.subplots(1, 3, figsize=(6, 2))
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
fig.savefig(os.path.join(config.analysis_directory, 'misc', 'B_vulgatus_varying_cf', "overall_trend.pdf"))
