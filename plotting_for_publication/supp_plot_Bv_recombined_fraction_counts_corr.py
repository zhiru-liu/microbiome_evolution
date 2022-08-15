import os
import numpy as np
import sys
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.stats import gamma
sys.path.append("..")
import config
from utils import close_pair_utils, parallel_utils


dat = pickle.load(open(config.B_vulgatus_data_path, 'rb'))

clonal_divs, within_counts, between_counts, filtered_full_df, cf_mask = close_pair_utils.prepare_HMM_results_for_B_vulgatus(
        config.B_vulgatus_data_path, 0.8, cache_intermediate=False, merge_threshold=0)

clonal_divs, within_fractions, between_fractions, filtered_full_df, cf_mask = close_pair_utils.prepare_HMM_results_for_B_vulgatus(
        config.B_vulgatus_data_path, 0.8, mode='fraction', cache_intermediate=False, merge_threshold=0)

mean_length = np.mean(filtered_full_df[filtered_full_df['types']==0]['lengths'].astype(float) * config.second_pass_block_size)

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5
fig, axes = plt.subplots(1, 2, figsize=(6, 1.8))
plt.subplots_adjust(wspace=0.35)
ax = axes[0]
ax.scatter(clonal_divs, within_fractions, s=2)
ax.scatter(clonal_divs, -between_fractions, s=2)
xs = np.linspace(0, 2e-4)
ax.plot(xs, np.zeros(xs.shape), 'k-')
trend_directory = os.path.join(config.plotting_intermediate_directory, "B_vulgatus_fraction_trend_line.csv")
if os.path.exists(trend_directory):
    trend_data = pd.read_csv(trend_directory)
    mean_count = trend_data['within_y'][np.argmin(np.abs(trend_data['within_x'] - 1e-4))]
    print("mean at 1e-4: {:.2f}".format(mean_count))
    ax.plot(trend_data['within_x'], trend_data['within_y'])
    ax.plot(trend_data['between_x'], -trend_data['between_y'])
    ax.fill_between(trend_data['within_x'], trend_data['within_y'] - trend_data['within_sigma'],
                    trend_data['within_y'] + trend_data['within_sigma'], color='tab:blue', alpha=0.25)
    ax.fill_between(trend_data['between_x'], - trend_data['between_y'] - trend_data['between_sigma'],
                    - trend_data['between_y'] + trend_data['between_sigma'], color='tab:orange', alpha=0.25)

ax.add_patch(Rectangle((1e-4, 0), 3e-4, 1, facecolor='grey', edgecolor=None, alpha=0.3))
ax.add_patch(Rectangle((1e-4, 0), 3e-4, -1, facecolor='grey', edgecolor=None, alpha=0.3))
ax.add_patch(Rectangle((0, 0.25), 1e-4, 1, facecolor='grey', edgecolor=None, alpha=0.3))
ax.set_xlim([0, 2e-4])
ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
ax.set_yticklabels(['0.1', '0', '0.1', '0.2', '0.3', '0.4'])
ax.set_xticks([0, 0.5e-4, 1e-4, 1.5e-4, 2e-4])
ax.set_xticklabels([0, 0.5, 1, 1.5, 2])
ax.set_xlabel("Clonal Divergence ($\\times 10^{-4}$)")
ax.set_ylabel('Recombined fraction')

ax = axes[1]
genome_length = 288079.
intervals = [(0, 0)]
xs = np.arange(0, 30)
for i in range(1, 30):
    res = gamma.interval(0.95, i, scale=mean_length)
    intervals.append(res)
intervals = np.array(intervals) / genome_length
ax.fill_between(xs, intervals[:, 0], intervals[:, 1], alpha=0.25)

c = 2057681 / 1e6
total_within_counts = within_counts * c
total_between_counts = between_counts * c
max_interval = intervals[total_within_counts.astype(int), 1]
mask = within_fractions > max_interval
# ax.scatter(total_within_counts[~mask], within_fractions[~mask], color='tab:blue', s=1)
im = ax.scatter(total_within_counts, within_fractions, c=clonal_divs, s=2)
cbar = plt.colorbar(im)
cbar.set_label('Clonal divergence ($\\times 10^{-4}$)', labelpad=8, rotation=90)
cbarax = cbar.ax
cbar.set_ticks([0, 1e-4, 2e-4, 3e-4])
cbar.set_ticklabels(['0', '1', '2', '3'])

# plt.plot(total_between_counts * c, between_fractions, '.')
ax.set_xlabel('Transfer counts')
ax.set_ylabel('Recombined fraction')

fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_Bv_fraction_count_corr.pdf'), bbox_inches='tight')
