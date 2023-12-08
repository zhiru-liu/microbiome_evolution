import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

from utils import snp_data_utils, figure_utils
import config

species_name = 'Bacteroides_vulgatus_57955'

species_name = 'Bacteroides_vulgatus_57955'
# species_name = 'Eubacterium_siraeum_57634'
save_path = os.path.join(config.analysis_directory, "misc", "dedup", species_name, "unique_events.csv")

dedup_events = pd.read_csv(save_path)
core_mask = snp_data_utils.get_general_site_mask(species_name, allowed_variants=['1D', '2D', '3D', '4D'])
syn_core_mask = snp_data_utils.get_general_site_mask(species_name, allowed_variants=['4D'])
full_to_syn_mask = syn_core_mask[core_mask]  # core -> 4D core
syn_core_length = syn_core_mask.sum()
full_coord_to_syn_coord = np.empty(shape=full_to_syn_mask.shape)
full_coord_to_syn_coord[full_to_syn_mask] = np.arange(syn_core_length).astype(int)
full_coord_to_syn_coord[~full_to_syn_mask] = -1  # sites not in syn core

close_pair_pileup_dedup_within = np.zeros(syn_core_length)
within_events = dedup_events[dedup_events['between clade?'] == 'N']
for i, row in within_events.iterrows():
    start = row['Core genome start loc']
    end = row['Core genome end loc']
    syn_core_start = int(full_coord_to_syn_coord[start])
    syn_core_end = int(full_coord_to_syn_coord[end])
    close_pair_pileup_dedup_within[syn_core_start:syn_core_end + 1] += 1

close_pair_pileup_dedup_between = np.zeros(syn_core_length)
between_events = dedup_events[dedup_events['between clade?'] == 'Y']
for i, row in between_events.iterrows():
    start = row['Core genome start loc']
    end = row['Core genome end loc']
    syn_core_start = int(full_coord_to_syn_coord[start])
    syn_core_end = int(full_coord_to_syn_coord[end])
    close_pair_pileup_dedup_between[syn_core_start:syn_core_end + 1] += 1

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
pielup_path = os.path.join(base_path, 'between_host.csv')
within_cumu_runs = np.loadtxt(pielup_path)

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name + '_between')
pielup_path = os.path.join(base_path, 'between_host.csv')
between_cumu_runs = np.loadtxt(pielup_path)

cp_pileup = close_pair_pileup_dedup_within

# panel A
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5
# mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

fig, ax = plt.subplots(figsize=(7*0.58, 4.8 * 0.24), dpi=300)
colors=['tab:blue', 'tab:orange']

scale_factor = cp_pileup.max() / within_cumu_runs[:, 0].max()
ax.plot(cp_pileup / scale_factor, linewidth=1, color=colors[0])
xs = np.arange(cp_pileup.shape[0])
zeros = np.zeros(cp_pileup.shape[0])
ax.fill_between(xs, zeros, cp_pileup / scale_factor, color=colors[0], alpha=0.5, rasterized=True)

ax.plot(-within_cumu_runs[:, 0], linewidth=1)
ax.fill_between(xs, -within_cumu_runs[:, 0], zeros, alpha=0.5, rasterized=True)

ax.hlines(0, 0, within_cumu_runs.shape[0], 'black', linewidth=1)
ax.set_xlim([0, within_cumu_runs.shape[0]])
ax.set_xlabel("Location along core genome")
ax.set_yticks([0, -0.2])
ax.set_yticklabels(np.around(map(np.abs, ax.get_yticks()), decimals=1))

ax.text(0.02, 0.9, "Number of close-pair transfers (arbitrary scale)", transform=ax.transAxes, fontsize=5)
ax.text(0.02, 0.05, "Sharing probability", transform=ax.transAxes, fontsize=5)
fig.savefig(os.path.join(config.figure_directory, 'supp', 'S36_Bv_pileup_compare_metagenomics.pdf'), bbox_inches='tight')

# panel B
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5
# mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

fig, ax = plt.subplots(figsize=(7*0.58, 4.8 * 0.24), dpi=300)
colors=['tab:blue', 'tab:orange']

scale_factor = close_pair_pileup_dedup_between.max() / 0.25
ax.plot(close_pair_pileup_dedup_between / scale_factor, linewidth=1, color=colors[1])
xs = np.arange(close_pair_pileup_dedup_between.shape[0])
zeros = np.zeros(close_pair_pileup_dedup_between.shape[0])
ax.fill_between(xs, zeros, close_pair_pileup_dedup_between / scale_factor, color=colors[1], alpha=0.5, rasterized=True)

ax.plot(-between_cumu_runs[:, 0], linewidth=1, color=colors[1])
ax.fill_between(xs, -between_cumu_runs[:, 0], zeros, color=colors[1], alpha=0.5, rasterized=True)

ax.hlines(0, 0, between_cumu_runs.shape[0], 'black', linewidth=1)
ax.set_xlim([0, between_cumu_runs.shape[0]])
ax.set_xlabel("Location along core genome")
ax.set_yticks([0, -0.2])
ax.set_ylim(ymin=-0.3, ymax=0.3)
ax.set_yticklabels(np.around(map(np.abs, ax.get_yticks()), decimals=1))

ax.text(0.02, 0.9, "Number of close-pair transfers (arbitrary scale)", transform=ax.transAxes, fontsize=5)
ax.text(0.02, 0.05, "Sharing probability", transform=ax.transAxes, fontsize=5)
fig.savefig(os.path.join(config.figure_directory, 'supp', 'S36_Bv_pileup_compare_metagenomics_between_clade.pdf'), bbox_inches='tight')


# panel C
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5
# mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
ax.scatter(within_cumu_runs[:, 0], cp_pileup, s=2, rasterized=True)
ax.set_xlabel('Sharing probability')
ax.set_ylabel('Num detected transfers \n in close pairs')
corr = np.corrcoef(within_cumu_runs[:, 0], cp_pileup)[0, 1]
ax.text(0.6, 0.85, "B. vulgatus \n$r={0:.2f}$".format(corr), transform=ax.transAxes,)
fig.savefig(os.path.join(config.figure_directory, 'supp', 'S36_Bv_pileup_compare_metagenomics_cvs.pdf'), bbox_inches='tight')

# panel D
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
# mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

corr_df = pd.read_csv(os.path.join(config.plotting_intermediate_directory, 'pileup_correlation_coeff.csv'), index_col=0)

xs = np.arange(corr_df.shape[0])
fig, ax = plt.subplots(figsize=(4, 1), dpi=300)
ax.bar(xs, corr_df.iloc[:, 1],linewidth=0)
ax.axhline(0, color='k')
ax.set_ylabel('$r$')
ax.set_xticks(xs)
species_names = map(lambda x: figure_utils.get_pretty_species_name(x, manual=True), corr_df.iloc[:, 0])
_ = ax.set_xticklabels(species_names, rotation=90)
ax.set_ylim([-1, 1])
ax.set_xlim([-0.5, xs.max()+0.5])
ax.get_xticklabels()[-8].set_fontweight('bold')
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
fig.savefig(os.path.join(config.figure_directory, 'supp', 'S36_pileup_cvs_gut_commensals.pdf'), bbox_inches='tight')