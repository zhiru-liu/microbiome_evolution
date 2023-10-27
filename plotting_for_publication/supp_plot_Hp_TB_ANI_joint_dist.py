import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import config

ani_res = os.path.join(config.plotting_intermediate_directory, 'pathogen_joint_distribution', 'Hp_refseq_ani.txt')
hp_ani = pd.read_csv(ani_res, delimiter='\t', header=None)
hp_ani.columns = ['query', 'ref', 'ani', 'num maps', 'num frags']

ani_res = os.path.join(config.plotting_intermediate_directory, 'pathogen_joint_distribution', 'Tb_refseq_ani.txt')
tb_ani = pd.read_csv(ani_res, delimiter='\t', header=None)
tb_ani.columns = ['query', 'ref', 'ani', 'num maps', 'num frags']

hp_ani = hp_ani[hp_ani['query']!=hp_ani['ref']]
tb_ani = tb_ani[tb_ani['query']!=tb_ani['ref']]

from utils.typical_pair_utils import partial_recombination_curve
hp_dat = np.load(os.path.join(config.plotting_intermediate_directory, 'pathogen_joint_distribution', 'H_pylori_joint.npy'))
tb_dat = np.load(os.path.join(config.plotting_intermediate_directory, 'pathogen_joint_distribution', 'TB_joint.npy'))

theta = 0.03
F_hp, alpha_hp = partial_recombination_curve(hp_dat[:, 1], hp_dat[:, 2], theta=theta, return_alpha=True, min_x=0.5)
hp_rm = 1000 * theta / (1/alpha_hp - 1)

asexual_part = -np.log(tb_dat[:, 1]) / 1000
theta = 1e-3
F_tb, alpha_tb = partial_recombination_curve(tb_dat[:, 1], tb_dat[:, 2]-asexual_part, theta=theta, return_alpha=True, min_x=0.5)
tb_rm = 1000 * theta / (1/alpha_tb - 1)

print(hp_rm, tb_rm)

def sample_uniform(df):
    num_bins = 100
    hist, bin_edges = np.histogram(df['ani'], bins=num_bins)
    # Use digitize to find the bin indices for each data point
    bin_indices = np.digitize(df['ani'], bin_edges) - 1

    num_samples_per_bin = 50
    data = np.arange(len(df['ani']))
    sampled_data = []
    for bin_index in range(num_bins):
        bin_mask = bin_indices == bin_index
        bin_data = data[bin_mask]
        num_samples = min(num_samples_per_bin, len(bin_data))
        if num_samples > 0:
            sampled_bin_data = np.random.choice(bin_data, size=num_samples, replace=False)
            sampled_data.extend(sampled_bin_data)
    return sampled_data

hp_sampled = sample_uniform(hp_ani)
tb_sampled = sample_uniform(tb_ani)

import matplotlib as mpl
fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

fig, axes = plt.subplots(2, 2, figsize=(6, 4), dpi=300)
plt.subplots_adjust(wspace=0.28, hspace=0.3)

# axes[0, 0].set_yscale('log')
_ = axes[0, 0].hist(hp_ani['ani'], bins=100,alpha=0.5)
axes[0, 0].set_xlim([85, 100])
left, bottom, width, height = [0.18, 0.74, 0.12, 0.12]
ax2 = fig.add_axes([left, bottom, width, height])
_, bins, _ = ax2.hist(hp_ani['ani'], bins=100,alpha=0.5)
ax2.hist(hp_ani['ani'].iloc[hp_sampled], bins=bins, alpha=0.5, color='tab:orange', histtype='step', label='sampled')
ax2.set_yscale('log')
ax2.set_yticks([1e3, 1e5])
ax2.set_xticks([90, 95, 100])

# axes[0, 1].set_yscale('log')
_ = axes[0, 1].hist(tb_ani['ani'], bins=100,alpha=0.5)

left, bottom, width, height = [0.62, 0.74, 0.12, 0.12]
ax3 = fig.add_axes([left, bottom, width, height])
_, bins, _ = ax3.hist(tb_ani['ani'], bins=100,alpha=0.5)
ax3.hist(tb_ani['ani'].iloc[tb_sampled], bins=bins, alpha=0.5, color='tab:orange', histtype='step', label='sampled')
ax3.set_yscale('log')
ax3.set_xticks([99, 99.5, 100])

xs = np.linspace(0.01, 1, 100)
asexual_ys = -np.log(xs) / 1000 # block size=1000
axes[1, 0].plot(xs, asexual_ys, color='grey', linestyle='--')
axes[1, 1].plot(xs, asexual_ys, color='grey', linestyle='--')

xs = np.linspace(0.05, 1, 100)
ys = F_hp(xs)
axes[1, 0].plot(xs, ys, '--', color='tab:red', zorder=2, label='partial recombination')

xs = np.linspace(0.05, 1, 100)
ys = F_tb(xs)
axes[1, 1].plot(xs, ys + asexual_ys, '--', color='tab:red', zorder=2, label='partial recombination')

axes[1, 0].scatter(hp_dat[:, 1], hp_dat[:, 2], s=1)
axes[1, 1].scatter(tb_dat[:, 1], tb_dat[:, 2], s=1)

axes[0, 0].set_xlabel('ANI estimate')
axes[0, 1].set_xlabel('ANI estimate')
axes[1, 0].set_xlabel('Fraction of identical blocks')
axes[1, 1].set_xlabel('Fraction of identical blocks')
axes[1, 0].set_ylabel('Pairwise divergence')
axes[1, 1].set_ylabel('Pairwise divergence')

axes[0, 0].text(-0.1, 1.12, "A", transform=axes[0, 0].transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
axes[0, 1].text(-0.1, 1.12, "B", transform=axes[0, 1].transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.1, 1.12, "C", transform=axes[1, 0].transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.1, 1.12, "D", transform=axes[1, 1].transAxes, fontsize=9, fontweight='bold', va='top', ha='left')

fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_Hp_TB_joint.pdf'), bbox_inches='tight')