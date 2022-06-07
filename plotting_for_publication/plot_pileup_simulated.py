import pandas as pd
import os
import numpy as np
import config
from utils import parallel_utils, core_gene_utils, typical_pair_utils
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotting_for_publication import default_fig_styles

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5

plot_neutral_sim = True

if plot_neutral_sim:
    ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'fastsimbac_rbymu_1')
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    for sim_id in range(400, 500):
        filename = '%d.txt' % sim_id
        save_path = os.path.join(ckpt_dir, filename)
        cumu_runs = np.loadtxt(save_path)
        if sim_id==450:
            ax.plot(cumu_runs, color='tab:blue', alpha=1)
        else:
            ax.plot(cumu_runs, color='grey', alpha=0.1)
    ax.set_ylim([0, 0.3])
    ax.set_xlim([0, 280000])
    fig.savefig(os.path.join(config.figure_directory, 'pileup_fig', 'neutral_sim.pdf'))

############ plot the CV comparison
# load real data
cvs_path = os.path.join(config.plotting_intermediate_directory, 'species_sharing_pileup_cvs.csv')
if os.path.exists(cvs_path):
    real_cvs = np.loadtxt(cvs_path)
else:
    ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical')
    all_cvs = []
    for species in os.listdir(ckpt_dir):
        if species.startswith('.'):
            continue
        sharing_pileup = np.loadtxt(os.path.join(ckpt_dir, species, 'between_host.csv'))
        cv = np.std(sharing_pileup, axis=0) / np.mean(sharing_pileup, axis=0)
        all_cvs.append(cv)
    real_cvs = np.array(all_cvs)
    np.savetxt(cvs_path, real_cvs)
# load sim data
sim_cvs = np.loadtxt(os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'r_scan_statistics', 'cvs.csv'))
sim_cvs = sim_cvs.reshape((-1, 100))
mean_sim_cvs = sim_cvs.mean(axis=1)
sigma_sim_csv = np.std(sim_cvs, axis=1)

# plot comparison
fig, ax = plt.subplots(1, 1, figsize=(3, 2))
ax.scatter(real_cvs[:, 0], np.ones(real_cvs.shape[0]) + np.random.uniform(-0.1, 0.1, size=real_cvs.shape[0]), marker='o', alpha=0.5)
ax.errorbar(mean_sim_cvs, np.zeros(mean_sim_cvs.shape) + np.random.uniform(-0.1, 0.1, size=mean_sim_cvs.shape), xerr=sigma_sim_csv, fmt='o', color='grey', alpha=0.3)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Neutral', 'Real'])
ax.set_xlabel('CV')
fig.savefig(os.path.join(config.figure_directory, 'pileup_fig', 'CV_comparison.pdf'), bbox_inches='tight')

############# plot effect of varying rho
# loading precomputed data for scanning a range of rho
sim_medians = np.loadtxt(os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'r_scan_statistics', 'median_frac.csv'))
sim_medians = sim_medians.reshape((-1, 100))
mean_sim_medians = sim_medians.mean(axis=1)
sigma_sim_medians = np.std(sim_medians, axis=1)
rbymu = [0, 0.1, 0.2, 0.5, 1, 2, 4]

length_scan_median = np.loadtxt(os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'length_scan_median.csv'))
length_scan_cvs = np.loadtxt(os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'length_scan_cvs.csv'))
thresholds = np.arange(10, 35, 2) / 0.00825

fig, axes = plt.subplots(2, 2, figsize=(4, 3))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plot rbymu scan
axes[0, 1].errorbar(rbymu, mean_sim_medians, yerr=sigma_sim_medians, fmt='o', markersize=3)
axes[1, 1].errorbar(rbymu, mean_sim_cvs, yerr=sigma_sim_csv, fmt='o', markersize=3)
axes[1, 1].set_ylim([0, 1])
axes[0, 1].set_xlabel(r'$r/\mu$')
axes[1, 1].set_xlabel(r'$r/\mu$')
axes[0, 1].set_ylabel('Median sharing fraction')
axes[1, 1].set_ylabel('CV')

# plot threshold length scan
axes[0, 0].errorbar(thresholds, np.mean(length_scan_median, axis=0), yerr=np.std(length_scan_median, axis=0), fmt='o', markersize=3)
axes[1, 0].errorbar(thresholds, np.mean(length_scan_cvs, axis=0), yerr=np.std(length_scan_cvs, axis=0), fmt='o', markersize=3)
axes[1, 0].set_ylim([0, 1])
axes[0, 0].set_xlabel('threshold / bps')
axes[1, 0].set_xlabel('threshold / bps')
axes[0, 0].set_ylabel('Median sharing fraction')
axes[1, 0].set_ylabel('CV')

# Hard coded B. vulgatus median sharing fraction...
axes[0, 0].axhline(0.02797646, linestyle='--', color='grey')
axes[0, 0].axvline(3400, linestyle='--', color='grey')

ax = axes[0, 0]
ax.text(-0.00, 1.12, "A", transform=ax.transAxes,
      fontsize=7, fontweight='bold', va='top', ha='left')
ax = axes[0, 1]
ax.text(-0.00, 1.12, "B", transform=ax.transAxes,
              fontsize=7, fontweight='bold', va='top', ha='left')
ax = axes[1, 0]
ax.text(-0.00, 1.12, "C", transform=ax.transAxes,
              fontsize=7, fontweight='bold', va='top', ha='left')
ax = axes[1, 1]
ax.text(-0.00, 1.12, "D", transform=ax.transAxes,
              fontsize=7, fontweight='bold', va='top', ha='left')

fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_BSMC_pileup_CV_varying_rho.pdf'), bbox_inches='tight')