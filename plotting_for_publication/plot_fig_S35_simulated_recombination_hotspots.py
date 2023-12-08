import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append("..")
import config
from plotting_for_publication.plot_pileup import plot_single_side

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

recomb_rates = np.loadtxt(os.path.join(config.analysis_directory, 'fastsimbac_data', 'hotspot', 'recomb_rates.txt'))
region_starts = 280000 * recomb_rates[:, 0]
region_ends = 280000 * recomb_rates[:, 1]

fig, pileup_ax = plt.subplots(figsize=(5, 2), dpi=300)
# threshold_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'hotspot', 'thresholds.txt')
# sim_thresholds = np.loadtxt(os.path.join(threshold_path))
for i in range(100):
    ckpt_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'hotspot', '%d.txt' % i)
    cumu_runs = np.loadtxt(ckpt_path)
    # dat = cumu_runs / cumu_runs.mean()
    print(cumu_runs.mean())
    pileup_ax.plot(cumu_runs, linewidth=1, alpha=0.1, color='grey')
    # pileup_ax.set_ylim([0, 0.3])
    pileup_ax.set_xlim([0, len(cumu_runs)])
pileup_ax.set_ylim([0, 0.13])

for i in range(3):
    pileup_ax.axvspan(region_starts[i], region_ends[i], alpha=0.2, color='tab:red')
for i in range(4):
    pileup_ax.axvspan(region_starts[i+3], region_ends[i+3], alpha=0.2, color='tab:blue')
pileup_ax.set_xlabel("Genome location")
pileup_ax.set_ylabel('Sharing probability')
fig.savefig(os.path.join(config.figure_directory, 'supp', 'S35_supp_BSMC_hotcold.pdf'), bbox_inches='tight')