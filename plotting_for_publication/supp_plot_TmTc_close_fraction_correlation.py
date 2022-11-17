import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import config

mpl.rcParams['font.size'] = 5
mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'
scatter_size=2

fig, axes = plt.subplots(1, 3, figsize=(6, 1.5))
plt.subplots_adjust(wspace=0.3)

TmTc_df = pd.read_csv(os.path.join(config.figure_directory, 'supp_table', 'TcTm_estimation.csv'))
TmTc_df['Tm/Tc'] = 1 / TmTc_df['Tc/Tm']

axes[0].scatter(TmTc_df['Tm/Tc'], TmTc_df['Close pair fraction'], s=scatter_size)
axes[0].set_xlabel('$T_{mosaic} / T_{mrca}$')
axes[0].set_ylabel('Fraction of "close pairs"')
axes[0].set_title("Empirical")

bsmc_df = pd.read_csv(os.path.join(config.analysis_directory,
                                  'misc', 'BSMC_joint_plot', 'statistics.txt'))
close_counts = bsmc_df['close counts 0.2']
N = 200  # sample size in BSMC sim
frac_close = close_counts / (N * (N-1) / 2)
bsmc_df['Tm/Tc'] = bsmc_df['t'] / (bsmc_df['rho'] * bsmc_df['lambda'] * bsmc_df['t'])

axes[1].scatter(bsmc_df['Tm/Tc'], frac_close, scatter_size)
axes[1].set_xlim(axes[0].get_xlim())
axes[1].set_xlabel('$T_{mosaic} / T_{mrca}$')
axes[1].set_ylabel('Fraction of "close pairs"')
axes[1].set_title("Neutral simulation (zoom in)")

axes[2].scatter(bsmc_df['Tm/Tc'], frac_close, scatter_size)
axes[2].set_xlabel('$T_{mosaic} / T_{mrca}$')
axes[2].set_ylabel('Fraction of "close pairs"')
axes[2].set_title("Neutral simulation")

texts = ['A', 'B', 'C']
for i in range(3):
    text = texts[i]
    ax = axes[i]
    ax.text(-0.12, 1.02, text, transform=ax.transAxes,
               fontsize=9, fontweight='bold', va='top', ha='left')

fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_TmTc_compare.pdf'), bbox_inches='tight')
