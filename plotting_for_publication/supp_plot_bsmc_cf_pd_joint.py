import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import config
from utils import figure_utils
from utils.typical_pair_utils import get_joint_plot_x_y

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'


fig = plt.figure(figsize=(6.5, 4.))
outer_grid = gridspec.GridSpec(3, 3, width_ratios=[1,1,1],wspace=0.35,figure=fig)
plt.subplots_adjust(wspace=0.25, hspace=0.35)

data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'for_pileup')
exp_df = pd.read_csv(os.path.join(data_dir, 'b_vulgatus', 'experiments.txt'), sep=' ')
exp_df = exp_df.set_index(['rbymu', 'lambda', 'rep'])

rbymus = [0.1, 0.5, 2]
lambs = [500, 1000, 2000]

for i in range(3):
    for j in range(3):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1,2, width_ratios=[4,1],wspace=0.2,subplot_spec=outer_grid[i, j])
        scatter_ax = fig.add_subplot(inner_grid[0])
        marg_ax = fig.add_subplot(inner_grid[1], sharey=scatter_ax)

        plt.setp(marg_ax.get_yticklabels(), visible=False)
        plt.setp(marg_ax.get_yticklabels(minor=True), visible=False)

        sim_id = int(exp_df.loc[(rbymus[i], lambs[j], 1), 'sim_id'])
        # preparing x&y
        pd_mat = np.loadtxt(os.path.join(data_dir, 'b_vulgatus', 'pd', 'pd_%d.csv' % sim_id))
        cf_mat = np.loadtxt(os.path.join(data_dir, 'b_vulgatus', 'cf', 'cf_%d.csv' % sim_id))
        x = cf_mat[np.triu_indices(cf_mat.shape[0], 1)]
        y = pd_mat[np.triu_indices(pd_mat.shape[0], 1)]

        xs = np.linspace(0.01, 1, 100)
        ys = -np.log(xs) / config.first_pass_block_size
        scatter_ax.plot(xs, ys, '--r', zorder=1, label='indep. SNPs')

        scatter_ax.scatter(x, y, s=0.3, linewidth=0, zorder=2, rasterized=True)
        marg_ax.hist(y, orientation='horizontal', bins=100, alpha=0.6)

        marg_ax.set_xscale('log')
        # if 'Barnesiella' in species_name:
        marg_ax.set_xticks([10, 1000])
        # else:
        #     marg_ax.set_xticks([1, 10, 100])

        if j==0:
            scatter_ax.set_ylabel('$\\rho/\\mu=%.1f$\n\nPairwise syn divergence'%rbymus[i])
        if i==2:
            scatter_ax.set_xlabel('Identical fraction')
        if i==0:
            # scatter_ax.set_title(r"$\rho/\mu=%f$")
            scatter_ax.set_title(r"$l_r=%d$"%lambs[j])
        # scatter_ax.legend()


fig.savefig(os.path.join(config.figure_directory, 'test_supp_joint_bsmc.pdf'), bbox_inches='tight', dpi=600)
