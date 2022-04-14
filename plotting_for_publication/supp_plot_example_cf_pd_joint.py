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


fig = plt.figure(figsize=(6, 4.))
outer_grid = gridspec.GridSpec(2,2, width_ratios=[1,1],wspace=0.3,figure=fig)
plt.subplots_adjust(wspace=0.25, hspace=0.6)

ex1 = 'Prevotella_copri_61740'  # lack of close pairs
ex2 = 'Roseburia_inulinivorans_61943'  # lack of close pairs
ex3 = 'Barnesiella_intestinihominis_62208'  # minimal spread in joint plot
# ex4 = 'Dialister_invisus_61905'  # large spread in joint plot
ex4 = 'Eubacterium_rectale_56927'  # population structure

species = [ex1, ex2, ex3, ex4]

for i in range(2):
    for j in range(2):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1,2, width_ratios=[4,1],wspace=0.2,subplot_spec=outer_grid[i, j])
        scatter_ax = fig.add_subplot(inner_grid[0])
        marg_ax = fig.add_subplot(inner_grid[1], sharey=scatter_ax)

        plt.setp(marg_ax.get_yticklabels(), visible=False)
        plt.setp(marg_ax.get_yticklabels(minor=True), visible=False)

        species_name = species[i*2+j]
        x,y = get_joint_plot_x_y(species_name)

        xs = np.linspace(0.01, 1, 100)
        ys = -np.log(xs) / config.first_pass_block_size
        scatter_ax.plot(xs, ys, '--r', zorder=1, label='indep. SNPs')

        scatter_ax.scatter(x, y, s=0.6, linewidth=0, zorder=2, rasterized=True)
        marg_ax.hist(y, orientation='horizontal', bins=100, alpha=0.6)

        marg_ax.set_xscale('log')
        if 'Barnesiella' in species_name:
            marg_ax.set_xticks([1, 100])
        else:
            marg_ax.set_xticks([1, 10, 100])

        scatter_ax.set_xlabel('Identical fraction')
        scatter_ax.set_ylabel('Pairwise syn divergence')
        scatter_ax.legend()
        scatter_ax.set_title(figure_utils.get_pretty_species_name(species_name))

fig.savefig(os.path.join(config.figure_directory, 'test_supp_joint.pdf'), bbox_inches='tight', dpi=600)