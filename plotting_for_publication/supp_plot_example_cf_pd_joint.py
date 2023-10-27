import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import config
from utils import figure_utils
from utils.typical_pair_utils import get_joint_plot_x_y, load_precomputed_theta, partial_recombination_curve

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'


fig = plt.figure(figsize=(6, 6.))
outer_grid = gridspec.GridSpec(3,2, width_ratios=[1,1],wspace=0.3,figure=fig)
plt.subplots_adjust(wspace=0.25, hspace=0.6)

ex1 = 'Prevotella_copri_61740'  # lack of close pairs
ex2 = 'Lachnospiraceae_bacterium_51870' # lack of typical pairs
# ex2 = 'Roseburia_inulinivorans_61943'  # lack of close pairs
ex3 = 'Barnesiella_intestinihominis_62208'  # minimal spread in joint plot
ex4 = 'Bacteroides_caccae_53434' # uneven distribution along cf
ex5 = 'Bacteroides_vulgatus_57955'  # population structure
ex6 = 'Eubacterium_rectale_56927'  # population structure

species = [ex1, ex2, ex3, ex4, ex5, ex6]

for i in range(3):
    for j in range(2):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1,2, width_ratios=[4,1],wspace=0.2,subplot_spec=outer_grid[i, j])
        scatter_ax = fig.add_subplot(inner_grid[0])
        marg_ax = fig.add_subplot(inner_grid[1], sharey=scatter_ax)

        plt.setp(marg_ax.get_yticklabels(), visible=False)
        plt.setp(marg_ax.get_yticklabels(minor=True), visible=False)

        species_name = species[i*2+j]
        x,y = get_joint_plot_x_y(species_name)

        theta = load_precomputed_theta(species_name)
        F = partial_recombination_curve(x, y, theta=theta)
        xs = np.linspace(0.01, 1, 100)
        ys = F(xs)
        scatter_ax.plot(xs, ys*100, '--', color='tab:red', zorder=2, label='partial recomb.')

        y = y * 100

        xs = np.linspace(0.01, 1, 100)
        ys = -np.log(xs) / config.first_pass_block_size * 100
        scatter_ax.plot(xs, ys, '--', color='grey', zorder=1, label='random mutations')

        scatter_ax.scatter(x, y, s=0.6, linewidth=0, zorder=2, rasterized=True)
        marg_ax.hist(y, orientation='horizontal', bins=100, alpha=0.6)
        marg_ax.minorticks_on()
        marg_ax.yaxis.set_tick_params(which='minor', bottom=False)
        scatter_ax.yaxis.set_tick_params(which='minor', bottom=False)

        # marg_ax.set_xscale('log')
        # if ('copri' in species_name) or ('Roseburia' in species_name) or ('Lachnospiraceae' in species_name):
        #     marg_ax.set_xticks([1, 10, 100])
        # else:
        #     marg_ax.set_xticks([1, 100])

        scatter_ax.set_xlabel('Fraction of identical blocks')
        scatter_ax.set_ylabel('Pairwise syn divergence (%)')
        scatter_ax.legend()
        scatter_ax.set_title(figure_utils.get_pretty_species_name(species_name))

fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_joint_examples.pdf'), bbox_inches='tight', dpi=600)

# now plot all other species
fig = plt.figure(figsize=(18, 12.))
outer_grid = gridspec.GridSpec(6,6, width_ratios=[1,1,1,1,1,1],wspace=0.3,figure=fig)
plt.subplots_adjust(wspace=0.25, hspace=0.6)
count = 0
all_species = [x for x in os.listdir(os.path.join(config.data_directory, 'zarr_snps')) if not x.startswith('.')]
plotted_species = ['Alistipes_putredinis_61533', ex1, ex2, ex3, ex4, ex5, ex6]
for i in range(6):
    for j in range(6):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1,2, width_ratios=[4,1],wspace=0.2,subplot_spec=outer_grid[i, j])
        scatter_ax = fig.add_subplot(inner_grid[0])
        marg_ax = fig.add_subplot(inner_grid[1], sharey=scatter_ax)

        plt.setp(marg_ax.get_yticklabels(), visible=False)
        plt.setp(marg_ax.get_yticklabels(minor=True), visible=False)

        species_name = all_species[count]
        while species_name in plotted_species:
            count += 1
            species_name = all_species[count]
        print("plotting {}".format(species_name))
        x,y = get_joint_plot_x_y(species_name)

        theta = load_precomputed_theta(species_name)
        F = partial_recombination_curve(x, y, theta=theta)
        xs = np.linspace(0.01, 1, 100)
        ys = F(xs)
        scatter_ax.plot(xs, ys*100, '--', color='tab:red', zorder=2, label='partial recomb.')

        y = y * 100

        xs = np.linspace(0.01, 1, 100)
        ys = -np.log(xs) / config.first_pass_block_size * 100
        scatter_ax.plot(xs, ys, '--', color='grey', zorder=1, label='random mutations')

        scatter_ax.scatter(x, y, s=0.6, linewidth=0, zorder=2, rasterized=True)
        marg_ax.hist(y, orientation='horizontal', bins=100, alpha=0.6)

        marg_ax.minorticks_on()
        marg_ax.yaxis.set_tick_params(which='minor', bottom=False)
        scatter_ax.yaxis.set_tick_params(which='minor', bottom=False)

        # marg_ax.set_xscale('log')
        # if ('copri' in species_name) or ('Roseburia' in species_name) or ('Lachnospiraceae' in species_name):
        #     marg_ax.set_xticks([1, 10, 100])
        # else:
        #     marg_ax.set_xticks([1, 100])

        if i==5:
            scatter_ax.set_xlabel('Fraction of identical blocks')
        if j==0:
            scatter_ax.set_ylabel('Pairwise syn divergence (%)')
        if i==0 and j==0:
            scatter_ax.legend()
        scatter_ax.set_title(figure_utils.get_pretty_species_name(species_name))
        count += 1

fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_all_joint_plots.pdf'), bbox_inches='tight', dpi=600)