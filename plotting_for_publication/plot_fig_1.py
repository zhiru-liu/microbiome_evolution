import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import default_fig_styles
import config
from utils import parallel_utils, figure_utils
from utils.typical_pair_utils import get_joint_plot_x_y, load_precomputed_theta, partial_recombination_curve
from HGT_scripts.compute_variance_explained_joint_dist import plot_var_exaplained

dh = None  # only useful when I accidentally deleted the cached snp vectors
species_name = 'Alistipes_putredinis_61533'

def plot_example_genomes(axes):
    # pairs = [(43, 96), (0, 251), (371, 11)]
    # Example pairs chosen for A. putredinis
    close_pair = (335, 343)
    inter_pair = (5, 270)
    mosaic_pair = (0, 28)
    pairs = [close_pair, inter_pair, mosaic_pair]

    for i in range(3):
        pair = pairs[i]
        cache_file = os.path.join(config.plotting_intermediate_directory, "A_putredinis_cached_close_pair_{}.csv".format(pair))
        if os.path.exists(cache_file):
            snp_vec = np.loadtxt(cache_file).astype(bool)
        else:
            if dh is None:
                dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])
            snp_vec, _ = dh.get_snp_vector(pair)
            np.savetxt(cache_file, snp_vec)
        window_size = 300
        local_pi = np.convolve(snp_vec, np.ones(window_size) / float(window_size), mode='same')

        axes[i].plot(local_pi, color='tab:grey')
        axes[i].plot(np.nonzero(snp_vec)[0], np.zeros(np.sum(snp_vec)), '|', markeredgewidth=0.5, rasterized=True)
        axes[i].set_yticks((0.0, 0.04, 0.08))
        axes[i].set_ylim(-0.001, 0.085)
        #     ax[i].set_xlim([0, 260000])
        axes[i].set_xlim([0, 25000])
    axes[2].set_xticklabels([0, 5, 10, 15, 20, 25])
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xlabel("Location along core genome (kb)")
    axes[1].set_ylabel("SNV density")


def plot_cf_pd_joint(axes):
    scatter_ax, marg_ax = axes
    plt.setp(marg_ax.get_yticklabels(), visible=False)
    plt.setp(marg_ax.get_yticklabels(minor=True), visible=False)

    x,y = get_joint_plot_x_y(species_name)

    xs = np.linspace(0.01, 1, 100)
    ys = -np.log(xs) / config.first_pass_block_size
    scatter_ax.plot(xs, ys, '--r', zorder=1, label='random mut\'s')

    theta = load_precomputed_theta(species_name)
    F = partial_recombination_curve(x, y, theta=theta)
    xs = np.linspace(0., 1, 100)
    ys = F(xs)
    scatter_ax.plot(xs, ys, '-.', color='tab:orange', zorder=2, label='partial recomb.')

    scatter_ax.scatter(x, y, s=0.6, linewidth=0, zorder=2, rasterized=True)
    marg_ax.hist(y, orientation='horizontal', bins=100, alpha=0.6)

    # marg_ax.set_xscale('log')
    marg_ax.set_xticks([1, 500, 1000])
    marg_ax.set_xticklabels(['1', '500', '1k'])

    scatter_ax.set_xlabel('Fraction of identical blocks')
    scatter_ax.set_ylabel('Pairwise syn divergence (%)')
    scatter_ax.set_yticks([0, 0.5e-2, 1e-2, 1.5e-2, 2e-2, 2.5e-2])
    scatter_ax.set_yticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
    scatter_ax.legend()
    scatter_ax.set_title(figure_utils.get_pretty_species_name(species_name))


#
# fig, axes = plt.subplots(3, 1, figsize=(4, 3))
# plt.subplots_adjust(hspace=0.2)
# plot_example_genomes(axes)
# fig.savefig(os.path.join(config.figure_directory,
#                          "genome_divergence_examples_A_putredinis.pdf"),
#             bbox_inches='tight', dpi=1200)


fig = plt.figure(figsize=(6, 6.5))
outer_grid = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[2.5, 2.5, 2.5], hspace=0.5, figure=fig)

mid_grid = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, width_ratios=[1,1],wspace=0.3,subplot_spec=outer_grid[1])
joint_plot_grid = gridspec.GridSpecFromSubplotSpec(1,2, width_ratios=[4,1],wspace=0.1,subplot_spec=mid_grid[0])
example_grid = gridspec.GridSpecFromSubplotSpec(ncols=1,nrows=3, height_ratios=[1,1,1],hspace=0.2,subplot_spec=mid_grid[1])

bottom_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 0.4], hspace=0., subplot_spec=outer_grid[2])


# adding axes
cartoon_ax = fig.add_subplot(outer_grid[0])
cartoon_ax.set_yticklabels([])
cartoon_ax.set_xticklabels([])
# cartoon_ax.set_visible(False)
scatter_ax = fig.add_subplot(joint_plot_grid[0])
marg_ax = fig.add_subplot(joint_plot_grid[1], sharey=scatter_ax)

example_ax1 = fig.add_subplot(example_grid[0])
example_ax2 = fig.add_subplot(example_grid[1])
example_ax3 = fig.add_subplot(example_grid[2])

var_exp_ax1 = fig.add_subplot(bottom_grid[0])
var_exp_ax2 = fig.add_subplot(bottom_grid[1])

plot_cf_pd_joint([scatter_ax, marg_ax])
plot_example_genomes([example_ax1, example_ax2, example_ax3])
plot_var_exaplained([var_exp_ax1, var_exp_ax2], plot_only_y=True)
fig.delaxes(var_exp_ax2)


scatter_ax.text(-0.1, 1.12, "B", transform=scatter_ax.transAxes,
         fontsize=9, fontweight='bold', va='top', ha='left')
example_ax1.text(-0.1, 1.45, "C", transform=example_ax1.transAxes,
         fontsize=9, fontweight='bold', va='top', ha='left')
var_exp_ax1.text(-0.1, 1.12, "D", transform=var_exp_ax1.transAxes,
         fontsize=9, fontweight='bold', va='top', ha='left')

fig.savefig(os.path.join(config.figure_directory, 'final_fig', 'fig1.pdf'), bbox_inches='tight', dpi=1200)
