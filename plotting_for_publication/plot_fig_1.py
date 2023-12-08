import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import default_fig_styles
import config
from utils import snp_data_utils, figure_utils
from utils.typical_pair_utils import get_joint_plot_x_y, load_precomputed_theta, partial_recombination_curve
from scripts.compute_variance_explained_joint_dist import plot_var_exaplained, plot_effective_rbym_from_alpha

dh = None  # only useful when I accidentally deleted the cached snp vectors
species_name = 'Alistipes_putredinis_61533'

fontsize = 5.5
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 0.8
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

highlight_markers = ['^', 's', '*']
sizes = [8, 8, 25]
facecolor='#f4a582'
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
                dh = snp_data_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])
            snp_vec, _ = dh.get_snp_vector(pair)
            np.savetxt(cache_file, snp_vec)
        window_size = 300
        local_pi = np.convolve(snp_vec, np.ones(window_size) / float(window_size), mode='same')

        axes[i].plot(local_pi, color='tab:grey')
        axes[i].plot(np.nonzero(snp_vec)[0], np.zeros(np.sum(snp_vec)), '|', markeredgewidth=0.5, rasterized=True)
        axes[i].scatter(-10, -10, marker=highlight_markers[i], label=' ', s=sizes[i], edgecolors='k', facecolors=facecolor, linewidth=0.5)
        axes[i].set_yticks((0.0, 0.04, 0.08))
        axes[i].set_ylim(-0.001, 0.085)
        #     ax[i].set_xlim([0, 260000])
        axes[i].set_xlim([0, 25000])
        axes[i].legend(loc='upper left', bbox_to_anchor=(-0.05, 1.15))
    axes[2].set_xticklabels([0, 5, 10, 15, 20, 25])
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xlabel("Location along core genome (kb)")
    axes[1].set_ylabel("SNV density")


def plot_cf_pd_joint(axes):
    scatter_ax, marg_ax = axes
    plt.setp(marg_ax.get_yticklabels(), visible=False)
    plt.setp(marg_ax.get_yticklabels(minor=True), visible=False)

    close_pair = (335, 343)
    inter_pair = (5, 270)
    mosaic_pair = (0, 28)
    pairs = [close_pair, inter_pair, mosaic_pair]
    clonal_frac_dir = os.path.join(config.analysis_directory, 'pairwise_clonal_fraction',
                                   'between_hosts', '%s.csv' % species_name)
    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'between_hosts', '%s.csv' % species_name)
    cf_mat = np.loadtxt(clonal_frac_dir, delimiter=',')
    pd_mat = np.loadtxt(div_dir, delimiter=',')

    x,y = get_joint_plot_x_y(species_name)

    theta = load_precomputed_theta(species_name)
    F = partial_recombination_curve(x, y, theta=theta)
    xs = np.linspace(0., 1, 100)
    ys = F(xs)
    scatter_ax.plot(xs, ys, '--', color='tab:red', zorder=2, label='partial recomb.')

    xs = np.linspace(0.01, 1, 100)
    ys = -np.log(xs) / config.first_pass_block_size
    scatter_ax.plot(xs, ys, '--', color='grey', zorder=1, label='random mut\'s')

    scatter_ax.scatter(x, y, s=0.6, alpha=0.8, linewidth=0, zorder=2, rasterized=True)
    for i, pair in enumerate(pairs):
        x_, y_ = cf_mat[pair], pd_mat[pair]
        # highlight pairs
        scatter_ax.scatter(x_, y_, marker=highlight_markers[i], s=sizes[i]+8, zorder=3, edgecolors='k', facecolors=facecolor, linewidth=0.5)
    marg_ax.hist(y, orientation='horizontal', bins=100, alpha=0.6)

    # marg_ax.set_xscale('log')
    marg_ax.set_xticks([0, 500, 1000])
    marg_ax.minorticks_on()
    marg_ax.yaxis.set_tick_params(which='minor', bottom=False)
    scatter_ax.yaxis.set_tick_params(which='minor', bottom=False)
    marg_ax.set_xticklabels(['0', '500', '1k'])

    scatter_ax.set_xlabel('Fraction of identical blocks')
    scatter_ax.set_ylabel('Pairwise syn divergence (%)')
    scatter_ax.set_yticks([0, 0.5e-2, 1e-2, 1.5e-2, 2e-2, 2.5e-2])
    scatter_ax.set_yticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
    scatter_ax.legend()
    scatter_ax.set_title(figure_utils.get_pretty_species_name(species_name))
    return x, y


def plot_neutral_joint(scatter_ax, marg_ax):
    plt.setp(marg_ax.get_yticklabels(), visible=False)
    plt.setp(marg_ax.get_yticklabels(minor=True), visible=False)
    data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'for_pileup')
    exp_df = pd.read_csv(os.path.join(data_dir, 'b_vulgatus', 'experiments.txt'), sep=' ')
    exp_df = exp_df.set_index(['rbymu', 'lambda', 'rep'])
    rbymus = [0.1, 0.5, 2]
    lambs = [500, 1000, 2000]
    i = 1
    j = 2
    sim_id = int(exp_df.loc[(rbymus[i], lambs[j], 1), 'sim_id'])
    # preparing x&y
    pd_mat = np.loadtxt(os.path.join(data_dir, 'b_vulgatus', 'pd', 'pd_%d.csv' % sim_id))
    cf_mat = np.loadtxt(os.path.join(data_dir, 'b_vulgatus', 'cf', 'cf_%d.csv' % sim_id))
    x = cf_mat[np.triu_indices(cf_mat.shape[0], 1)]
    y = pd_mat[np.triu_indices(pd_mat.shape[0], 1)]

    xs = np.linspace(0.01, 1, 100)
    ys = -np.log(xs) / config.first_pass_block_size
    scatter_ax.plot(xs, ys, '--', color='grey', zorder=1, label='random mut\'s', linewidth=0.8)
    F, alpha = partial_recombination_curve(x, y, theta=None, return_alpha=True)
    x_save = x
    y_save = y
    xs = np.linspace(0., 1, 100)
    ys = F(xs)
    scatter_ax.plot(xs, ys, '--', color='tab:red', zorder=2, label='partial recomb.', linewidth=0.8)
    print(rbymus[i] * lambs[j], 1 / alpha - 1)

    scatter_ax.scatter(x, y, s=0.3, linewidth=0, zorder=2, rasterized=True, color='tab:grey')
    marg_ax.hist(y, orientation='horizontal', bins=100, alpha=0.6, color='tab:grey')

    scatter_ax.set_yticks([0, 0.5e-2, 1e-2])
    scatter_ax.set_yticklabels(['0', '0.5', '1'])
    marg_ax.set_xticks([0, 1000])
    marg_ax.set_xticklabels(['0', '1k'])
    marg_ax.minorticks_on()

    scatter_ax.set_title("Neutral simulation")
    scatter_ax.set_ylabel('Pairwise divergence (%)')
    scatter_ax.set_xlabel('Fraction of identical blocks')
    return x_save, y_save


#
# fig, axes = plt.subplots(3, 1, figsize=(4, 3))
# plt.subplots_adjust(hspace=0.2)
# plot_example_genomes(axes)
# fig.savefig(os.path.join(config.figure_directory,
#                          "genome_divergence_examples_A_putredinis.pdf"),
#             bbox_inches='tight', dpi=1200)


fig = plt.figure(figsize=(7, 5.5))
outer_grid = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[1.7, 1.7, 1.8], hspace=0.6, figure=fig)

mid_grid = gridspec.GridSpecFromSubplotSpec(ncols=3, nrows=1, width_ratios=[2.5,2.,2.5],wspace=0.4,subplot_spec=outer_grid[1])
Ap_joint_plot_grid = gridspec.GridSpecFromSubplotSpec(1,2, width_ratios=[4,1],wspace=0.1,subplot_spec=mid_grid[0])
neutral_joint_plot_grid=gridspec.GridSpecFromSubplotSpec(1,2, width_ratios=[4,1],wspace=0.1,subplot_spec=mid_grid[2])
example_grid = gridspec.GridSpecFromSubplotSpec(ncols=1,nrows=3, height_ratios=[1,1,1],hspace=0.2,subplot_spec=mid_grid[1])
bottom_grid = gridspec.GridSpecFromSubplotSpec(1, 3, width_ratios=[0.5, 6, 0.5], hspace=0., subplot_spec=outer_grid[2])
bottom_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 0.4], hspace=0., subplot_spec=bottom_grid[1])


# adding axes
cartoon_ax = fig.add_subplot(outer_grid[0])
cartoon_ax.set_yticklabels([])
cartoon_ax.set_xticklabels([])
# cartoon_ax.set_visible(False)
scatter_ax = fig.add_subplot(Ap_joint_plot_grid[0])
marg_ax = fig.add_subplot(Ap_joint_plot_grid[1], sharey=scatter_ax)
neutral_scatter_ax = fig.add_subplot(neutral_joint_plot_grid[0])
neutral_marg_ax = fig.add_subplot(neutral_joint_plot_grid[1], sharey=neutral_scatter_ax)

example_ax1 = fig.add_subplot(example_grid[0])
example_ax2 = fig.add_subplot(example_grid[1])
example_ax3 = fig.add_subplot(example_grid[2])

var_exp_ax1 = fig.add_subplot(bottom_grid[0])
var_exp_ax2 = fig.add_subplot(bottom_grid[1])

# panel c
x, y = plot_cf_pd_joint([scatter_ax, marg_ax])
figure_utils.save_figure_data([x, y], ['x', 'y'], config.figure_data_directory, 'fig1/1c')

# panel d
plot_example_genomes([example_ax1, example_ax2, example_ax3])

# panel e
x, y = plot_neutral_joint(neutral_scatter_ax, neutral_marg_ax)
figure_utils.save_figure_data([x, y], ['x', 'y'], config.figure_data_directory, 'fig1/1e')

# panel f
# plot_var_exaplained([var_exp_ax1, var_exp_ax2], plot_only_y=True)
plot_effective_rbym_from_alpha(var_exp_ax1)

fig.delaxes(var_exp_ax2)

fig.savefig(os.path.join(config.figure_directory, 'final_fig', 'fig1.pdf'), bbox_inches='tight', dpi=1200)
