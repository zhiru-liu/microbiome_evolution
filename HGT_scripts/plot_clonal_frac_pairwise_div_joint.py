import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from utils.typical_pair_utils import get_joint_plot_x_y, fit_quadratic_curve, partial_recombination_curve, load_precomputed_theta

sys.path.append("..")
import config


def plot_one_species(x, y, species_name, asexual_line=True, fit_line=True, theory_line=True, same_ylim=None, logscale=True, semilogy=False):
    height = 6
    ratio = 5

    f = plt.figure(figsize=(height, height))
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    ax_joint = f.add_subplot(gs[1:, :-1])
    ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)

    if asexual_line:
        xs = np.linspace(0.05, 1, 100)
        ys = -np.log(xs) / config.first_pass_block_size
        ax_joint.plot(xs, ys, '--r', zorder=2, label='random mutations')

    if fit_line:
        F = fit_quadratic_curve(x, y)
        xs = np.linspace(0., 1, 100)
        ys = F(xs)
        ax_joint.plot(xs, ys, '--', color='orange', zorder=2, label='fit')

    if theory_line:
        theta = load_precomputed_theta(species_name)
        F = partial_recombination_curve(x, y, theta=theta)
        xs = np.linspace(0., 1, 100)
        ys = F(xs)
        ax_joint.plot(xs, ys, '--', color='tab:orange', zorder=2, label='partial recombination')


    ax_joint.scatter(x, y, s=1, zorder=2, rasterized=True)
    ax_marg_x.hist(x, bins=100, alpha=0.6)
    if semilogy:
        bins = np.geomspace(min(y[y>0]), max(y), 100)
    else:
        bins = 100
    ax_marg_y.hist(y, orientation='horizontal', bins=bins, alpha=0.6)

    if logscale:
        ax_marg_x.set_yscale('log')
        ax_marg_y.set_xscale('log')

    if semilogy:
        ax_joint.set_yscale('log')
    else:
        if same_ylim is not None:
            ax_joint.set_ylim([-1e-3, same_ylim])
        else:
            ax_joint.set_ylim([-1e-3, max(y) + 1e-3])
    ax_joint.set_xlim([-1e-2, 1 + 1e-2])

    ax_joint.set_xlabel('Identical fraction')
    ax_joint.set_ylabel('Pairwise divergence')
    ax_marg_y.set_xticks([10, 1000])
    ax_joint.legend()
    return f, (ax_joint, ax_marg_x, ax_marg_y)


if __name__ == "__main__":
    for species_name in os.listdir(os.path.join(config.data_directory, 'zarr_snps')):
    # for species_name in ['MGYG-HGUT-02478']:
        if species_name.startswith('.'):
            continue
        # if 'vulgatus' not in species_name:
        #     # plotting specific species
        #     continue
        x, y = get_joint_plot_x_y(species_name)

        save_path = os.path.join(config.analysis_directory, 'clonal_frac_pairwise_div_joint',
                                 'theory_curve', '{}.pdf'.format(species_name))
        f, axes = plot_one_species(x, y, species_name, asexual_line=True, fit_line=True, same_ylim=None, semilogy=False)

        f.savefig(save_path, dpi=600)
        plt.close()
