import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import close_pair_utils, parallel_utils
import config

def plot_one_species(x, y, asexual_line=True, same_ylim=False):
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
        ax_joint.plot(xs, ys, '--r', zorder=1)

    ax_joint.scatter(x, y, s=1, zorder=2)
    ax_marg_x.hist(x, bins=100, alpha=0.6)
    ax_marg_y.hist(y, orientation='horizontal', bins=100, alpha=0.6)

    if same_ylim:
        ax_joint.set_ylim([-1e-3, 0.05])
    else:
        ax_joint.set_ylim([-1e-3, max(y) + 1e-3])
    ax_joint.set_xlim([-1e-2, 1 + 1e-2])

    ax_marg_x.set_yscale('log')
    ax_marg_y.set_xscale('log')

    ax_joint.set_xlabel('Clonal fraction')
    ax_joint.set_ylabel('Pairwise divergence')
    return f, (ax_joint, ax_marg_x, ax_marg_y)


def fit_polynomial(x, y, threshold=0.2, deg=2):
    np.std(y[x >= threshold])
    xfit = x[x >= threshold]
    yfit = y[x >= threshold]
    params = np.polyfit(xfit, yfit, deg)
    res = params[0] * xfit ** 2 + params[1] * xfit + params[2] - yfit
    return params, res


base_dir = 'zarr_snps'
for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
    if species_name.startswith('.'):
        continue

    clonal_frac_dir = os.path.join(config.analysis_directory, 'pairwise_clonal_fraction',
                                   'between_hosts', '%s.csv' % species_name)
    clonal_frac_mat = np.loadtxt(clonal_frac_dir, delimiter=',')
    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'between_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')

    x = clonal_frac_mat[np.triu_indices(clonal_frac_mat.shape[0], 1)]
    y = div_mat[np.triu_indices(div_mat.shape[0], 1)]
    save_path = os.path.join(config.analysis_directory, 'clonal_frac_pairwise_div_joint',
                             'same_ylim', '{}.pdf'.format(species_name))
    f, axes = plot_one_species(x, y, asexual_line=True, same_ylim=True)

    f.savefig(save_path)
    plt.close()
