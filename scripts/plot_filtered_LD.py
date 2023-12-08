import os
import config
import matplotlib.pyplot as plt
import numpy as np
from utils import figure_utils, linkage_utils
import GarudGood2019_scripts.calculate_linkage_disequilibria

def plot_species(species_name):
    fig, example_axis = plt.subplots(figsize=(4, 3))

    figdir = os.path.join(config.analysis_directory, "LD")
    ld_map = GarudGood2019_scripts.calculate_linkage_disequilibria.load_ld_map(species_name, cf=0.0)
    cf_ld_map = GarudGood2019_scripts.calculate_linkage_disequilibria.load_ld_map(species_name, cf=0.1)

    # set up axis
    example_axis.set_ylabel('Linkage disequilibrium, $\sigma^2_d$')
    example_axis.set_xlabel('Distance between SNVs, $\ell$')

    example_axis.spines['top'].set_visible(False)
    example_axis.spines['right'].set_visible(False)
    example_axis.spines['bottom'].set_zorder(22)

    example_axis.get_xaxis().tick_bottom()
    example_axis.get_yaxis().tick_left()

    example_axis.set_xlim([2, 1e04])
    example_axis.set_ylim([1e-02, 1])

    example_axis.text(6e03, 4e-03, 'Genome-\nwide', horizontalalignment='center')
    example_axis.set_title(figure_utils.get_pretty_species_name(species_name), y=0.95)

    # plotting
    def plot_axis(example_axis, ld_map, color, all_style, legend_prefix=''):
        distances, rsquareds, control_rsquared, all_distances, all_rsquareds, all_control_rsquared, early_distances, early_rsquareds = linkage_utils.prepare_LD(
            ld_map)
        example_axis.loglog(all_distances, all_rsquareds, all_style, color='0.7', label=legend_prefix + 'All samples')
        example_axis.fill_between(np.array([3.3e03, 1e04]), np.array([1e-02, 1e-02]), np.array([1, 1]), color='w',
                                  zorder=20)

        example_axis.loglog([all_distances[-1], 6e03], [all_rsquareds[-1], all_control_rsquared], ':', color='0.7',
                            zorder=21)
        example_axis.loglog([6e03], [all_control_rsquared], 'o', color='0.7', markersize=3, markeredgewidth=0,
                            zorder=21)
        # example_axis.fill_between(distances[good_distances],lower_rsquareds[good_distances], upper_rsquareds[good_distances], linewidth=0, color=color,alpha=0.5)
        example_axis.loglog(distances, rsquareds, '-', color=color, label=legend_prefix + 'Largest clade')
        example_axis.loglog(early_distances, early_rsquareds, 'o', color=color, markersize=2, markeredgewidth=0,
                            alpha=0.5)

        example_axis.loglog([distances[-1], 6e03], [rsquareds[-1], control_rsquared], ':', color=color, zorder=21)
        example_axis.loglog([6e03], [control_rsquared], 'o', color=color, markersize=3, markeredgewidth=0, zorder=21)

        line, = example_axis.loglog([distances[-1], distances[-1]], [1e-02, 1], ':', color='0.5')

    def plot_null(example_axis, ld_map):
        distances, rsquareds, control_rsquared, all_distances, all_rsquareds, all_control_rsquared, early_distances, early_rsquareds = linkage_utils.prepare_LD(
            ld_map)
        theory_ls = np.logspace(0, np.log10(distances[-1]), 100)
        theory_NRs = theory_ls / 200.0
        theory_rsquareds = (10 + 2 * theory_NRs) / (22 + 26 * theory_NRs + 4 * theory_NRs * theory_NRs)
        example_axis.loglog(theory_ls, theory_rsquareds / theory_rsquareds[0] * 3e-01, 'k-', linewidth=0.3, zorder=0,
                            label='Neutral')

    plot_axis(example_axis, ld_map, color='tab:blue', all_style='-')
    plot_axis(example_axis, cf_ld_map, color='tab:orange', all_style='--', legend_prefix='CF ')
    plot_null(example_axis, ld_map)

    example_axis.xaxis.get_major_ticks()[-2].label1.set_visible(False)
    example_axis.xaxis.get_major_ticks()[-2].tick1line.set_visible(False)
    minorticks = example_axis.xaxis.get_minor_ticks()
    for tick_idx in xrange(len(minorticks)):
        if example_axis.xaxis.get_minorticklocs()[tick_idx] > 3000:
            minorticks[tick_idx].tick1line.set_visible(False)
            minorticks[tick_idx].tick2line.set_visible(False)

    example_axis.legend(prop={'size': 6})
    fig.savefig(os.path.join(figdir, "%s.pdf"%species_name), bbox_inches='tight')
    plt.close()


for species in os.listdir(os.path.join(config.data_directory, 'linkage_disequilibria', 'cf_0.1')):
    if species.startswith('.'):
        continue
    else:
        species_name = species.split('.')[0]
        plot_species(species_name)
