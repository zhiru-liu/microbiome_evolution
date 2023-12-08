import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import config
from utils import close_pair_utils, figure_utils

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'


fig1, axes1 = plt.subplots(5, 3, figsize=(7,8))
plt.subplots_adjust(wspace=0.25, hspace=0.6)
fig2, axes2 = plt.subplots(5, 3, figsize=(7,8))
plt.subplots_adjust(wspace=0.25, hspace=0.6)

species_to_plot = [
    'Bacteroides_uniformis_57318',
    'Bacteroides_stercoris_56735',
    'Bacteroides_caccae_53434',
    'Bacteroides_ovatus_58035',
    'Bacteroides_thetaiotaomicron_56941',
    'Bacteroides_massiliensis_44749',
    'Bacteroides_cellulosilyticus_58046',
    'Parabacteroides_merdae_56972',
    'Parabacteroides_distasonis_56985',
    'Barnesiella_intestinihominis_62208',
    'Alistipes_putredinis_61533',
    'Alistipes_onderdonkii_55464',
    'Alistipes_sp_60764',
    'Oscillibacter_sp_60799',
    'Akkermansia_muciniphila_55290'
]

# hand annotated cutoff for x range according to total recombined fraction
clonal_div_cutoff = [
    None,
    2.e-4,
    1.e-4,
    2.2e-4,
    None,
    None,
    1.1e-4,
    1.2e-4,
    None,
    None,
    2.e-4,
    1.5e-4,
    None,
    None,
    1.2e-4
]
species_cutoff_dict = dict(zip(species_to_plot, clonal_div_cutoff))

data_dir = os.path.join(config.analysis_directory, "closely_related")
for i in range(5):
    for j in range(3):
        idx = 3 * i + j
        if idx >= len(species_to_plot):
            break
        species = species_to_plot[idx]
        div_cutoff = species_cutoff_dict[species]
        data = pd.read_pickle(os.path.join(data_dir, 'third_pass', species + '.pickle'))
        # plot the transfer count scatter
        ax = axes1[i, j]
        fit_data = pd.read_csv(os.path.join(config.analysis_directory,
                                            "closely_related", "fourth_pass", "{}.csv".format(species)))
        x, y = close_pair_utils.prepare_x_y(data)
        x_plot, y_plot, sigmas = fit_data['x'], fit_data['y'], fit_data['sigma']

        ax.scatter(x, y, s=2)
        ax.plot(x_plot, y_plot, '--', color='tab:orange')
        ax.fill_between(x_plot, y_plot - sigmas,
                        y_plot + sigmas, alpha=0.25)
        if div_cutoff is not None:
            ax.set_xlim([0, species_cutoff_dict[species]])
        else:
            ax.set_xlim([0, max(x)*1.05])
        ax.set_ylim([0, max(y)*1.05])
        ax.set_title(figure_utils.get_pretty_species_name(species))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        # then plot the transfer length scatter
        ax = axes2[i, j]
        fit_data = pd.read_csv(os.path.join(config.analysis_directory,
                                            "closely_related", "fourth_pass", "{}_length.csv".format(species)))
        x_plot, y_plot, sigmas = fit_data['x'], fit_data['y'], fit_data['sigma']

        x, y, mask = close_pair_utils.prepare_x_y(data, mode='fraction', return_unfiltered=True)
        ax.scatter(x, y, s=2)
        ax.plot(x_plot, y_plot, '--', color='tab:orange')
        ax.fill_between(x_plot, y_plot - sigmas,
                        y_plot + sigmas, alpha=0.25)
        max_x, max_y = max(x), max(y)
        if div_cutoff is not None:
            ax.add_patch(Rectangle((species_cutoff_dict[species], 0), 4e-4, 1, facecolor='grey', edgecolor=None, alpha=0.3))
        ax.set_xlim([0, max(x)*1.05])
        ax.set_ylim([0, max(y)*1.05])
        ax.set_title(figure_utils.get_pretty_species_name(species))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

for ax in axes1[:, 0]:
    ax.set_ylabel("Num transfer per Mb")
for ax in axes1[-1, :]:
    ax.set_xlabel("Syn clonal divergence")
for ax in axes2[:, 0]:
    ax.set_ylabel("Recombined fraction")
for ax in axes2[-1, :]:
    ax.set_xlabel("Syn clonal divergence")

fig1.savefig(os.path.join(config.figure_directory, 'supp', 'S16_supp_grid_counts.pdf'), bbox_inches='tight')
fig2.savefig(os.path.join(config.figure_directory, 'supp', 'S17_supp_grid_lengths.pdf'), bbox_inches='tight')
json.dump(species_cutoff_dict, open(os.path.join(config.plotting_intermediate_directory, 'clonal_div_cutoff.json'), 'w'))
