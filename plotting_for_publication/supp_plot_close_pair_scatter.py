import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import config
from utils import close_pair_utils, figure_utils

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'


fig1, axes1 = plt.subplots(5, 3, figsize=(7,9))
plt.subplots_adjust(wspace=0.25, hspace=0.6)
fig2, axes2 = plt.subplots(5, 3, figsize=(7,9))
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
    'Oscillibacter_sp_60799',
    'Akkermansia_muciniphila_55290'
]

data_dir = os.path.join(config.analysis_directory, "closely_related")
for i in range(5):
    for j in range(3):
        idx = 3 * i + j
        if idx >= len(species_to_plot):
            break
        species = species_to_plot[idx]
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
        ax.set_xlim([0, max(x)*1.05])
        ax.set_ylim([0, max(y)*1.05])
        ax.set_title(figure_utils.get_pretty_species_name(species))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        # then plot the transfer length scatter
        ax = axes2[i, j]
        x, y = close_pair_utils.prepare_x_y(data, mode='fraction', cf_cutoff=0.5)
        ax.scatter(x, y, s=2)
        ax.set_xlim([0, max(x)*1.05])
        ax.set_ylim([0, max(y)*1.05])
        ax.set_title(figure_utils.get_pretty_species_name(species))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

for ax in axes1[:, 0]:
    ax.set_ylabel("Num transfer per 1Mbps")
for ax in axes1[-1, :]:
    ax.set_xlabel("Syn clonal divergence")
for ax in axes2[:, 0]:
    ax.set_ylabel("Total transfer length")
for ax in axes2[-1, :]:
    ax.set_xlabel("Syn clonal divergence")

fig1.delaxes(axes1[-1][-1])
fig2.delaxes(axes2[-1][-1])
fig1.savefig(os.path.join(config.analysis_directory, 'closely_related', 'supp_grid_counts.pdf'), bbox_inches='tight')
fig2.savefig(os.path.join(config.analysis_directory, 'closely_related', 'supp_grid_lengths.pdf'), bbox_inches='tight')
