import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import config
from utils import typical_pair_utils, figure_utils


def prepare_var_explained_data():
    # manually controlling species with apparent multi clade structure
    simple_pop_struct_dict = {'Bacteroides_vulgatus_57955': 0.03,
                              'Alistipes_shahii_62199': 0.03,
                              'Bacteroides_uniformis_57318': 0.03,
                              'Ruminococcus_bromii_62047': 0.055,
                              'Roseburia_intestinalis_56239': 0.03,
                              'Bacteroides_cellulosilyticus_58046': 0.04,
                              'Alistipes_finegoldii_56071': 0.025,
                              'Bacteroides_ovatus_58035': 0.035}

    bin_num = 500
    plot_only = True

    if not plot_only:
        # first fitting all species regularly
        all_species = []
        all_var = []
        all_var_fit = []
        all_var_asex = []

        all_var_wt = []
        all_var_fit_wt = []
        all_var_asex_wt = []

        for species_name in os.listdir(os.path.join(config.data_directory, 'zarr_snps')):
            if species_name.startswith('.'):
                continue
            x, y = typical_pair_utils.get_joint_plot_x_y(species_name)
            F = typical_pair_utils.fit_quadratic_curve(x, y)
            y_fit = y - F(x)
            y_asex = y - typical_pair_utils.asexual_curve(x, default=y[x == 0].mean())

            # obtain the weighting by density in y direction
            counts, bins = np.histogram(y, bins=bin_num)
            # mapping values to bins
            locs = np.digitize(y, bins)
            locs[locs == len(bins)] = len(bins) - 1  # adjust y.max() to the right most bin
            locs -= 1  # so that can index counts
            weights = np.reciprocal(counts[locs].astype(float))

            all_species.append(species_name)
            all_var.append(np.var(y))
            all_var_fit.append(np.mean(y_fit ** 2))
            all_var_asex.append(np.mean(y_asex ** 2))

            all_var_wt.append(np.sum((y - np.mean(y))**2 * weights) / np.sum(weights))
            all_var_fit_wt.append(np.sum(y_fit**2 * weights) / np.sum(weights))
            all_var_asex_wt.append(np.sum(y_asex**2 * weights) / np.sum(weights))

        # then process species with easy population structure (taking only the largest clade)
        all_var_ct = []
        all_var_fit_ct = []
        all_var_asex_ct = []

        all_var_ct_wt = []
        all_var_fit_ct_wt = []
        all_var_asex_ct_wt = []

        for species_name in simple_pop_struct_dict:
            x, y = typical_pair_utils.get_joint_plot_x_y(species_name, clade_cutoff=simple_pop_struct_dict[species_name])
            F = typical_pair_utils.fit_quadratic_curve(x, y)
            y_fit = y - F(x)
            y_asex = y - typical_pair_utils.asexual_curve(x, default=y[x == 0].mean())

            # obtain the weighting by density in y direction
            counts, bins = np.histogram(y, bins=bin_num)
            # mapping values to bins
            locs = np.digitize(y, bins)
            locs[locs == len(bins)] = len(bins) - 1  # adjust y.max() to the right most bin
            locs -= 1  # so that can index counts
            weights = np.reciprocal(counts[locs].astype(float))

            all_var_ct.append(np.var(y))
            all_var_fit_ct.append(np.mean(y_fit ** 2))
            all_var_asex_ct.append(np.mean(y_asex ** 2))

            all_var_ct_wt.append(np.sum((y - np.mean(y))**2 * weights) / np.sum(weights))
            all_var_fit_ct_wt.append(np.sum(y_fit**2 * weights) / np.sum(weights))
            all_var_asex_ct_wt.append(np.sum(y_asex**2 * weights) / np.sum(weights))

        # accumulate data into df
        var_df = pd.DataFrame({'Species':all_species, 'Variance':all_var, 'Variance fit':all_var_fit, 'Variance asexual':all_var_asex,
                               'Variance weighted':all_var_wt, 'Variance fit weighted':all_var_fit_wt, 'Variance asexual weighted':all_var_asex_wt,})
        var_df['Variance explained'] = (var_df['Variance'] - var_df['Variance fit']) / var_df['Variance']
        var_df['Variance explained weighted'] = (var_df['Variance weighted'] - var_df['Variance fit weighted']) / var_df['Variance weighted']
        var_df = var_df.set_index('Species')

        control_species = simple_pop_struct_dict.keys()
        var_df.loc[control_species, 'Variance after control'] = all_var_ct
        var_df.loc[control_species, 'Variance fit after control'] = all_var_fit_ct
        var_df.loc[control_species, 'Variance asexual after control'] = all_var_asex_ct

        var_df.loc[control_species, 'Variance weighted after control'] = all_var_ct_wt
        var_df.loc[control_species, 'Variance fit weighted after control'] = all_var_fit_ct_wt
        var_df.loc[control_species, 'Variance asexual weighted after control'] = all_var_asex_ct_wt
        var_df['Variance explained control'] = (var_df['Variance after control'] - var_df['Variance fit after control']) / var_df['Variance after control']
        var_df['Variance explained control'] = var_df['Variance explained control'].fillna(var_df['Variance explained'])

        var_df['Variance explained weighted control'] = (var_df['Variance weighted after control'] - var_df['Variance fit weighted after control']) / var_df['Variance weighted after control']
        var_df['Variance explained weighted control'] = var_df['Variance explained weighted control'].fillna(var_df['Variance explained weighted'])
        var_df = var_df.sort_values('Variance explained weighted control', ascending=False)
        var_df.to_csv(os.path.join(config.plotting_intermediate_directory, 'variance_explained.csv'))


###################################
##### finally plot the statistics
###################################
def plot_var_exaplained(axes, plot_only_y=True):
    var_df = pd.read_csv(os.path.join(config.plotting_intermediate_directory, 'variance_explained.csv'))
    var_df['Genus'] = var_df['Species'].apply(lambda x: x.split('_')[0])
    var_df = var_df.set_index('Species')
    val_genus = var_df.groupby('Genus')['Variance explained weighted control'].mean()
    val_dict = val_genus.to_dict()
    var_df['Genus mean'] = var_df['Genus'].apply(val_dict.get)
    # var_df = var_df.sort_values(['Genus mean', 'Variance explained weighted control'], ascending=[False, False])
    var_df = var_df.sort_values('Variance explained weighted control', ascending=True)

    fontsize = 6
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['legend.frameon']  = False
    mpl.rcParams['legend.fontsize']  = 'small'

    haploid_color = '#08519c'
    light_haploid_color = '#6699CC'
    good_witin_color = '#ef8a62'

    num_species = var_df.shape[0]
    ys = 0-np.arange(0,num_species)

    axes[0].bar(ys+0.5, var_df['Variance explained weighted control'],color=light_haploid_color,label='Largest clade',linewidth=0,zorder=1)
    axes[0].bar(ys+0.5, var_df['Variance explained weighted'],color=haploid_color,linewidth=0,zorder=1, label='All pairs')

    if not plot_only_y:
        axes[1].bar(ys+0.5, -var_df['Variance explained control'],color=light_haploid_color,label='Largest clade',linewidth=0,zorder=1)
        axes[1].bar(ys+0.5, -var_df['Variance explained'],color=haploid_color,linewidth=0,zorder=1)
        axes[1].set_ylim([-1,0])
        axes[1].set_yticks([-1,-0.5])
        axes[1].set_yticklabels(['1.0', '0.5'])

    print("R2y explained >50% in {} species out of {}".format(sum(var_df['Variance explained weighted control']>0.5), var_df.shape[0]))
    print("R2 explained >50% in {} species out of {}".format(sum(var_df['Variance explained control']>0.5), var_df.shape[0]))

    # ax.barh(ys+0.5, num_samples,color=light_haploid_color,linewidth=0,label='hard non-QP',zorder=0)
    # ax.barh(ys+0.5, num_within_samples,left=num_qp_samples,color=good_witin_color,linewidth=0,label='simple non-QP')
    axes[0].set_ylim([0,1])
    axes[0].set_yticks([0,0.5,1])
    #
    # axes[1].yaxis.tick_right()
    # axes[0].xaxis.tick_bottom()
    # axes[1].xaxis.tick_bottom()
    #
    if not plot_only_y:
        axes[0].set_xticks([])
        axes[1].set_xticks(ys+0.5)
        species_names = map(lambda x: figure_utils.get_pretty_species_name(x, manual=True), var_df.index.to_numpy())
        axes[1].set_xticklabels(species_names,fontsize=4, rotation = 90)
        axes[0].set_xlim([-1*num_species+0.5,1.5])
        axes[1].set_xlim([-1*num_species+0.5,1.5])
        #
        # axes[1].tick_params(axis='y', direction='out',length=3,pad=1)
        #
        axes[1].legend(loc='lower right',frameon=False)
        axes[0].set_ylabel(r"$R^2_Y$")
        axes[1].set_ylabel(r"$R^2$")
    else:
        axes[0].set_ylabel(r"Weighted $R^2$")
        axes[0].set_xticks(ys+0.5)
        species_names = map(lambda x: figure_utils.get_pretty_species_name(x, manual=True), var_df.index.to_numpy())
        axes[0].set_xticklabels(species_names,fontsize=5, rotation = 90)
        axes[0].set_xlim([-1*num_species+0.5,1.5])
        axes[0].legend(loc='upper right',frameon=False)


if __name__ == "__main__":
    # prepare_var_explained_data()
    fig, axes = plt.subplots(2, 1, figsize=(3, 2),dpi=300)
    plt.subplots_adjust(hspace=0)
    plot_var_exaplained(axes)
    fig.savefig(os.path.join(config.figure_directory, 'variance_explained_v3.pdf'), bbox_inches='tight')
