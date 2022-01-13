import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import config
from utils import typical_pair_utils

# manually controlling species with apparent multi clade structure
simple_pop_struct_dict = {'Bacteroides_vulgatus_57955': 0.03,
                          'Alistipes_shahii_62199': 0.03,
                          'Bacteroides_uniformis_57318': 0.03,
                          'Ruminococcus_bromii_62047':0.055,
                          'Roseburia_intestinalis_56239':0.03,
                          'Bacteroides_cellulosilyticus_58046':0.04,
                          'Alistipes_finegoldii_56071':0.025}

# first fitting all species regularly
all_species = []
all_var = []
all_var_fit = []
all_var_asex = []

for species_name in os.listdir(os.path.join(config.data_directory, 'zarr_snps')):
    if species_name.startswith('.'):
        continue
    x, y = typical_pair_utils.get_joint_plot_x_y(species_name)
    F = typical_pair_utils.fit_quadratic_curve(x, y)
    y_fit = y - F(x)
    y_asex = y - typical_pair_utils.asexual_curve(x, default=y[x == 0].mean())

    all_species.append(species_name)
    all_var.append(np.var(y))
    all_var_fit.append(np.mean(y_fit ** 2))
    all_var_asex.append(np.mean(y_asex ** 2))

# then process species with easy population structure (taking only the largest clade)
all_var_ct = []
all_var_fit_ct = []
all_var_asex_ct = []

for species_name in simple_pop_struct_dict:
    x, y = typical_pair_utils.get_joint_plot_x_y(species_name, clade_cutoff=simple_pop_struct_dict[species_name])
    F = typical_pair_utils.fit_quadratic_curve(x, y)
    y_fit = y - F(x)
    y_asex = y - typical_pair_utils.asexual_curve(x, default=y[x == 0].mean())

    all_var_ct.append(np.var(y))
    all_var_fit_ct.append(np.mean(y_fit ** 2))
    all_var_asex_ct.append(np.mean(y_asex ** 2))

# accumulate data into df
var_df = pd.DataFrame({'Species':all_species, 'Variance':all_var, 'Variance fit':all_var_fit, 'Variance asexual':all_var_asex})
var_df['Variance explained'] = (var_df['Variance'] - var_df['Variance fit']) / var_df['Variance']
var_df = var_df.set_index('Species')

control_species = simple_pop_struct_dict.keys()
var_df.loc[control_species, 'Variance after control'] = all_var_ct
var_df.loc[control_species, 'Variance fit after control'] = all_var_fit_ct
var_df.loc[control_species, 'Variance asexual after control'] = all_var_asex_ct
var_df['Variance explained control'] = (var_df['Variance after control'] - var_df['Variance fit after control']) / var_df['Variance after control']
var_df['Variance explained control'] = var_df['Variance explained control'].fillna(var_df['Variance explained'])
var_df = var_df.sort_values('Variance explained control', ascending=False)



###################################
##### finally plot the statistics
###################################

fig, ax = plt.subplots(figsize=(1.5, 3),dpi=300)
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

ax.barh(ys+0.5, var_df['Variance explained control'],color=light_haploid_color,label='Largest clade',linewidth=0,zorder=1)
ax.barh(ys+0.5, var_df['Variance explained'],color=haploid_color,linewidth=0,zorder=1)

# ax.barh(ys+0.5, num_samples,color=light_haploid_color,linewidth=0,label='hard non-QP',zorder=0)
# ax.barh(ys+0.5, num_within_samples,left=num_qp_samples,color=good_witin_color,linewidth=0,label='simple non-QP')
ax.set_xlim([0,1])
ax.set_xticks([0,0.5,1])

ax.yaxis.tick_right()
ax.xaxis.tick_bottom()

ax.set_yticks(ys+0.5)
ax.set_yticklabels(var_df.index,fontsize=4)
ax.set_ylim([-1*num_species+0.5,1.5])

ax.tick_params(axis='y', direction='out',length=3,pad=1)

ax.legend(loc='lower right',frameon=False)
ax.set_xlabel(r"$R^2$")

fig.savefig(os.path.join(config.figure_directory, 'variance_explained.pdf'), bbox_inches='tight')
