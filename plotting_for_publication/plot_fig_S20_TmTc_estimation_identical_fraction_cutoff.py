import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import config
from scipy.stats import gaussian_kde
from utils import close_pair_utils, typical_pair_utils, figure_utils

# process all species in figure 3
# mostly the same as plot_fig_S18_TmTc_estimation.py, but used identical fraction as cutoffs
# instead of clonal fraction, so no inference is needed
species_to_plot = json.load(open(os.path.join(config.plotting_intermediate_directory, 'fig3_species.json'), 'r'))
plotted_species = []

mpl.rcParams['font.size'] = 5
mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

data_dir = os.path.join(config.analysis_directory, "closely_related")
# hand annotated cutoffs to focus on the diversity within a single clade
Tc_cutoffs = json.load(open(os.path.join(config.analysis_directory, 'misc', 'Tc_cutoffs.json'), 'r'))
all_Tm = []
all_Tc = []

# set up figure
fig, ax = plt.subplots(figsize=(6, 2.5))

plot_idx = 0
for i, species_name in enumerate(species_to_plot):
    data = pd.read_pickle(os.path.join(data_dir, 'third_pass', species_name + '.pickle'))
    cf_mat = typical_pair_utils.load_clonal_frac_mat(species_name)
    data['identical fractions'] = data['pairs'].map(lambda x: cf_mat[x])
    mask = data['identical fractions'] > 0.6
    cf = data['clonal fractions']
    x = data['clonal divs'].to_numpy()[mask]
    y = (1 - cf)[mask]

    x_ = x[x > 0]
    y_ = y[x > 0]
    rates = y_ / x_
    Tm = 1 / np.mean(rates)

    single_sub_idxs = typical_pair_utils.load_single_subject_sample_idxs(species_name)
    Tc = typical_pair_utils._compute_theta(species_name, single_sub_idxs, clade_cutoff=Tc_cutoffs.get(species_name, [None, None]))
    if Tc is None:
        continue
    all_Tm.append(Tm)
    all_Tc.append(Tc)

    # now plot the distribution of rates

df = pd.DataFrame({'Species':plotted_species})
df['d(Tc)'] = all_Tc
df['d(Tm)'] = all_Tm
df['Tc/Tm'] = df['d(Tc)'] / df['d(Tm)']
df.to_csv(os.path.join(config.plotting_intermediate_directory, 'TcTm_estimation_identical_fraction_cutoff.csv'))

og_df = pd.read_csv(os.path.join(config.figure_directory, 'supp_table', 'TcTm_estimation.csv'))

fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(og_df['Tc/Tm'], df['Tc/Tm'], s=5, c='k', alpha=0.5)
xs = np.linspace(0, 100, 100)
ax.plot(xs, xs, c='k', linestyle='--')
ax.set_xlabel('$T_{mrca} / T_{mosaic}$ (original)')
ax.set_ylabel('$T_{mrca} / T_{mosaic}$ (fraction of identical blocks cutoff)')
fig.savefig(os.path.join(config.figure_directory, 'supp', 'S20_supp_TcTm_different_cutoffs.pdf'), bbox_inches='tight', dpi=600)
