import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import config
from scipy.stats import gaussian_kde
from utils import close_pair_utils, typical_pair_utils, figure_utils

species_to_plot = [
    'Bacteroides_vulgatus_57955',
    'Bacteroides_uniformis_57318',
    'Bacteroides_stercoris_56735',
    'Bacteroides_caccae_53434',
    'Bacteroides_ovatus_58035',
    'Bacteroides_thetaiotaomicron_56941',
    'Bacteroides_massiliensis_44749',
    'Bacteroides_cellulosilyticus_58046',
    'Bacteroides_fragilis_54507',
    'Parabacteroides_merdae_56972',
    'Parabacteroides_distasonis_56985',
    'Barnesiella_intestinihominis_62208',
    'Alistipes_putredinis_61533',
    'Alistipes_onderdonkii_55464',
    'Alistipes_shahii_62199',
    'Oscillibacter_sp_60799',
    'Akkermansia_muciniphila_55290',
    'Eubacterium_rectale_56927',
    'Eubacterium_siraeum_57634',
    'Ruminococcus_bromii_62047'
]

mpl.rcParams['font.size'] = 5
mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

data_dir = os.path.join(config.analysis_directory, "closely_related")
Tc_cutoffs = json.load(open(os.path.join(config.analysis_directory, 'misc', 'Tc_cutoffs.json'), 'r'))
all_Tm = []
all_Tc = []
all_close_frac = []

# set up figure
fig, ax = plt.subplots(figsize=(6, 4))
ax2 = ax.twinx()
locs = np.arange(len(species_to_plot))
debug=False


def plot_jitters(ax, X, ys, width, colorVal='tab:blue'):
    kernel = gaussian_kde(ys)

    theory_ys = np.linspace(ys.min(),ys.max(),100)
    theory_pdf = kernel(theory_ys)

    scale = width/theory_pdf.max()

    xs = np.random.uniform(-1,1,size=len(ys))*kernel(ys)*scale

    q25 = np.quantile(ys,0.25)
    q50 = np.quantile(ys,0.5)
    q75 = np.quantile(ys,0.75)

    # ax.fill_betweenx(theory_ys, X-theory_pdf*scale,X+theory_pdf*scale,linewidth=0.25,facecolor=light_colorVal,edgecolor=colorVal)
    other_width = width+0.1
    ax.plot([X-other_width,X+other_width],[q25,q25],'-',color=colorVal,linewidth=1)
    ax.plot([X-other_width,X+other_width],[q50,q50],'-',color=colorVal,linewidth=1)
    ax.plot([X-other_width,X+other_width],[q75,q75],'-',color=colorVal,linewidth=1)
    ax.plot([X-other_width,X-other_width],[q25,q75],'-',color=colorVal,linewidth=1)
    ax.plot([X+other_width,X+other_width],[q25,q75],'-',color=colorVal,linewidth=1)

    if len(ys)<900:
        ax.plot(X+xs,ys,'.',color=colorVal,alpha=0.5,markersize=5,markeredgewidth=0.0)
    else:
        ax.plot(X+xs,ys,'.',color=colorVal,alpha=0.5,markersize=5,markeredgewidth=0.0, rasterized=True)


for i, species_name in enumerate(species_to_plot):
    data = pd.read_pickle(os.path.join(data_dir, 'third_pass', species_name + '.pickle'))
    x, y = close_pair_utils.prepare_x_y(data, mode='fraction')
    x_ = x[x > 0]
    y_ = y[x > 0]
    rates = y_ / x_
    Tm = 1 / np.mean(rates)
    print("%s has Tm %f" % (species_name, Tm))  # Tm

    single_sub_idxs = typical_pair_utils.load_single_subject_sample_idxs(species_name)
    Tc = typical_pair_utils._compute_theta(species_name, single_sub_idxs, clade_cutoff=Tc_cutoffs[species_name])
    print("%s has Tc %f" % (species_name, Tc))
    print("Tm/Tc is %f" %(Tm / Tc))

    clonal_frac_dir = os.path.join(config.analysis_directory, 'pairwise_clonal_fraction',
                                   'between_hosts', '%s.csv' % species_name)
    clonal_frac_mat = typical_pair_utils.load_clonal_frac_mat(species_name)
    clonal_frac_mat = clonal_frac_mat[single_sub_idxs, :][:, single_sub_idxs]
    pd_mat = typical_pair_utils.load_pairwise_div_mat(species_name)
    pd_mat = pd_mat[single_sub_idxs, :][:, single_sub_idxs]

    cf_dist = clonal_frac_mat[np.triu_indices(clonal_frac_mat.shape[0], 1)]
    pd_dist = pd_mat[np.triu_indices(clonal_frac_mat.shape[0], 1)]
    close_pairs = np.sum(cf_dist > 0.2)
    # total_pairs = np.sum(pd_dist < Tc_cutoffs[species_name][1])
    total_pairs = len(cf_dist)
    close_fraction = close_pairs / float(total_pairs)
    print(close_fraction)
    all_Tm.append(Tm)
    all_Tc.append(Tc)
    all_close_frac.append(close_fraction)

    # now plot the distribution of rates
    plot_jitters(ax, locs[i], rates, width=0.3)

    if debug:
        break

_ = ax.set_xticks(locs)
pretty_names = [figure_utils.get_pretty_species_name(name) for name in species_to_plot]
_ = ax.set_xticklabels(pretty_names, rotation=90, ha='center', fontsize=5)
ax.set_ylabel("recombined fraction / clonal divergence")
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks([])

fig.savefig(os.path.join(config.figure_directory, 'Tm_dist.pdf'), bbox_inches='tight', dpi=600)
df = pd.DataFrame({'Species':species_to_plot, 'Tm':all_Tm, 'Tc':all_Tc, 'Close pair fraction':all_close_frac})
df.to_csv(os.path.join(config.analysis_directory, 'misc', 'TmTc_estimation.csv'))
