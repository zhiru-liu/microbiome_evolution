import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import config
from scipy.stats import gaussian_kde
from utils import close_pair_utils, typical_pair_utils, figure_utils

# species_to_plot = [
#     'Bacteroides_vulgatus_57955',
#     'Bacteroides_uniformis_57318',
#     'Bacteroides_stercoris_56735',
#     'Bacteroides_caccae_53434',
#     'Bacteroides_ovatus_58035',
#     'Bacteroides_thetaiotaomicron_56941',
#     'Bacteroides_massiliensis_44749',
#     'Bacteroides_cellulosilyticus_58046',
#     'Bacteroides_fragilis_54507',
#     'Parabacteroides_merdae_56972',
#     'Parabacteroides_distasonis_56985',
#     'Barnesiella_intestinihominis_62208',
#     'Alistipes_putredinis_61533',
#     'Alistipes_onderdonkii_55464',
#     'Alistipes_shahii_62199',
#     'Oscillibacter_sp_60799',
#     'Akkermansia_muciniphila_55290',
#     'Eubacterium_rectale_56927',
#     'Eubacterium_siraeum_57634',
#     'Ruminococcus_bromii_62047'
# ]
# process all species in figure 3
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
all_close_frac = []
all_total_pairs = []

# set up figure
fig, ax = plt.subplots(figsize=(6, 2.5))
ax2 = ax.twinx()
# locs = np.arange(len(species_to_plot))
locs = []
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

    if len(ys)<900:
        ax.plot(X+xs,ys,'.',color=colorVal,alpha=0.5,markersize=5,markeredgewidth=0.0)
    else:
        ax.plot(X+xs,ys,'.',color=colorVal,alpha=0.5,markersize=5,markeredgewidth=0.0, rasterized=True)

    # ax.fill_betweenx(theory_ys, X-theory_pdf*scale,X+theory_pdf*scale,linewidth=0.25,facecolor=light_colorVal,edgecolor=colorVal)
    other_width = width+0.1
    ax.plot([X-other_width,X+other_width],[q25,q25],'-',color=colorVal,linewidth=1)
    ax.plot([X-other_width,X+other_width],[q50,q50],'-',color='tab:orange',linewidth=1)
    ax.plot([X-other_width,X+other_width],[q75,q75],'-',color=colorVal,linewidth=1)
    ax.plot([X-other_width,X-other_width],[q25,q75],'-',color=colorVal,linewidth=1)
    ax.plot([X+other_width,X+other_width],[q25,q75],'-',color=colorVal,linewidth=1)

plot_idx = 0
isolate_metadata = pd.read_csv(os.path.join(config.isolate_directory, 'isolate_info.csv'), index_col='MGnify_accession')
for species_name, row in isolate_metadata.iterrows():
    third_pass_path = os.path.join(data_dir, 'isolates', "{}_thirdpass.pickle".format(species_name))
    if not os.path.exists(third_pass_path):
        continue
    data = pd.read_pickle(third_pass_path)
    x, y = close_pair_utils.prepare_x_y(data, mode='fraction')
    x_ = x[x > 0]
    y_ = y[x > 0]
    rates = y_ / x_
    Tm = 1 / np.mean(rates)
    print("%s has Tm %f" % (species_name, Tm))  # Tm

    if '1346' in species_name:
        cutoff = [None, 0.06]
    elif '1378' in species_name:
        cutoff = [None, 0.07]
    elif '2366' in species_name:
        cutoff = [None, 0.07]
    elif '2422' in species_name:
        cutoff = [None, 0.04]
    elif '2438' in species_name:
        cutoff = [None, 0.04]
    elif '2478' in species_name:
        cutoff = [None, 0.04]
    elif '2538' in species_name:
        cutoff = [None, 0.03]
    else:
        cutoff = [None, None]

    Tc = typical_pair_utils._compute_theta(species_name, None, clade_cutoff=cutoff)
    if Tc is None:
        continue
    print("%s has Tc %f" % (species_name, Tc))
    print("Tm/Tc is %f" %(Tm / Tc))

    clonal_frac_dir = os.path.join(config.analysis_directory, 'pairwise_clonal_fraction',
                                   'isolates', '%s.csv' % species_name)
    clonal_frac_mat = typical_pair_utils.load_clonal_frac_mat(species_name)
    pd_mat = typical_pair_utils.load_pairwise_div_mat(species_name)

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
    all_total_pairs.append(total_pairs)

    # now plot the distribution of rates

    plot_jitters(ax, plot_idx, rates * Tc, width=0.3)
    locs.append(plot_idx)
    plotted_species.append(species_name)
    plot_idx += 1

    if debug:
        break

_ = ax.set_xticks(locs)
pretty_names = [isolate_metadata.loc[name, 'Species'] for name in plotted_species]
_ = ax.set_xticklabels(pretty_names, rotation=90, ha='center', fontsize=5)
ax.set_ylabel("$T_{mrca} / T_{mosaic}$ (pairwise est)")
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks([])

fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_TcTm_dist_isolates.pdf'), bbox_inches='tight', dpi=600)
df = pd.DataFrame({'Species':pretty_names})
df['MGnify_accession'] = plotted_species
df['d(Tc)'] = all_Tc
df['d(Tm)'] = all_Tm
df['Close pair fraction'] = all_close_frac
df['Tc/Tm'] = df['d(Tc)'] / df['d(Tm)']
df['Total pairs'] = all_total_pairs
df.to_csv(os.path.join(config.figure_directory, 'supp_table', 'TcTm_estimation_isolates.csv'))
