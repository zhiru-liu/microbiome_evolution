import numpy as np
import json
import sys
import os
import random
import json
import gzip
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate
from scipy.stats import gaussian_kde
from Bio import SeqIO

# Basically the same as plot fig 3, but added gc content track

sys.path.append("..")
import config
from utils import close_pair_utils, species_phylogeny_utils

from matplotlib.transforms import Bbox, TransformedBbox, \
    blended_transform_factory

from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, \
    BboxConnectorPatch

species_priority = json.load(open(os.path.join(config.analysis_directory, 'species_plotting_priority.json'), 'r'))
# a heuristic order roughly by phylus and genus and fraction of QP pairs
species_order = [
    'Bacteroides_vulgatus_57955',
    'Bacteroides_ovatus_58035',
    'Bacteroides_uniformis_57318',
    'Bacteroides_thetaiotaomicron_56941',
    'Bacteroides_plebeius_61623',
    'Bacteroides_stercoris_56735',
    'Bacteroides_coprocola_61586',
    'Bacteroides_eggerthii_54457',
    'Bacteroides_finegoldii_57739',
    'Bacteroides_caccae_53434',
    'Bacteroides_cellulosilyticus_58046',
    'Bacteroides_fragilis_54507',
    'Bacteroides_massiliensis_44749',

    'Barnesiella_intestinihominis_62208',
    'Odoribacter_splanchnicus_62174',
    'Parabacteroides_distasonis_56985',
    'Parabacteroides_merdae_56972',

    'Bacteroidales_bacterium_58650',

    'Prevotella_copri_61740',

    'Alistipes_shahii_62199',
    'Alistipes_putredinis_61533',
    'Alistipes_onderdonkii_55464',
    'Alistipes_finegoldii_56071',
    'Alistipes_sp_60764',

    # 'Lachnospiraceae_bacterium_51870',  # removed because of lack of close pairs
    'Coprococcus_sp_62244',
    'Roseburia_intestinalis_56239',

    'Eubacterium_rectale_56927',
    'Eubacterium_siraeum_57634',
    'Faecalibacterium_cf_62236',
    'Ruminococcus_bromii_62047',
    'Ruminococcus_bicirculans_59300',

    'Oscillibacter_sp_60799',

    'Dialister_invisus_61905',

    'Phascolarctobacterium_sp_59817',

    'Akkermansia_muciniphila_55290']

species_cutoff_dict = json.load(open(os.path.join(config.plotting_intermediate_directory, 'clonal_div_cutoff.json'), 'r'))
species_cutoff_dict['Bacteroides_vulgatus_57955'] = config.Bv_clonal_div_cutoff

data_dir = os.path.join(config.analysis_directory, "closely_related")

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'
plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(3, 1)
gs.update(hspace=0)
axt = fig.add_subplot(gs[0, :])
axm = fig.add_subplot(gs[1, :])
axd = fig.add_subplot(gs[2:, :])

idx = 1
xticks = []
xticklabels = []
plotted_species = []
scatter_species = []
connection_l = []
connection_r = []

transfer_length_data = []
all_num_pairs = []
plot_loc = []


def interpolate_curve(xval, yval, sample_locations=[2.5e-5, 5e-5, 7.5e-5, 1e-4], scale=True):
    f = interpolate.interp1d(xval, yval, bounds_error=False)
    res = f(sample_locations)
    if scale:
        res = res / (np.array([2.5e-5, 5e-5, 7.5e-5, 1e-4]))  # transfer per mutation
    return res


def plot_jitters(ax, X, ys, width, if_box=True, colorVal='tab:blue', alpha=0.1, marker='.'):
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
    if if_box:
        ax.plot([X-other_width,X+other_width],[q25,q25],'-',color=colorVal,linewidth=1)
        ax.plot([X-other_width,X+other_width],[q50,q50],'-',color=colorVal,linewidth=1)
        ax.plot([X-other_width,X+other_width],[q75,q75],'-',color=colorVal,linewidth=1)
        ax.plot([X-other_width,X-other_width],[q25,q75],'-',color=colorVal,linewidth=1)
        ax.plot([X+other_width,X+other_width],[q25,q75],'-',color=colorVal,linewidth=1)

    if len(ys)<900:
        ax.scatter(X+xs,ys,marker=marker,color=colorVal,alpha=alpha,s=2)
    else:
        ax.scatter(X+xs,ys,marker=marker,color=colorVal,alpha=alpha,s=2, rasterized=True)


def find_GC_content(species_name):
    fasta_file = os.path.join(config.data_directory, 'midas_db', 'rep_genomes', species_name, 'genome.fna.gz')

    with gzip.open(fasta_file, 'rt') as f:
        headers = []
        seqs = []
        for record in SeqIO.parse(f, 'fasta'):
            # Access information about the sequence record
            header = record.id
            sequence = str(record.seq)
            headers.append(header)
            seqs.append(sequence)
    total_len = 0
    total_gc = 0
    for seq in seqs:
        total_len += seq.count('A') + seq.count('T') + seq.count('G') + seq.count('C')
        total_gc += seq.count('G') + seq.count('C')
    return float(total_gc) / total_len

other_species_to_plot = []

full_transfer_df = pd.read_csv(os.path.join(config.figure_directory, 'supp_table', 'all_transfers.csv'), index_col=0)
full_transfer_df['Sample 1'] = full_transfer_df['Sample 1'].astype(str)
full_transfer_df['Sample 2'] = full_transfer_df['Sample 2'].astype(str)
for species_full_name in species_order:
    species_name = ' '.join(species_full_name.split('_')[:2])
    try:
        raw_data = pd.read_pickle(os.path.join(data_dir, 'third_pass', species_full_name + '.pickle'))
    except IOError:
        # print(species_name + " does not have data")
        continue
    filename = species_full_name + '.csv'

    if 'vulgatus' in species_name:
        # data_path = os.path.join(data_dir, 'third_pass', species_full_name + '_all_transfers_processed.pickle')
        data_path = os.path.join(data_dir, 'third_pass', species_full_name + '_all_transfers.pickle')
    else:
        data_path = os.path.join(data_dir, 'third_pass', species_full_name + '_all_transfers.pickle')

    transfer_data = pd.read_pickle(data_path)

    n = raw_data.shape[0]
    n_filtered = np.sum(raw_data['clonal fractions'] > config.clonal_fraction_cutoff)
    if n_filtered < 3:
        continue
    # print(filename, n, n_filtered, raw_data['clonal divs'].max())
    plot_jitter = False

    clonal_div_cutoff = species_cutoff_dict.get(species_full_name, 1)
    if clonal_div_cutoff is None:
        clonal_div_cutoff = 1

    if 'vulgatus' in species_name:
        # for B vulgatus, everything needs to be done twice
        trend_directory = os.path.join(config.plotting_intermediate_directory, "B_vulgatus_trend_line.csv")
        fitted_data = pd.read_csv(trend_directory)

        x, y1, y2, _, cf_mask = close_pair_utils.prepare_HMM_results_for_B_vulgatus(save_path=config.B_vulgatus_data_path,
            cf_cutoff=config.clonal_fraction_cutoff, mode='fraction', cache_intermediate=False)
        x = x[cf_mask]
        y1 = y1[cf_mask]
        y2= y2[cf_mask]
        x_ = x[(x > 0) & (x<clonal_div_cutoff)]
        y1_ = y1[(x > 0) & (x<clonal_div_cutoff)]
        y2_ = y2[(x > 0) & (x<clonal_div_cutoff)]

        # within first
        mid = interpolate_curve(fitted_data['within_x'], fitted_data['within_y'])
        w = interpolate_curve(fitted_data['within_x'], fitted_data['within_sigma'])
        plot_loc.append(2 * idx)
        #
        xloc = np.linspace(2 * idx - 0.3, 2 * idx + 0.3, 4, endpoint=True)
        axm.scatter(xloc, mid / 1e6, s=5)
        axm.plot(xloc, mid / 1e6, linestyle=':', linewidth=1)
        # axm.vlines(xloc, mid - w, mid + w, alpha=0.2)
        axt.plot(idx * 2, find_GC_content(species_full_name), 'o')


        # good_runs, num_pairs = close_pair_utils.prepare_run_lengths(raw_data, transfer_data, desired_type=0, clonal_div_cutoff=clonal_div_cutoff)
        good_runs_mask = (full_transfer_df['Species name']==species_full_name) & (full_transfer_df['Shown in Fig3?']) & (full_transfer_df['between clade?']=='N')
        good_runs_mask = good_runs_mask & ~full_transfer_df['Potential duplicate of other events?']
        good_runs = full_transfer_df[good_runs_mask]['Transfer length (# covered sites on core genome)']

        transfer_length_data.append(good_runs)

        # plot fraction recombined
        # rates = y1_ / x_
        # frac_recombined = rates * 1e-4
        # plot_jitters(axm, 2*idx, frac_recombined, width=0.4, colorVal=plot_colors[(idx-1) % len(plot_colors)])

        xticks.append(idx * 2)
        xticklabels.append(species_name + '\n(within clade)')
        idx += 1

        # between next
        mid = interpolate_curve(fitted_data['between_x'], fitted_data['between_y'])
        w = interpolate_curve(fitted_data['between_x'], fitted_data['between_sigma'])
        plot_loc.append(2 * idx)

        xloc = np.linspace(2 * idx - 0.3, 2 * idx + 0.3, 4, endpoint=True)
        axm.scatter(xloc, mid / 1e6, s=5)
        axm.plot(xloc, mid / 1e6, linestyle=':', linewidth=1)
        # axm.vlines(xloc, mid - w, mid + w, alpha=0.2)

        # good_runs, num_pairs = close_pair_utils.prepare_run_lengths(raw_data, transfer_data, desired_type=1, clonal_div_cutoff=clonal_div_cutoff)
        good_runs_mask = (full_transfer_df['Species name']==species_full_name) & (full_transfer_df['Shown in Fig3?']) & (full_transfer_df['between clade?']=='Y')
        num_pairs = len(pd.unique(zip(full_transfer_df[good_runs_mask]['Sample 1'], full_transfer_df[good_runs_mask]['Sample 2'])))
        good_runs_mask = good_runs_mask & (~full_transfer_df['Potential duplicate of other events?'])
        good_runs = full_transfer_df[good_runs_mask]['Transfer length (# covered sites on core genome)']

        transfer_length_data.append(good_runs)

        # rates = y2_ / x_
        # frac_recombined = rates * 1e-4
        # plot_jitters(axm, 2*idx, frac_recombined, width=0.4, colorVal=plot_colors[(idx-1) % len(plot_colors)])

        xticks.append(idx * 2)
        xticklabels.append(species_name + '\n(between clade)')
        plotted_species.append(species_full_name)
        idx += 1
        continue


    ######### Plotting the scatter summary #########
    x, y = close_pair_utils.prepare_x_y(raw_data)
    try:
        fitted_data = pd.read_csv(os.path.join(data_dir, 'fourth_pass', filename), index_col=0)
        # using divergence as x values
        mid = interpolate_curve(fitted_data['x'], fitted_data['y'], scale=True)
        w = interpolate_curve(fitted_data['x'], fitted_data['sigma'], scale=True)
        if np.isnan(mid).sum() > 0:
            # data does not cover the range 0 to 1e-4
            plot_jitter = True
        if n_filtered < 100:
            plot_jitter = True
    except IOError:
        plot_jitter = True

    # plotting species summary
    plot_loc.append(2 * idx)
    # older version
    # good_runs, num_pairs = close_pair_utils.prepare_run_lengths(raw_data, transfer_data, clonal_div_cutoff=clonal_div_cutoff)

    good_runs_mask = (full_transfer_df['Species name']==species_full_name) & (full_transfer_df['Shown in Fig3?'])
    num_pairs = len(pd.unique(zip(full_transfer_df[good_runs_mask]['Sample 1'], full_transfer_df[good_runs_mask]['Sample 2'])))
    good_runs_mask = good_runs_mask & (~full_transfer_df['Potential duplicate of other events?'])
    good_runs = full_transfer_df[good_runs_mask]['Transfer length (# covered sites on core genome)']

    transfer_length_data.append(good_runs)
    xloc = np.linspace(2 * idx - 0.3, 2 * idx + 0.3, 4, endpoint=True)

    color = plot_colors[(idx-1) % len(plot_colors)]
    if plot_jitter:
        x, y = close_pair_utils.prepare_x_y(raw_data, mode='count')
        x_ = x[(x > 0)&(x<clonal_div_cutoff)]
        y_ = y[(x > 0)&(x<clonal_div_cutoff)]
        rates = y_ / x_ / 1e6
        # frac_recombined = rates * 1e-4
        plot_jitters(axm, 2*idx, rates, width=0.4, colorVal=color, alpha=0.5, marker='^')
    else:
        axm.scatter(xloc, mid / 1e6, s=5, color=color)
        axm.plot(xloc, mid / 1e6, linestyle=':', linewidth=1, color=color)
        scatter_species.append(species_name)
        # axm.vlines(xloc, mid - w, mid + w, alpha=0.2, color=color)

    axt.plot(idx * 2, find_GC_content(species_full_name), 'o', color=color)

    ########### Plot the count vs divergence highlights ########
    x_, y_, cf_mask = close_pair_utils.prepare_x_y(raw_data, return_unfiltered=True)
    x = x_[cf_mask]
    y = y_[cf_mask]
    kw = dict(linestyle=":", linewidth=1, color='grey')
    xloc = np.linspace(2 * idx - 0.3, 2 * idx + 0.3, 4, endpoint=True)
    x_highlight = np.array([2.5e-5, 5e-5, 7.5e-5, 1e-4])
    y_highlight = x_highlight * mid

    xticks.append(idx * 2)
    xticklabels.append(species_name)
    plotted_species.append(species_full_name)
    idx += 1

# plotting all other violin plots
# violins = axd.violinplot(transfer_length_data[:], positions=plot_loc[:], vert=True, showmedians=True, showextrema=False,
#                          widths=0.8)
# print("\nSpecies name, median transfer, mean transfer")
for i, loc in enumerate(plot_loc):
    points_to_plot = 1000
    ys = transfer_length_data[i]
    if len(ys) > points_to_plot:
        ys = np.array(random.sample(ys, points_to_plot))
    # xs = np.ones(ys.shape) * loc + np.random.normal(scale=0.05, size=ys.shape)
    # axd.scatter(xs, ys, color=plot_colors[(i) % len(plot_colors)], s=0.2, rasterized=True, alpha=0.1)
    # print(xticklabels[i], np.median(ys), np.mean(ys))
    plot_jitters(axd, loc, ys, width=0.4, colorVal=plot_colors[i % len(plot_colors)])

# for i, pc in enumerate(violins['bodies']):
#     pc.set_facecolor(plot_colors[(i) % len(plot_colors)])
#     pc.set_alpha(0.5)
#     pc.set_edgecolor('black')
# violins['cmedians'].set_edgecolor('black')

# plot the shaded background for genera
genera = [x.split(' ')[0] for x in xticklabels]
prev_genus = genera[0]
plot_bkg = False
for i, genus in enumerate(genera):
    x_span = [2 * i + 1, 2 * i + 3]
    if genus != prev_genus:
        # alternate background color
        plot_bkg = not plot_bkg
    if plot_bkg:
        axm.axvspan(x_span[0], x_span[1], color='grey', alpha=0.1, zorder=6, linewidth=0)
        axd.axvspan(x_span[0], x_span[1], color='grey', alpha=0.1, zorder=6, linewidth=0)
    prev_genus = genus
# axm.set_ylabel('Num transfers')
# axm.set_ylabel("Recombined fraction \n @ $d_c=10^{-4}$")
axm.set_ylabel("Transfers / SNV")
axm.grid(linestyle='--', axis='y')
axm.set_yscale('log')
axm.set_ylim([2e-3, 2])
# axm.set_ylim([-0.05, 0.5])

ymax = axm.get_ylim()[1]
# adding connection lines
# for i in range(3):
#     ax = [ax1, ax2, ax3][i]
#     cpl = ConnectionPatch((0, -0.4), (connection_l[i], ymax), "axes fraction", "data",
#                           axesA=ax, axesB=axm, **kw)
#     cpr = ConnectionPatch((1, -0.4), (connection_r[i], ymax), "axes fraction", "data",
#                           axesA=ax, axesB=axm, **kw)
#     axm.add_artist(cpl)
#     axm.add_artist(cpr)
#     cprr = ConnectionPatch((1, -0.4), (1, 0), "axes fraction", "axes fraction",
#                            axesA=ax, axesB=ax, **kw)
#     ax.add_artist(cprr)

# axd.set_ylabel('Num transfers\n per 4d clonal snp')
axd.set_ylabel('Transfer length')
_ = axd.set_xticks([])
_ = axd.set_xticklabels([])
# axd.set_yticks([0, 0.25, 0.5, 0.75, 1])
# axd.set_yscale('log')
axd.grid(linestyle='--', axis='y')
axd.set_ylim([0.9e3, 1.3e5])
axd.set_yscale('log')
axd.set_yticks([1e3, 1e4, 1e5])
axd.set_yticklabels(['1kb', '10kb', '100kb'])
axm.set_xlim([1, 2 * len(xticklabels) + 1])
axd.set_xlim([1, 2 * len(xticklabels) + 1])
axt.set_xlim(axm.get_xlim())
axt.set_ylim([0.38, 0.62])
axt.set_ylabel('GC content')

_ = axd.set_xticks(xticks)
_ = axd.set_xticklabels(xticklabels, rotation=90, ha='center', fontsize=5)
json.dump(plotted_species, open(os.path.join(config.plotting_intermediate_directory, 'fig3_species.json'), 'w'))


axt.text(0.03, 0.9, "A", transform=axt.transAxes,
        fontsize=9, fontweight='bold', va='top', ha='left')
axm.text(0.03, 0.9, "B", transform=axm.transAxes,
        fontsize=9, fontweight='bold', va='top', ha='left')
axd.text(0.03, 0.9, "C", transform=axd.transAxes,
         fontsize=9, fontweight='bold', va='top', ha='left')

fig.tight_layout()
fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_gc_content.pdf'), dpi=600)