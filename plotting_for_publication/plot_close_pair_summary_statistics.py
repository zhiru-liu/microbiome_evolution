import numpy as np
import json
import sys
import os
import random
import json
import pandas as pd
import seaborn as sns
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from scipy.stats import gaussian_kde

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

    'Lachnospiraceae_bacterium_51870',
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


data_dir = os.path.join(config.analysis_directory, "closely_related")

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'
plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(11, 3)
gs.update(hspace=0)
ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[:3, 1], sharey=ax1)
ax3 = fig.add_subplot(gs[:3, 2], sharey=ax1)
axm = fig.add_subplot(gs[5:8, :])
axd = fig.add_subplot(gs[8:, :])

idx = 1
xticks = []
xticklabels = []
plotted_species = []
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

other_species_to_plot = []
for species_full_name in species_order:
    species_name = ' '.join(species_full_name.split('_')[:2])
    try:
        raw_data = pd.read_pickle(os.path.join(data_dir, 'third_pass', species_full_name + '.pickle'))
    except IOError:
        print(species_name + " does not have data")
        continue
    filename = species_full_name + '.csv'

    if 'vulgatus' in species_name:
        data_path = os.path.join(data_dir, 'third_pass', species_full_name + '_all_transfers_processed.pickle')
    else:
        data_path = os.path.join(data_dir, 'third_pass', species_full_name + '_all_transfers.pickle')

    transfer_data = pd.read_pickle(data_path)

    n = raw_data.shape[0]
    n_filtered = np.sum(raw_data['clonal fractions'] > config.clonal_fraction_cutoff)
    if n_filtered < 3:
        continue
    print(filename, n, n_filtered, raw_data['clonal divs'].max())
    plot_jitter = False

    if 'vulgatus' in species_name:
        # for B vulgatus, everything needs to be done twice
        trend_directory = os.path.join(config.plotting_intermediate_directory, "B_vulgatus_trend_line.csv")
        fitted_data = pd.read_csv(trend_directory)


        x, y1, y2, _ = close_pair_utils.prepare_HMM_results_for_B_vulgatus(save_path=config.B_vulgatus_data_path,
            cf_cutoff=config.clonal_fraction_cutoff, mode='fraction', cache_intermediate=False)
        x_ = x[x > 0]
        y1_ = y1[x > 0]
        y2_ = y2[x > 0]

        # within first
        mid = interpolate_curve(fitted_data['within_x'], fitted_data['within_y'])
        w = interpolate_curve(fitted_data['within_x'], fitted_data['within_sigma'])
        plot_loc.append(2 * idx)
        #
        xloc = np.linspace(2 * idx - 0.3, 2 * idx + 0.3, 4, endpoint=True)
        axm.scatter(xloc, mid, s=5)
        axm.plot(xloc, mid, linestyle=':', linewidth=1)
        # axm.vlines(xloc, mid - w, mid + w, alpha=0.2)

        good_runs, num_pairs = close_pair_utils.prepare_run_lengths(raw_data, transfer_data, desired_type=0)
        transfer_length_data.append(good_runs)
        all_num_pairs.append(num_pairs)

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
        axm.scatter(xloc, mid, s=5)
        axm.plot(xloc, mid, linestyle=':', linewidth=1)
        # axm.vlines(xloc, mid - w, mid + w, alpha=0.2)

        good_runs, num_pairs = close_pair_utils.prepare_run_lengths(raw_data, transfer_data, desired_type=1)
        transfer_length_data.append(good_runs)
        all_num_pairs.append(num_pairs)

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
    good_runs, num_pairs = close_pair_utils.prepare_run_lengths(raw_data, transfer_data)
    transfer_length_data.append(good_runs)
    all_num_pairs.append(num_pairs)
    xloc = np.linspace(2 * idx - 0.3, 2 * idx + 0.3, 4, endpoint=True)

    color = plot_colors[(idx-1) % len(plot_colors)]
    if plot_jitter:
        x, y = close_pair_utils.prepare_x_y(raw_data, mode='count')
        x_ = x[x > 0]
        y_ = y[x > 0]
        rates = y_ / x_
        # frac_recombined = rates * 1e-4
        plot_jitters(axm, 2*idx, rates, width=0.4, colorVal=color, alpha=0.5, marker='^')
    else:
        axm.scatter(xloc, mid, s=5, color=color)
        axm.plot(xloc, mid, linestyle=':', linewidth=1, color=color)
        # axm.vlines(xloc, mid - w, mid + w, alpha=0.2, color=color)


    ######### Plotting the T_m estimation ##########
    # plot_loc.append(2 * idx)
    # good_runs = close_pair_utils.prepare_run_lengths(raw_data, transfer_data)
    # transfer_length_data.append(good_runs)
    #
    # x, y = close_pair_utils.prepare_x_y(raw_data, mode='fraction')
    # x_ = x[x > 0]
    # y_ = y[x > 0]
    # rates = y_ / x_
    # frac_recombined = rates * 1e-4
    # plot_jitters(axm, 2*idx, frac_recombined, width=0.4, colorVal=plot_colors[(idx-1) % len(plot_colors)])

    ########### Plot the count vs divergence highlights ########
    x, y = close_pair_utils.prepare_x_y(raw_data)
    kw = dict(linestyle=":", linewidth=1, color='grey')
    xloc = np.linspace(2 * idx - 0.3, 2 * idx + 0.3, 4, endpoint=True)
    x_highlight = np.array([2.5e-5, 5e-5, 7.5e-5, 1e-4])
    y_highlight = x_highlight * mid
    if 'caccae' in species_name:
        fitted_data = pd.read_csv(os.path.join(data_dir, 'fourth_pass', filename), index_col=0)
        ax1.scatter(x, y, s=1)
        ax1.plot(fitted_data['x'], fitted_data['y'], linestyle=':', color=color)
        ax1.plot(x_highlight, y_highlight, '.', color=color, markersize=5)
        ax1.fill_between(fitted_data['x'], fitted_data['y'] - fitted_data['sigma'],
                         fitted_data['y'] + fitted_data['sigma'], alpha=0.25)
        ax1.set_title(species_name)
        ax1.set_xlabel("Clonal divergence")
        ax1.set_xlim([0, 1e-4])
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        axm.axvline(xloc[0] - 0.4, **kw)
        axm.axvline(xloc[-1] + 0.4, **kw)
        axd.axvline(xloc[0] - 0.4, **kw)
        axd.axvline(xloc[-1] + 0.4, **kw)
        connection_l.append(xloc[0] - 0.4)
        connection_r.append(xloc[-1] + 0.4)

    if 'massiliensis' in species_name:
        fitted_data = pd.read_csv(os.path.join(data_dir, 'fourth_pass', filename), index_col=0)
        ax2.scatter(x, y, s=1)
        ax2.plot(fitted_data['x'], fitted_data['y'], linestyle='--', color=color)
        ax2.plot(x_highlight, y_highlight, '.', color=color, markersize=5)
        ax2.fill_between(fitted_data['x'], fitted_data['y'] - fitted_data['sigma'],
                         fitted_data['y'] + fitted_data['sigma'], alpha=0.25)
        ax2.set_title(species_name)
        ax2.set_xlabel("Clonal divergence")
        ax2.set_xlim([0, 2e-4])
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        axm.axvline(xloc[0] - 0.4, **kw)
        axm.axvline(xloc[-1] + 0.4, **kw)
        axd.axvline(xloc[0] - 0.4, **kw)
        axd.axvline(xloc[-1] + 0.4, **kw)
        connection_l.append(xloc[0] - 0.4)
        connection_r.append(xloc[-1] + 0.4)

    if 'putredinis' in species_name:
        fitted_data = pd.read_csv(os.path.join(data_dir, 'fourth_pass', filename), index_col=0)
        ax3.scatter(x, y, s=1)
        ax3.plot(x_highlight, y_highlight, '.', color=color, markersize=5)

        # select three examples pairs
        # example_mask = raw_data['pairs'].isin([(282, 387), (297, 331), (269, 313)])
        # ax3.scatter(x[example_mask], y[example_mask], s=1, color='r')
        print('\n')
        print("A putredinis average rate:{:e}".format(np.mean(y[x>0]/x[x>0])))
        print("A putredinis intermediate count:{}".format(y_highlight))
        print('\n')
        ax3.plot(fitted_data['x'], fitted_data['y'], linestyle='--', color=color)
        # ax3.set_xlim([0, 50])
        ax3.set_xlim([0, 2e-4])
        ax3.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax3.set_xlabel("Clonal divergence")
        ax3.fill_between(fitted_data['x'], fitted_data['y'] - fitted_data['sigma'],
                         fitted_data['y'] + fitted_data['sigma'], alpha=0.25)
        ax3.set_title(species_name)
        axm.axvline(xloc[0] - 0.4, **kw)
        axm.axvline(xloc[-1] + 0.4, **kw)
        axd.axvline(xloc[0] - 0.4, **kw)
        axd.axvline(xloc[-1] + 0.4, **kw)
        connection_l.append(xloc[0] - 0.4)
        connection_r.append(xloc[-1] + 0.4)

    xticks.append(idx * 2)
    xticklabels.append(species_name)
    plotted_species.append(species_full_name)
    idx += 1

# plotting all other violin plots
# violins = axd.violinplot(transfer_length_data[:], positions=plot_loc[:], vert=True, showmedians=True, showextrema=False,
#                          widths=0.8)
print("\nSpecies name, median transfer, mean transfer")
for i, loc in enumerate(plot_loc):
    points_to_plot = 1000
    ys = transfer_length_data[i]
    if len(ys) > points_to_plot:
        ys = np.array(random.sample(ys, points_to_plot))
    # xs = np.ones(ys.shape) * loc + np.random.normal(scale=0.05, size=ys.shape)
    # axd.scatter(xs, ys, color=plot_colors[(i) % len(plot_colors)], s=0.2, rasterized=True, alpha=0.1)
    plot_jitters(axd, loc, ys, width=0.4, colorVal=plot_colors[i % len(plot_colors)])
    print(xticklabels[i], np.median(ys), np.mean(ys))

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
print("In total {} species".format(len(genera)-1))
# axm.set_ylabel('Num transfers')
# axm.set_ylabel("Recombined fraction \n @ $d_c=10^{-4}$")
axm.set_ylabel("Transfer / divergence")
axm.grid(linestyle='--', axis='y')
axm.set_yscale('log')
# axm.set_ylim([-0.5, axm.get_ylim()[1]])
# axm.set_ylim([-0.05, 0.5])

ymax = axm.get_ylim()[1]
# adding connection lines
for i in range(3):
    ax = [ax1, ax2, ax3][i]
    cpl = ConnectionPatch((0, 0), (connection_l[i], ymax), "axes fraction", "data",
                          axesA=ax, axesB=axm, **kw)
    cpr = ConnectionPatch((1, 0), (connection_r[i], ymax), "axes fraction", "data",
                          axesA=ax, axesB=axm, **kw)
    axm.add_artist(cpl)
    axm.add_artist(cpr)

# axd.set_ylabel('Num transfers\n per 4d clonal snp')
axd.set_ylabel('Transfer length')
_ = axd.set_xticks([])
_ = axd.set_xticklabels([])
# axd.set_yticks([0, 0.25, 0.5, 0.75, 1])
# axd.set_yscale('log')
axd.grid(linestyle='--', axis='y')
axd.set_xlim(axm.get_xlim())
axd.set_ylim([0, 50e3])
axd.set_yticks([0, 20e3, 40e3])
axd.set_yticklabels([0, '20kb', '40kb'])
# axd.set_yscale('log')
axm.set_xlim([1, 2 * len(xticklabels) + 1])
axd.set_xlim([1, 2 * len(xticklabels) + 1])

_ = axd.set_xticks(xticks)
_ = axd.set_xticklabels(xticklabels, rotation=90, ha='center', fontsize=5)
json.dump(plotted_species, open(os.path.join(config.plotting_intermediate_directory, 'fig3_species.json'), 'w'))

fig.tight_layout()
fig.savefig(os.path.join(config.figure_directory, 'fig3_.pdf'), dpi=600)


total_events = 0
for lengths in transfer_length_data:
    total_events += len(lengths)
print("Total {} species; total {} pairs;  total {} detected events".format(len(transfer_length_data)-1, np.sum(all_num_pairs), total_events))
