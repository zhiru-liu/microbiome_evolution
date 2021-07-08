import numpy as np
import json
import sys
import os
import pandas as pd
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch

sys.path.append("..")
import config
from utils import close_pair_utils

from matplotlib.transforms import Bbox, TransformedBbox, \
    blended_transform_factory

from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, \
    BboxConnectorPatch

species_priority = json.load(open('species_plotting_priority.json', 'r'))
data_dir = os.path.join(config.analysis_directory, "closely_related")
files_to_plot = sorted(filter(lambda x: not x.startswith('.'), os.listdir(os.path.join(data_dir, 'fourth_pass'))), key=lambda x: species_priority.get(x.split('.')[0]))
files_to_plot.insert(10, files_to_plot.pop(2))

data_dir = os.path.join(config.analysis_directory, "closely_related")

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'
plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(10, 3)
gs.update(hspace=0)
ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[:3, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[:3, 2], sharex=ax1, sharey=ax1)
axm = fig.add_subplot(gs[4:7, :])
axd = fig.add_subplot(gs[7:, :])

idx = 1
xticks = []
xticklabels = []
connection_l = []
connection_r = []

transfer_length_data = []
plot_loc = []

for filename in files_to_plot:
    species_name = ' '.join(filename.split('.')[0].split('_')[:2])
    raw_data = pd.read_pickle(os.path.join(data_dir, 'third_pass', filename.split('.')[0] + '.pickle'))

    if 'vulgatus' in species_name:
        data_path = os.path.join(data_dir, 'third_pass', filename.split('.')[0] + '_all_transfers_two_clades.pickle')
    else:
        data_path = os.path.join(data_dir, 'third_pass', filename.split('.')[0] + '_all_transfers.pickle')

    transfer_data = pd.read_pickle(data_path)

    n = raw_data.shape[0]
    if n < 100:
        continue
    print(filename, n, raw_data['clonal snps'].max())
    transfer_length_data.append(transfer_data['lengths'].to_numpy().astype(float) * config.second_pass_block_size)
    fitted_data = pd.read_csv(os.path.join(data_dir, 'fourth_pass', filename), index_col=0)

    x, y = close_pair_utils.prepare_x_y(raw_data)

    f1 = interpolate.interp1d(fitted_data['x'], fitted_data['y'], bounds_error=False)
    f2 = interpolate.interp1d(fitted_data['x'], fitted_data['sigma'], bounds_error=False)

    # plotting species summary
    plot_loc.append(2 * idx)
    mid = f1([10, 20, 30, 40])
    w = f2([10, 20, 30, 40])
    xloc = np.linspace(2 * idx - 0.3, 2 * idx + 0.3, 4, endpoint=True)
    axm.scatter(xloc, mid, s=5)
    axm.plot(xloc, mid, linestyle=':', linewidth=1)
    axm.vlines(xloc, mid - w, mid + w, alpha=0.2)

    # plotting transfer per snp
    #     axd.scatter(xloc, np.divide(mid, [10, 20, 30, 40]), s=5)
    #     axd.vlines(xloc, np.divide(mid-w, [10, 20, 30, 40]),
    #                np.divide(mid+w, [10, 20, 30, 40]), alpha=0.2)
    # plotting mean transfer length
    # axd.scatter(2 * idx, transfer_data['lengths'].median() * config.second_pass_block_size, s=5)

    kw = dict(linestyle=":", linewidth=1, color='grey')
    if 'vulgatus' in species_name:
        # for B vulgatus
        # plot for within transfer first
        vb1 = axd.violinplot([config.second_pass_block_size * transfer_data[transfer_data['types']==0]['lengths'].astype(float)],
                             positions=plot_loc, vert=True, showmedians=True, showextrema=False, widths=0.8)
        for b in vb1['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further right than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            b.set_facecolor(plot_colors[0])
            b.set_alpha(0.5)
            b.set_edgecolor('black')

        vb2 = axd.violinplot([config.second_pass_block_size * transfer_data[transfer_data['types']==1]['lengths'].astype(float)],
                             positions=plot_loc, vert=True, showmedians=True, showextrema=False, widths=0.8)
        for b in vb2['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further left than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_alpha(0.5)
            b.set_facecolor(plot_colors[0])
            b.set_edgecolor('black')

    if 'caccae' in species_name:
        ax1.scatter(x, y, s=1)
        ax1.plot(fitted_data['x'], fitted_data['y'])
        ax1.fill_between(fitted_data['x'], fitted_data['y'] - fitted_data['sigma'],
                         fitted_data['y'] + fitted_data['sigma'], alpha=0.25)
        ax1.set_title(species_name)
        axm.axvline(xloc[0] - 0.4, **kw)
        axm.axvline(xloc[-1] + 0.4, **kw)
        axd.axvline(xloc[0] - 0.4, **kw)
        axd.axvline(xloc[-1] + 0.4, **kw)
        connection_l.append(xloc[0] - 0.4)
        connection_r.append(xloc[-1] + 0.4)

    if 'massiliensis' in species_name:
        ax2.scatter(x, y, s=1)
        ax2.plot(fitted_data['x'], fitted_data['y'])
        ax2.fill_between(fitted_data['x'], fitted_data['y'] - fitted_data['sigma'],
                         fitted_data['y'] + fitted_data['sigma'], alpha=0.25)
        ax2.set_title(species_name)
        axm.axvline(xloc[0] - 0.4, **kw)
        axm.axvline(xloc[-1] + 0.4, **kw)
        axd.axvline(xloc[0] - 0.4, **kw)
        axd.axvline(xloc[-1] + 0.4, **kw)
        connection_l.append(xloc[0] - 0.4)
        connection_r.append(xloc[-1] + 0.4)

    if 'putredinis' in species_name:
        ax3.scatter(x, y, s=1)

        # select three examples pairs
        example_mask = raw_data['pairs'].isin([(282, 387), (297, 331), (269, 313)])
        ax3.scatter(x[example_mask], y[example_mask], s=1, color='r')

        ax3.plot(fitted_data['x'], fitted_data['y'])
        ax3.set_xlim([0, 50])
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
    idx += 1

# plotting all other violin plots
violins = axd.violinplot(transfer_length_data[1:], positions=plot_loc[1:], vert=True, showmedians=True, showextrema=False,
                         widths=0.8)

for i, pc in enumerate(violins['bodies']):
    pc.set_facecolor(plot_colors[(i + 1) % len(plot_colors)])
    pc.set_alpha(0.5)
    pc.set_edgecolor('black')
violins['cmedians'].set_edgecolor('black')

axm.set_ylabel('Num transfers')
axm.grid(linestyle='--', axis='y')
axm.set_ylim([-5, axm.get_ylim()[1]])

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
axd.set_ylabel('Median transfer length')
_ = axd.set_xticks([])
_ = axd.set_xticklabels([])
# axd.set_yticks([0, 0.25, 0.5, 0.75, 1])
# axd.set_yscale('log')
axd.grid(linestyle='--', axis='y')
axd.set_ylim([0, 5900])

_ = axd.set_xticks(xticks)
_ = axd.set_xticklabels(xticklabels, rotation=90, ha='center', fontsize=5)

fig.tight_layout()
fig.savefig(os.path.join(config.analysis_directory, 'closely_related', 'summary_v4.pdf'), dpi=300)