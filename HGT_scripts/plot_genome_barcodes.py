import os
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.style import context
from matplotlib import cm
import pandas as pd
import itertools
import pickle
import config
from utils import parallel_utils, close_pair_utils

run_data_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data')

species_name = 'Bacteroides_vulgatus_57955'
dh_wt = parallel_utils.DataHoarder(species_name, mode='within')
dh_bt = parallel_utils.DataHoarder(species_name, mode='QP')

save_path = os.path.join(run_data_dir, 'within_hosts', '{}_same_clade.pickle'.format(species_name))
within_runs_data = pickle.load(open(save_path, 'rb'))
save_path = os.path.join(run_data_dir, 'between_hosts', '{}_same_clade.pickle'.format(species_name))
between_runs_data = pickle.load(open(save_path, 'rb'))

num_barcodes = len(within_runs_data)
fig, axes = plt.subplots(len(within_runs_data)+1, 1, figsize=(5, int(num_barcodes / 3)), dpi=600)
xlim = 1e6
for i, idx in enumerate(within_runs_data):
    snp_vec, _ = dh_wt.get_snp_vector(idx)
    snp_blk = close_pair_utils.to_block(snp_vec, 1000)
    barcode = np.concatenate([snp_blk>0, [0]])
    xlim = min(xlim, len(snp_blk)-1)
    axes[i].imshow(np.expand_dims(barcode, axis=0), aspect='auto',
                   cmap = mpl.colors.ListedColormap(['white', 'tab:blue']), interpolation='nearest')
    axes[i].set_yticklabels([])
    axes[i].set_xticklabels([])
    axes[i].set_yticks([])

# for showing the pixel size
bwarr = np.resize([1,-1], xlim)
axes[-1].imshow(np.expand_dims(bwarr, axis=0), aspect='auto',
                   cmap = mpl.colors.ListedColormap(['white', 'tab:blue']), interpolation='nearest')
for i in range(len(axes)):
    axes[i].set_xlim([0, xlim])
fig.savefig(os.path.join(config.analysis_directory, 'misc', 'Bv_barcodes', 'within_host_same_clade.pdf'))
plt.close()

fig, axes = plt.subplots(len(within_runs_data), 1, figsize=(5, int(num_barcodes / 3)), dpi=600)
xlim = 1e6
sampled_between_pairs = random.sample(between_runs_data.keys(), len(within_runs_data))
for i, idx in enumerate(sampled_between_pairs):
    snp_vec, _ = dh_bt.get_snp_vector(idx)
    snp_blk = close_pair_utils.to_block(snp_vec, 1000)
    barcode = np.concatenate([snp_blk > 0, [0]])
    xlim = min(xlim, len(snp_blk))
    axes[i].imshow(np.expand_dims(barcode, axis=0), aspect='auto',
                   cmap=mpl.colors.ListedColormap(['white', 'tab:blue']), interpolation='nearest')
    axes[i].set_yticklabels([])
    axes[i].set_xticklabels([])
    axes[i].set_yticks([])

for i in range(len(axes)):
    axes[i].set_xlim([0, xlim])
fig.savefig(os.path.join(config.analysis_directory, 'misc', 'Bv_barcodes', 'between_host_same_clade.pdf'))
plt.close()


save_path = os.path.join(run_data_dir, 'within_hosts', '{}_diff_clade.pickle'.format(species_name))
within_runs_data = pickle.load(open(save_path, 'rb'))
save_path = os.path.join(run_data_dir, 'between_hosts', '{}_diff_clade.pickle'.format(species_name))
between_runs_data = pickle.load(open(save_path, 'rb'))

num_barcodes = len(within_runs_data)
fig, axes = plt.subplots(len(within_runs_data)+1, 1, figsize=(5, int(num_barcodes / 3)), dpi=600)
xlim = 1e6
for i, idx in enumerate(within_runs_data):
    snp_vec, _ = dh_wt.get_snp_vector(idx)
    snp_blk = close_pair_utils.to_block(snp_vec, 500)
    barcode = np.concatenate([snp_blk>0, [0]])
    xlim = min(xlim, len(snp_blk)-1)
    axes[i].imshow(np.expand_dims(barcode, axis=0), aspect='auto',
                   cmap = mpl.colors.ListedColormap(['white', 'tab:blue']), interpolation='nearest')
    axes[i].set_yticklabels([])
    axes[i].set_xticklabels([])
    axes[i].set_yticks([])

# for showing the pixel size
bwarr = np.resize([1,-1], xlim)
axes[-1].imshow(np.expand_dims(bwarr, axis=0), aspect='auto',
                cmap = mpl.colors.ListedColormap(['white', 'tab:blue']), interpolation='nearest')
for i in range(len(axes)):
    axes[i].set_xlim([0, xlim])
fig.savefig(os.path.join(config.analysis_directory, 'misc', 'Bv_barcodes', 'within_host_diff_clade.pdf'))
plt.close()

fig, axes = plt.subplots(len(within_runs_data), 1, figsize=(5, int(num_barcodes / 3)), dpi=600)
xlim = 1e6
sampled_between_pairs = random.sample(between_runs_data.keys(), len(within_runs_data))
for i, idx in enumerate(sampled_between_pairs):
    snp_vec, _ = dh_bt.get_snp_vector(idx)
    snp_blk = close_pair_utils.to_block(snp_vec, 500)
    barcode = np.concatenate([snp_blk > 0, [0]])
    xlim = min(xlim, len(snp_blk))
    axes[i].imshow(np.expand_dims(barcode, axis=0), aspect='auto',
                   cmap=mpl.colors.ListedColormap(['white', 'tab:blue']), interpolation='nearest')
    axes[i].set_yticklabels([])
    axes[i].set_xticklabels([])
    axes[i].set_yticks([])

for i in range(len(axes)):
    axes[i].set_xlim([0, xlim])
fig.savefig(os.path.join(config.analysis_directory, 'misc', 'Bv_barcodes', 'between_host_diff_clade.pdf'))
plt.close()
