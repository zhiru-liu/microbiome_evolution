import os
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.style import context
from matplotlib import cm
import pandas as pd
from scipy import interpolate

import config
from utils import close_pair_utils, diversity_utils, parallel_utils, typical_pair_utils

species_name = "Alistipes_putredinis_61533"
good_sample_indices = typical_pair_utils.load_single_subject_sample_idxs(species_name)

# loading hmm results
df = pd.read_pickle(os.path.join(config.analysis_directory, 'closely_related', 'third_pass', 'Alistipes_putredinis_61533.pickle'))
data_path = os.path.join(config.analysis_directory, "closely_related", 'third_pass', species_name + '_all_transfers.pickle')
run_df = pd.read_pickle(data_path)

Tc_cutoffs = json.load(open(os.path.join(config.analysis_directory, 'misc', 'Tc_cutoffs.json'), 'r'))

def compute_downsampling():
    size_list = np.arange(50, len(good_sample_indices), 10)
    size_list = np.hstack([size_list, len(good_sample_indices)])
    rep_size = np.ones(size_list.shape) * 100
    rep_size[-1] = 1

    TcTm = []
    TcTm_std = []
    transfer_len = []
    transfer_len_std = []
    for i, downsample_size in enumerate(size_list):
        curr_TcTm = []
        curr_transfer_len = []
        for i in range(int(rep_size[i])):
            downsampled_hosts = np.random.choice(good_sample_indices, downsample_size, replace=False)
            downsampled_hosts.sort()
            mask = np.isin(df['pair 1'], downsampled_hosts) & np.isin(df['pair 2'], downsampled_hosts)
            downsampled_df = df[mask]

            x, y = close_pair_utils.prepare_x_y(downsampled_df, mode='fraction')
            # x, y = close_pair_utils.prepare_x_y(df, mode='fraction')
            x_ = x[x > 0]
            y_ = y[x > 0]
            rates = y_ / x_
            Tm = 1 / np.mean(rates)

            Tc = typical_pair_utils._compute_theta(species_name, good_sample_indices, clade_cutoff=Tc_cutoffs.get(species_name, [None, None]))
            curr_TcTm.append(Tc / Tm)

            # computing run length
            clonal_div_cutoff = 0.0002  # defined for Ap
            good_pairs = downsampled_df[(downsampled_df['clonal fractions'] >= config.clonal_fraction_cutoff) & (downsampled_df['clonal divs'] <= clonal_div_cutoff)]['pairs']
            mask = run_df['pairs'].isin(good_pairs)
            sub_df = run_df[mask]
            runs = sub_df[sub_df['types'] == 0]['transfer lengths (core genome)'].to_numpy().astype(float)
            curr_transfer_len.append(np.median(runs))
        TcTm.append(np.mean(curr_TcTm))
        transfer_len.append(np.mean(curr_transfer_len))
        TcTm_std.append(np.std(curr_TcTm))
        transfer_len_std.append(np.std(curr_transfer_len))

    savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_sizes.csv")
    np.savetxt(savepath, size_list)
    savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_TcTm.csv")
    np.savetxt(savepath, TcTm)
    savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_TcTm_std.csv")
    np.savetxt(savepath, TcTm_std)
    savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_transfer_len.csv")
    np.savetxt(savepath, transfer_len)
    savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_transfer_len_std.csv")
    np.savetxt(savepath, transfer_len_std)

# only need to run this once
#compute_downsampling()

savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_sizes.csv")
size_list = np.loadtxt(savepath)
savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_TcTm.csv")
TcTm = np.loadtxt(savepath)
savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_TcTm_std.csv")
TcTm_std = np.loadtxt(savepath)
savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_transfer_len.csv")
transfer_len = np.loadtxt(savepath)
savepath = os.path.join(config.plotting_intermediate_directory, "downsampling_transfer_len_std.csv")
transfer_len_std = np.loadtxt(savepath)

# setting up figures
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['legend.frameon'] = False

fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
axes[0].errorbar(size_list[:-1], TcTm[:-1], yerr=TcTm_std[:-1], color='tab:blue', fmt='.')
axes[0].axhline(TcTm[-1], alpha=0.3, label='all data')
axes[0].set_xlabel("Number of samples")
axes[0].set_ylabel(r"$T_{mrca}/T_{mosaic}$")

axes[1].errorbar(size_list[:-1], transfer_len[:-1], yerr=transfer_len_std[:-1], color='tab:blue', fmt='.')
axes[1].axhline(transfer_len[-1], alpha=0.3, label='all data')
axes[1].set_xlabel("Number of samples")
axes[1].set_ylabel("Median transfer length")
axes[0].legend()
axes[1].legend()
plt.tight_layout()
fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_downsampling_Ap.pdf'), dpi=600)