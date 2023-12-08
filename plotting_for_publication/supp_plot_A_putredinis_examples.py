import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append("..")
import config
from utils import snp_data_utils, close_pair_utils

species_name = 'Alistipes_putredinis_61533'
dh = parallel_utils.DataHoarder(species_name, mode="QP", allowed_variants=['4D'])
chromosomes = dh.chromosomes[dh.general_mask]

# loading close pair analysis data
save_path = os.path.join(config.analysis_directory,
                         "closely_related", "third_pass", "{}.pickle".format(species_name))
df = pd.read_pickle(save_path)

save_path = os.path.join(config.analysis_directory,
                         "closely_related", "third_pass", "{}_all_transfers.pickle".format(species_name))
full_df = pd.read_pickle(save_path)


# set up figure
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

fig, axes = plt.subplots(3, 1, figsize=(5, 2.5))

for i, pair in enumerate([(297, 331), (282, 387), (269, 313)]):
    snp_vec, covered_mask = dh.get_snp_vector(pair)
    window_size = 1000
    local_pi = np.convolve(snp_vec, np.ones(window_size) / float(window_size), mode='same')

    shown_snp_locs = np.nonzero(snp_vec)[0]

    ax = axes[i]
    ax.plot(local_pi, label='local heterozygosity')
    ax.plot(shown_snp_locs, np.zeros(len(shown_snp_locs)), '|', label='individual snps', color='grey')

    block_size = 10
    sub_df = full_df[full_df['pairs'] == pair]

    good_chromo = chromosomes[covered_mask]
    contig_lengths = parallel_utils.get_contig_lengths(good_chromo)

    xs = np.arange(len(snp_vec))
    ys = np.zeros(xs.shape)
    for _, row in sub_df.iterrows():
        start = close_pair_utils.block_loc_to_genome_loc(row['starts'], contig_lengths, block_size, left=True)
        end = close_pair_utils.block_loc_to_genome_loc(row['ends'], contig_lengths, block_size, left=False)
        ys[int(start):int(end)] = row['types'] + 1

    ax.plot(xs, -ys * 0.015, label='detected transfers', color='tab:green')
    # ax.set_title(pair)
    ax.set_ylabel(pair)

    clonal_snp_locs = np.nonzero(snp_vec & (~ys.astype(bool)))
    ax.plot(clonal_snp_locs, np.zeros(len(clonal_snp_locs)), '|', color='r')

    for loc in parallel_utils.get_contig_boundary(good_chromo):
        ax.axvline(loc, linestyle='--', linewidth=1, color='grey')

    ax.set_xlim([0, len(snp_vec)])
    ax.set_ylim([-0.018, 0.065])
    # ax.set_xlim([0, 100000])
    if i==0:
        ax.legend(loc='upper left')
    if i==2:
        ax.set_xlabel('Synonymous core genome location')
    else:
        ax.set_xticklabels([])
plt.tight_layout()
fig.savefig(os.path.join(config.analysis_directory, 'closely_related', 'A_putredinis_examples.pdf'), dpi=300)