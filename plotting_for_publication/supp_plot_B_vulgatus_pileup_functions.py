import pandas as pd
import os
import numpy as np
import config
from utils import snp_data_utils, core_gene_utils
import matplotlib.pyplot as plt
from plotting_for_publication import plot_pileup_mirror

species_name = 'Bacteroides_vulgatus_57955'

# loading the gene name array
data_dir = os.path.join(config.data_directory, 'zarr_snps', species_name, 'site_info.txt')
res = snp_data_utils.parse_snp_info(data_dir)
chromosomes = res[0]
gene_names = res[2]
variants = res[3]
pvalues = res[4]

core_genes = core_gene_utils.get_sorted_core_genes(species_name)
general_mask = parallel_utils._get_general_site_mask(
    gene_names, variants, pvalues, core_genes, allowed_variants=['4D'])

good_genes = gene_names[general_mask]

# loading gene annotations
all_genes = pd.read_csv(os.path.join(config.data_directory, 'genome_features', '%s.csv'%species_name))
gene_annotation_dict = dict(zip(all_genes['PATRIC ID'].apply(lambda x: x.split('|')[1]), all_genes['Product']))

all_products = [gene_annotation_dict.get(x) for x in good_genes]
PUL_locs = np.array(['PUL' in x for x in all_products]).astype(int)
transferase_locs = np.array([('Glycosyltransferase' in x) or ('Glycosyl transferase' in x) or ('glycosyltransferase' in x) for x in all_products]).astype(int)

# setting up figure
fig, axes = plt.subplots(2, 1, figsize=(7, 5))

# plot data
base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955_between')
within_path = os.path.join(base_path, 'within_host.csv')
between_path = os.path.join(base_path, 'between_host.csv')
thresholds = np.loadtxt(os.path.join(base_path, 'between_host_thresholds.txt'))

between_cumu_runs, within_cumu_runs = plot_pileup_mirror.load_data_and_plot_mirror(
    between_path, within_path, axes[1], threshold_lens=thresholds, ind_to_plot=[1, 2, 3], ylim=0.3)

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')
within_path = os.path.join(base_path, 'within_host.csv')
between_path = os.path.join(base_path, 'between_host.csv')
within_thresholds = np.loadtxt(os.path.join(base_path, 'within_host_thresholds.txt'))
bh_between_cumu_runs, bh_within_cumu_runs = plot_pileup_mirror.load_data_and_plot_mirror(
    between_path, within_path, axes[0], threshold_lens=within_thresholds, ind_to_plot=[0, 2, 4], ylim=0.3)

axes[0].get_xaxis().set_visible(False)
axes[0].set_ylabel('same-clade\nsharing fraction')
axes[1].set_ylabel('diff-clade\nsharing fraction')
fig.subplots_adjust(hspace=0)

# plotting all the highlighted regions
axes[0].plot(transferase_locs, 'r', alpha=0.2)
axes[0].plot(PUL_locs, 'b', alpha=0.2)
axes[1].plot(transferase_locs, 'r', alpha=0.2)
axes[1].plot(PUL_locs, 'b', alpha=0.2)

axes[0].plot(-transferase_locs, 'r', alpha=0.2)
axes[0].plot(-PUL_locs, 'b', alpha=0.2)
axes[1].plot(-transferase_locs, 'r', alpha=0.2)
axes[1].plot(-PUL_locs, 'b', alpha=0.2)

# hacking way to add legend
axes[0].axvspan(-5, -1, alpha=0.1, color='r', label='Glycosyltransferase')
axes[0].axvspan(-5, -1, alpha=0.1, color='b', label='PUL')
axes[0].legend(bbox_to_anchor=(1, 1))

fig.savefig(os.path.join(config.analysis_directory, 'misc', 'quad_pileup_B_vulgatus_with_highlight.pdf'), bbox_inches='tight')