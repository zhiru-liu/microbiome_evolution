import pandas as pd
import os
import numpy as np
import config
from scipy.stats import fisher_exact
from utils import snp_data_utils, core_gene_utils
import matplotlib.pyplot as plt
from plotting_for_publication import plot_pileup_mirror

species_name = 'Eubacterium_rectale_56927'

# loading the gene name array
data_dir = os.path.join(config.data_directory, 'zarr_snps', species_name, 'site_info.txt')
res = parallel_utils.parse_snp_info(data_dir)
chromosomes = res[0]
gene_names = res[2]
variants = res[3]
pvalues = res[4]

core_genes = core_gene_utils.get_sorted_core_genes(species_name)
general_mask = snp_data_utils._get_general_site_mask(
    gene_names, variants, pvalues, core_genes, allowed_variants=['4D'])

good_genes = gene_names[general_mask]

# loading gene annotations
all_genes = pd.read_csv(os.path.join(config.data_directory, 'genome_features', '%s.csv'%species_name))
gene_annotation_dict = dict(zip(all_genes['PATRIC ID'].apply(lambda x: x.split('|')[1]), all_genes['Product']))

all_products = [gene_annotation_dict.get(x) for x in good_genes]
PUL_locs = np.array(['PUL' in x for x in all_products]).astype(int)
transferase_locs = np.array([('Glycosyltransferase' in x) or ('Glycosyl transferase' in x) or ('glycosyltransferase' in x) for x in all_products]).astype(int)

# setting up figure
fig, ax = plt.subplots(1, 1, figsize=(7, 3))

# plot data
base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
between_host_path = os.path.join(base_path, 'between_host.csv')
thresholds = np.loadtxt(os.path.join(base_path, 'between_host_thresholds.txt'))

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
within_host_path = os.path.join(base_path, 'within_host.csv')
within_thresholds = np.loadtxt(os.path.join(base_path, 'within_host_thresholds.txt'))

between_cumu_runs, within_cumu_runs = plot_pileup_mirror.load_data_and_plot_mirror(
    between_host_path, within_host_path, ax, ind_to_plot=0, ylim=0.5)


# computing p value for within vs between differences using Fisher exact test
between_comparisons = 2999  # recorded from pile up computation
within_comparisons = 28
between_counts = between_cumu_runs[:, 1] * between_comparisons
within_counts = within_cumu_runs[:, 1] * within_comparisons

fisher_table = np.ones((between_cumu_runs.shape[0], 2, 2))
fisher_table[:, 0, 0] = within_counts
fisher_table[:, 0, 1] = between_counts
fisher_table[:, 1, 0] = within_comparisons - within_counts
fisher_table[:, 1, 1] = between_comparisons - between_counts

pgt = np.zeros(between_cumu_runs.shape[0])
pls = np.zeros(between_cumu_runs.shape[0])
# should take ~1min to compute all sites
for site in xrange(between_cumu_runs.shape[0]):
    _, p = fisher_exact(fisher_table[site, :, :], alternative='greater')
    pgt[site] = p
    _, p = fisher_exact(fisher_table[site, :, :], alternative='less')
    pls[site] = p

# TODO: plot the p values appropriately

# plotting all the highlighted regions
# ax.plot(transferase_locs, 'r', alpha=0.2)
# ax.plot(PUL_locs, 'b', alpha=0.2)
# ax.plot(-transferase_locs, 'r', alpha=0.2)
# ax.plot(-PUL_locs, 'b', alpha=0.2)
#
# # hacking way to add legend
# ax.axvspan(-5, -1, alpha=0.1, color='r', label='Glycosyltransferase')
# ax.axvspan(-5, -1, alpha=0.1, color='b', label='PUL')
# ax.legend(bbox_to_anchor=(1, 1))

fig.savefig(os.path.join(config.analysis_directory, 'misc', 'E_rectale_pileup.pdf'), bbox_inches='tight')