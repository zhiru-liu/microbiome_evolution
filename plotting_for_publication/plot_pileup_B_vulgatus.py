import pandas as pd
import os
import numpy as np
import config
from utils import snp_data_utils, core_gene_utils, typical_pair_utils, pileup_utils
import matplotlib.pyplot as plt
from plotting_for_publication import plot_pileup_mirror

species_name = 'Bacteroides_vulgatus_57955'
compute_function_enrichment = False
plot_polymorphism = True
plot_within_host = False

# loading the gene name array
data_dir = os.path.join(config.data_directory, 'zarr_snps', species_name, 'site_info.txt')
res = parallel_utils.parse_snp_info(data_dir)
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


def highlight_sweep_region(ax, genes):
    sweep_df = pd.read_csv(os.path.join(config.figure_directory, 'supp_table', 'temporal_sweep_genes.csv'))
    sweep_genes = [x.split('|')[-1] for x in sweep_df['PATRIC ID']]
    mask = np.isin(genes, sweep_genes)
    # using the run finding function to find the stretch
    runs, starts, ends= parallel_utils._compute_runs_single_chromosome(~mask, return_locs=True)
    for start, end in zip(starts, ends):
        ax.axvspan(start, end, color='red', alpha=0.2, linewidth=1, zorder=3)


def prepare_features(gene_df, good_genes, pileup_data):
    features = []
    for index, row in gene_df.iterrows():
        start = row['Start']
        end = row['End']
        strand = 1 if row['Strand'] == '+' else -1
        label = row['Product']
        if '(' in label:
            label = label.split('(')[0]
        if 'hypothetical' in label:
            label = None
        gene_name = row['PATRIC ID'].split('|')[-1]
        if gene_name not in good_genes:
            mean_sharing = -1  # non core
        else:
            mean_sharing = pileup_data[good_genes == gene_name].mean()
        features.append((start, end, strand, label, mean_sharing))
    annotated_genes = pd.DataFrame(features)
    annotated_genes.columns = ['start', 'end', 'strand', 'label', 'mean sharing']
    annotated_genes.to_csv(os.path.join(config.plotting_intermediate_directory, 'Bv_annotated_genes.csv'))


""" Plot within clade between clade """
# setting up figure
fig, ax = plt.subplots(1, 1, figsize=(7, 3))

# plot data
base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955_between')
between_clade_path = os.path.join(base_path, 'between_host.csv')
thresholds = np.loadtxt(os.path.join(base_path, 'between_host_thresholds.txt'))

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')
within_clade_path = os.path.join(base_path, 'between_host.csv')
within_thresholds = np.loadtxt(os.path.join(base_path, 'between_host_thresholds.txt'))

within_cumu_runs, between_cumu_runs = plot_pileup_mirror.load_data_and_plot_mirror(
    within_clade_path, between_clade_path, ax, ind_to_plot=0, ylim=0.3)
highlight_sweep_region(ax, good_genes)
prepare_features(all_genes, good_genes, within_cumu_runs[:, 0] / within_cumu_runs[:, 0].max())
# printing the mean or median of sharing fraction in order to match in simulated data
print(between_cumu_runs[:, 0].mean(), within_cumu_runs[:, 0].mean())
print(np.median(between_cumu_runs[:, 0]), np.median(within_cumu_runs[:, 0]))


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

fig.savefig(os.path.join(config.analysis_directory, 'misc', 'B_vulgatus_pileup_test.pdf'), bbox_inches='tight')
plt.close()


""" Compute enrichment at glycosyltransferases and ribosomal proteins """
if compute_function_enrichment:
    reps = 1e5
    def if_gt(gene):
        x = gene_annotation_dict[gene]
        return ('Glycosyltransferase' in x) or ('Glycosyl transferase' in x) or ('glycosyltransferase' in x)
    site_mask = between_cumu_runs[:, 0] > 0.1
    true_res, perm_res = pileup_utils.enrichment_test(good_genes, site_mask, if_gt, shuffle_size=10, shuffle_reps=int(reps))
    print("p-val for glycosyltransferases is {:e}".format(np.sum(np.array(perm_res) >= true_res) / reps))
    # np.savetxt("gt_perm_counts.txt", perm_res)

    def if_rp(gene):
        x = gene_annotation_dict[gene]
        return 'ribosomal protein' in x
    site_mask = within_cumu_runs[:, 0] > 0.1
    true_res, perm_res = pileup_utils.enrichment_test(good_genes, site_mask, if_rp, shuffle_size=10, shuffle_reps=int(reps))
    print("p-val for ribosomal protein is {:e}".format(np.sum(np.array(perm_res) >= true_res) / reps))

""" Plot within host between host """
if plot_within_host:
    # setting up figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    # plot data
    base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')
    within_host_path = os.path.join(base_path, 'within_host.csv')

    base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')
    between_host_path = os.path.join(base_path, 'between_host.csv')

    within_cumu_runs, between_cumu_runs = plot_pileup_mirror.load_data_and_plot_mirror(
        between_host_path, within_host_path, ax, ind_to_plot=0, ylim=0.3)
    fig.savefig(os.path.join(config.analysis_directory, 'misc', 'B_vulgatus_pileup_same_clade.pdf'), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    # plot data
    base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955_between')
    within_host_path = os.path.join(base_path, 'within_host.csv')

    base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955_between')
    between_host_path = os.path.join(base_path, 'between_host.csv')

    within_cumu_runs, between_cumu_runs = plot_pileup_mirror.load_data_and_plot_mirror(
        between_host_path, within_host_path, ax, ind_to_plot=0, ylim=0.4)
    fig.savefig(os.path.join(config.analysis_directory, 'misc', 'B_vulgatus_pileup_diff_clade.pdf'), bbox_inches='tight')
    plt.close()

""" Plot polymorphism """
if plot_polymorphism:
    fig, ax = plt.subplots(figsize=(7, 2))
    polymorphism_dir = os.path.join(config.plotting_intermediate_directory, 'B_vulgatus_polymorphism.csv')
    if os.path.exists(polymorphism_dir):
        pis = np.loadtxt(polymorphism_dir)
        pi = pis[:, 0]
        clade_pi = pis[:, 1]
    else:
        dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])
        pi, clade_pi = typical_pair_utils.get_sitewise_polymorphism(dh, clade_cutoff=0.03)
        np.savetxt(os.path.join(config.plotting_intermediate_directory, 'B_vulgatus_polymorphism.csv'),
                   np.vstack([pi, clade_pi]).transpose())
    kernel = np.ones(3000) / 3000.
    ax.plot(np.convolve(pi, kernel, mode='same'), color='tab:grey', label='within species')
    ax.plot(np.convolve(clade_pi, kernel, mode='same'), color='tab:blue', label='within largest clade')
    ax.legend()
    ax.set_xlim([0, between_cumu_runs.shape[0]])
    ax.set_xlabel("Genome location")
    ax.set_ylabel('$\pi$')
    fig.savefig(os.path.join(config.analysis_directory, 'misc', 'B_vulgatus_polymorphism.pdf'), bbox_inches='tight')
