import os
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.stats import ttest_ind
sys.path.append("..")
import config
from utils import close_pair_utils, parallel_utils, core_gene_utils, typical_pair_utils
from plotting_for_publication import plot_pileup_mirror

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 0.5

# mapping out grid
fig = plt.figure(figsize=(7, 4.5), dpi=600)

gs_pi = gridspec.GridSpec(1, 1)
gs_pi.update(left=0.36, right=0.98, top=0.95, bottom=0.86)

gs_Bv = gridspec.GridSpec(1, 1)
gs_Bv.update(left=0.36, right=0.98, top=0.84, bottom=0.59)

gs_neutral = gridspec.GridSpec(1, 1)
gs_neutral.update(left=0.36, right=0.98, top=0.52, bottom=0.395)

gs_Er = gridspec.GridSpec(1, 1)
gs_Er.update(left=0.36, right=0.98, top=0.33, bottom=0.08)

gs_cvs = gridspec.GridSpec(1, 1)
gs_cvs.update(left=0.075, right=0.275, top=0.33, bottom=0.08)

# adding axes
pi_ax = fig.add_subplot(gs_pi[0,0])
cvs_ax = fig.add_subplot(gs_cvs[0, 0])
Bv_ax = fig.add_subplot(gs_Bv[0,0])
neutral_ax = fig.add_subplot(gs_neutral[0,0])
Er_ax = fig.add_subplot(gs_Er[0,0])


####################### Plot CVs #######################
# load real data
cvs_path = os.path.join(config.plotting_intermediate_directory, 'species_sharing_pileup_cvs.csv')
if os.path.exists(cvs_path):
    real_cvs = np.loadtxt(cvs_path)
else:
    ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical')
    all_cvs = []
    for species in os.listdir(ckpt_dir):
        # species with > 50 contigs are not considered
        if species.startswith('.'):
            continue
        sharing_pileup = np.loadtxt(os.path.join(ckpt_dir, species, 'between_host.csv'))
        cv = np.std(sharing_pileup, axis=0) / np.mean(sharing_pileup, axis=0)
        all_cvs.append(cv)
    real_cvs = np.array(all_cvs)
    np.savetxt(cvs_path, real_cvs)
# load sim data
sim_cvs = np.loadtxt(os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'r_scan_statistics', 'cvs.csv'))
sim_cvs = sim_cvs.reshape((-1, 100))
mean_sim_cvs = sim_cvs.mean(axis=1)
sigma_sim_cvs = np.std(sim_cvs, axis=1)
print("Minimum real CV: {}".format(real_cvs.min()))
print("Number of species: {}".format(real_cvs.shape[0]))
print("Simulation CV mean: {}".format(mean_sim_cvs))
print("Simulation CV std: {}".format(sigma_sim_cvs))
print("Num std away: {}".format((real_cvs.min() - mean_sim_cvs)/sigma_sim_cvs))
ps = []
for i in range(sim_cvs.shape[0]):
    res = ttest_ind(real_cvs[:, 0], sim_cvs[i, :], equal_var=False)
    ps.append(res.pvalue)
print("worst t test pval: {}".format(max(ps)))

# plot comparison
cvs_ax.scatter(real_cvs[:, 0], np.ones(real_cvs.shape[0]) + np.random.uniform(-0.1, 0.1, size=real_cvs.shape[0]), marker='o', alpha=0.5)
cvs_ax.errorbar(mean_sim_cvs, np.zeros(mean_sim_cvs.shape) + np.random.uniform(-0.1, 0.1, size=mean_sim_cvs.shape), xerr=sigma_sim_cvs, fmt='o', color='grey', alpha=0.3)
cvs_ax.set_yticks([0, 1])
cvs_ax.set_yticklabels(['Neutral', 'Real'])
cvs_ax.set_xlabel('CV')


####################### Plot Bv #######################
def highlight_sweep_region(ax, genes):
    sweep_df = pd.read_csv(os.path.join(config.figure_directory, 'supp_table', 'temporal_sweep_genes.csv'))
    sweep_genes = [x.split('|')[-1] for x in sweep_df['PATRIC ID']]
    mask = np.isin(genes, sweep_genes)
    # using the run finding function to find the stretch
    runs, starts, ends= parallel_utils._compute_runs_single_chromosome(~mask, return_locs=True)
    for start, end in zip(starts, ends):
        ax.axvspan(start, end, color='red', alpha=0.2, linewidth=1, zorder=3)

species_name = 'Bacteroides_vulgatus_57955'
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

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955_between')
between_clade_path = os.path.join(base_path, 'between_host.csv')
thresholds = np.loadtxt(os.path.join(base_path, 'between_host_thresholds.txt'))

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', 'Bacteroides_vulgatus_57955')
within_clade_path = os.path.join(base_path, 'between_host.csv')
within_thresholds = np.loadtxt(os.path.join(base_path, 'between_host_thresholds.txt'))

within_cumu_runs, between_cumu_runs = plot_pileup_mirror.load_data_and_plot_mirror(
    within_clade_path, between_clade_path, Bv_ax, ind_to_plot=0, ylim=0.3)
print(within_thresholds[0], thresholds[0])
highlight_sweep_region(Bv_ax, good_genes)


####################### Plot pi #######################
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
pi_ax.plot(np.convolve(pi, kernel, mode='same'), color='tab:orange', label='within species')
pi_ax.plot(np.convolve(clade_pi, kernel, mode='same'), color='tab:blue', label='within largest clade')
# pi_ax.legend()
pi_ax.set_xlim([0, between_cumu_runs.shape[0]])
pi_ax.set_xlabel("Genome location")
pi_ax.set_ylabel('$\pi$')


# Plot neutral
ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'fastsimbac_rbymu_1')
for sim_id in range(400, 500):
    filename = '%d.txt' % sim_id
    save_path = os.path.join(ckpt_dir, filename)
    cumu_runs = np.loadtxt(save_path)
    if sim_id==450:
        neutral_ax.plot(cumu_runs, color='tab:blue', alpha=1, linewidth=1, zorder=4)
    else:
        neutral_ax.plot(cumu_runs, color='grey', linewidth=1, alpha=0.15, rasterized=True)
neutral_ax.set_ylim([0, 0.3])
neutral_ax.set_xlim([0, 280000])

# Plot Er
species_name = 'Eubacterium_rectale_56927'
base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
between_host_path = os.path.join(base_path, 'between_host.csv')
thresholds = np.loadtxt(os.path.join(base_path, 'between_host_thresholds.txt'))

base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
within_host_path = os.path.join(base_path, 'within_host.csv')
within_thresholds = np.loadtxt(os.path.join(base_path, 'within_host_thresholds.txt'))

between_cumu_runs, within_cumu_runs = plot_pileup_mirror.load_data_and_plot_mirror(
    between_host_path, within_host_path, Er_ax, ind_to_plot=0, ylim=0.5, colors=[config.between_host_color, config.within_host_color])

fig.savefig(os.path.join(config.figure_directory, 'final_fig', 'fig5.pdf'))
