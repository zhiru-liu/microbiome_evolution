import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
import sys
import os
sys.path.append("..")
from parsers import parse_midas_data
from utils import snp_data_utils, sfs_utils, HGT_utils, stats_utils
import config


# fig, haploid_axis = plt.subplots(figsize=(1.5, 3))
fig = plt.figure(figsize=(6.5, 4))
gs_bar = gridspec.GridSpec(1, 1)
gs_ex = gridspec.GridSpec(3, 2)

gs_bar.update(left=0.6, right=0.79, top=0.98, bottom=0.12)
gs_ex.update(left=0.08, right=0.55, top=0.98, bottom=0.12)

# adding axes
ex_axes = []
for i in range(3):
    ex_axes.append([fig.add_subplot(gs_ex[i, j]) for j in range(2)])
haploid_axis = fig.add_subplot(gs_bar[0, 0])

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

haploid_color = '#08519c'
light_haploid_color = '#6699CC'
good_witin_color = '#ef8a62'

sample_df = parallel_utils.compute_good_sample_stats()
sample_df = sample_df.sort_values('num_high_coverage_samples', ascending=False)
num_qp_samples = sample_df['num_qp_samples']
num_samples = sample_df['num_high_coverage_samples']
num_within_samples = sample_df['num_good_within_samples']

ys = 0-np.arange(0,len(num_qp_samples))

haploid_axis.barh(ys+0.5, num_qp_samples,color=haploid_color,linewidth=0,label='QP',zorder=1)
haploid_axis.barh(ys+0.5, num_samples,color=light_haploid_color,linewidth=0,zorder=0)
haploid_axis.barh(ys+0.5, num_within_samples,left=num_qp_samples,color=good_witin_color,linewidth=0,label='simple \nco-colonized')
haploid_axis.set_xlim([0,800])
haploid_axis.set_xticks([0,200,400,600,800])
haploid_axis.set_xlabel("Num samples")

haploid_axis.yaxis.tick_right()
haploid_axis.xaxis.tick_bottom()

haploid_axis.set_yticks(ys+0.5)
species_pretty_names = [' '.join(species_full_name.split('_')[:2]) for species_full_name in sample_df['species_name']]
haploid_axis.set_yticklabels(species_pretty_names,fontsize=6)
haploid_axis.set_ylim([-1*len(num_qp_samples)+0.5,1.5])

haploid_axis.tick_params(axis='y', direction='out',length=3,pad=1)

haploid_axis.legend(loc='lower right',frameon=False)

print("Total {} species".format(sample_df.shape[0]))



# Now plot the SFSs
example_SFSs = [['700161742', '700122408'],
                ['700015181', 'ERR912117'],
                ['700095410', '700095486']]
species_name = 'Bacteroides_vulgatus_57955'
samples, sfs_map = parse_midas_data.parse_within_sample_sfs(
        species_name, allowed_variant_types=set(['4D']))

sample_coverage_histograms, samples = parse_midas_data.parse_coverage_distribution(species_name)
median_coverages = np.array([stats_utils.calculate_nonzero_median_from_histogram(sample_coverage_histogram) for sample_coverage_histogram in sample_coverage_histograms])
sample_coverage_map = {samples[i]: median_coverages[i] for i in xrange(0,len(samples))}
for i in range(3):
    for j in range(2):
        ax = ex_axes[i][j]
        sample = example_SFSs[i][j]
        all_fs, all_pfs = sfs_utils.calculate_binned_sfs_from_sfs_map(
            sfs_map[sample], folding='major')
        df = all_fs[1] - all_fs[0]
        # For peak finding, only use the polymorphic sites
        pfs = all_pfs[all_fs < 0.95]
        fs = all_fs[all_fs < 0.95]

        # Find the max peak size
        within_sites, between_sites, total_sites = sfs_utils.calculate_polymorphism_rates_from_sfs_map(
            sfs_map[sample])
        between_line = between_sites * 1.0 / \
                       total_sites / ((fs > 0.2) * (fs < 0.5)).sum()
        pmax = np.max([pfs[(fs > 0.1) * (fs < 0.95)].max(), between_line])

        peak_idx, cutoff = HGT_utils._find_sfs_peaks_and_cutoff(fs, pfs, pmax)
        if cutoff:
            ax.axvspan(min(fs), cutoff, alpha=0.1, color='red', label='SNVs')
            # ax.legend()
        ax.set_xlim([0.50, 1.00])
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(['0.5', '0.75', '1.0'])
        ax.set_ylim([0, pmax * 3])
        ax.bar((all_fs - df / 2), all_pfs, width=df)
        # ax.plot(fs[peak_idx] - df / 2, pfs[peak_idx], 'rx', label='peaks detected')
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.text(0.05, 0.85, '$\\overline{D}=%d$' % (sample_coverage_map[sample]), horizontalalignment = 'left',
        verticalalignment = 'center', transform = ax.transAxes)
        if i!=2:
            ax.set_xticklabels([])
for ax in ex_axes[2]:
    ax.set_xlabel('Major allele freq')
ex_axes[0][0].set_ylabel('Quasi-\nphasable', labelpad=7)
ex_axes[1][0].set_ylabel('Simple\nco-colonization', labelpad=7)
ex_axes[2][0].set_ylabel('Complex\nco-colonization', labelpad=7)

fig.savefig(os.path.join(config.figure_directory, 'supp', 'supp_sample_stats.pdf'))