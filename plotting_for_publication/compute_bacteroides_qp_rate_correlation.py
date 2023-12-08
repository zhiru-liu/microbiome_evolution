import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr, pearsonr
import config
from utils import close_pair_utils, parallel_utils, diversity_utils
from plotting_for_publication import default_fig_styles

bacteroides = [
    'Bacteroides_vulgatus_57955',
    'Bacteroides_ovatus_58035',
    'Bacteroides_uniformis_57318',
    'Bacteroides_thetaiotaomicron_56941',
    'Bacteroides_stercoris_56735',
    'Bacteroides_coprocola_61586',
    'Bacteroides_eggerthii_54457',
    'Bacteroides_finegoldii_57739',
    'Bacteroides_caccae_53434',
    'Bacteroides_cellulosilyticus_58046',
    'Bacteroides_fragilis_54507',
    'Bacteroides_massiliensis_44749',
]

means = []
var = []
all_rates = []
# for species_full_name in bacteroides:
#     data_dir = os.path.join(config.analysis_directory, "closely_related")
#     raw_data = pd.read_pickle(os.path.join(data_dir, 'third_pass', species_full_name + '.pickle'))
#     if 'vulgatus' in species_full_name:
#         x, y1, y2, _ = close_pair_utils.prepare_HMM_results_for_B_vulgatus(save_path=config.B_vulgatus_data_path,
#             cf_cutoff=config.clonal_fraction_cutoff, mode='count', cache_intermediate=False)
#         x_ = x[x > 0]
#         y_ = y1[x > 0] + y2[x > 0]
#         rates = y_ / x_
#     else:
#         x, y = close_pair_utils.prepare_x_y(raw_data, mode='count')
#         x_ = x[x > 0]
#         y_ = y[x > 0]
#         rates = y_ / x_
#     means.append(rates.mean())
#     var.append(rates.var())
#     all_rates.append(rates)
# bacteroides_df = pd.DataFrame({"species": bacteroides, 'mean rates': means, 'variance rates': var})
TcTm_df = pd.read_csv(os.path.join(config.figure_directory, 'supp_table', 'TcTm_estimation.csv'))
TcTm_df.set_index('Species', inplace=True)
bacteroides_df = TcTm_df.loc[bacteroides, :]

sample_stats_df = parallel_utils.compute_good_sample_stats()
qp_fracs = sample_stats_df['num_qp_samples'] / sample_stats_df['num_high_coverage_samples'].astype(float)
qp_dict = dict(zip(sample_stats_df['species_name'], qp_fracs))
bacteroides_df['QP frac'] = bacteroides_df.index.to_series().apply(qp_dict.get)

# modified qp definition
strict_qp_num = []
for species_name in bacteroides:
    sample_names = parallel_utils.get_sample_names(species_name)
    QP_samples = set(diversity_utils.calculate_haploid_samples(species_name, upper_threshold=0.95, lower_threshold=0.05))
    highcoverage_samples = set(diversity_utils.calculate_highcoverage_samples(species_name))
    allowed_samples = QP_samples & highcoverage_samples
    strict_qp_mask = np.isin(sample_names, list(allowed_samples))
    strict_qp_num = np.sum(strict_qp_mask)
    bacteroides_df.loc[species_name, 'num_qp_samples_strict'] = strict_qp_num
    bacteroides_df.loc[species_name, 'QP frac strict'] = strict_qp_num / float(len(highcoverage_samples))

stringent_df = pd.read_csv(os.path.join(config.plotting_intermediate_directory, "stringent_poly_fraction.csv"), index_col='Species')
bacteroides_df['Polymorphic sample fraction (stringent)'] = stringent_df['Polymorphic sample fraction (stringent)']

# concat_rates = np.concatenate(all_rates)
# concat_qp = np.concatenate([np.ones(len(all_rates[i])) * qp_dict[x] for i, x in enumerate(bacteroides)])

fig, axes = plt.subplots(1, 3, figsize=(6, 2.))
# plt.subplots_adjust(wspace=0.5)
# axes[0].plot(concat_qp, concat_rates, '.')
# axes[0].set_ylim([0, 100000])
# # axes[0].set_yscale('log')
# axes[0].set_xlabel('Fraction of QP samples')
# axes[0].set_ylabel('Rates')
# print("Correlation using all points is:")
# print(spearmanr(concat_qp, concat_rates))
fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

axes[0].plot(bacteroides_df['QP frac'], bacteroides_df['Tc/Tm'].to_numpy(), '.', label=None)
axes[0].plot(bacteroides_df.loc['Bacteroides_cellulosilyticus_58046', 'QP frac'],
          bacteroides_df.loc['Bacteroides_cellulosilyticus_58046', 'Tc/Tm'], '.', color='tab:orange',
          label='Bacteroides cellulosilyticus')
axes[0].plot(bacteroides_df.loc['Bacteroides_caccae_53434','QP frac'],
          bacteroides_df.loc['Bacteroides_caccae_53434', 'Tc/Tm'], '.', color='tab:green',
          label='Bacteroides caccae')
axes[0].plot(bacteroides_df.loc['Bacteroides_fragilis_54507','QP frac'],
             bacteroides_df.loc['Bacteroides_fragilis_54507', 'Tc/Tm'], '.', color='tab:red',
             label='Bacteroides fragilis')
# axes.set_ylim([None, 110000])
axes[0].set_xlabel('Fraction of QP samples')
axes[0].set_ylabel('$T_{mrca}/T_{mosaic}$')
# axes[0].legend(loc='lower left')
print("Correlation using only the mean is:")
print(spearmanr(bacteroides_df['QP frac'], bacteroides_df['Tc/Tm']))

axes[1].plot(bacteroides_df['QP frac strict'], bacteroides_df['Tc/Tm'].to_numpy(), '.', label=None)
axes[1].plot(bacteroides_df.loc['Bacteroides_cellulosilyticus_58046', 'QP frac strict'],
             bacteroides_df.loc['Bacteroides_cellulosilyticus_58046', 'Tc/Tm'], '.', color='tab:orange',
             label='Bacteroides cellulosilyticus')
axes[1].plot(bacteroides_df.loc['Bacteroides_caccae_53434', 'QP frac strict'],
             bacteroides_df.loc['Bacteroides_caccae_53434', 'Tc/Tm'], '.', color='tab:green',
             label='Bacteroides caccae')
axes[1].plot(bacteroides_df.loc['Bacteroides_fragilis_54507','QP frac strict'],
             bacteroides_df.loc['Bacteroides_fragilis_54507', 'Tc/Tm'], '.', color='tab:red',
             label='Bacteroides fragilis')
# axes.set_ylim([None, 110000])
axes[1].set_xlabel('Fraction of QP samples (strict)')
axes[1].set_ylabel('$T_{mrca}/T_{mosaic}$')
# axes[1].legend(loc='lower left')

axes[2].plot(bacteroides_df['Polymorphic sample fraction (stringent)'], bacteroides_df['Tc/Tm'].to_numpy(), '.', label=None)
axes[2].plot(bacteroides_df.loc['Bacteroides_cellulosilyticus_58046', 'Polymorphic sample fraction (stringent)'],
             bacteroides_df.loc['Bacteroides_cellulosilyticus_58046', 'Tc/Tm'], '.', color='tab:orange',
             label='Bacteroides cellulosilyticus')
axes[2].plot(bacteroides_df.loc['Bacteroides_caccae_53434', 'Polymorphic sample fraction (stringent)'],
             bacteroides_df.loc['Bacteroides_caccae_53434', 'Tc/Tm'], '.', color='tab:green',
             label='Bacteroides caccae')
axes[2].plot(bacteroides_df.loc['Bacteroides_fragilis_54507','Polymorphic sample fraction (stringent)'],
             bacteroides_df.loc['Bacteroides_fragilis_54507', 'Tc/Tm'], '.', color='tab:red',
             label='Bacteroides fragilis')
# axes.set_ylim([None, 110000])
axes[2].set_xlabel('Fraction of \n\"mono-colonized\" samples')
axes[2].set_ylabel('$T_{mrca}/T_{mosaic}$')
axes[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

bacteroides_df.to_csv(os.path.join(config.analysis_directory, 'misc', 'bacteroides_QP_rate_statistics.csv'))
plt.tight_layout()
plt.savefig(os.path.join(config.figure_directory, 'supp', 'S40_supp_qp_frac_rate_corr.pdf'))
