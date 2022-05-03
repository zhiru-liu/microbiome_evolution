import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import config
from utils import close_pair_utils, parallel_utils
from plotting_for_publication import default_fig_styles

bacteroides = [
    'Bacteroides_vulgatus_57955',
    'Bacteroides_ovatus_58035',
    'Bacteroides_uniformis_57318',
    'Bacteroides_thetaiotaomicron_56941',
    'Bacteroides_plebeius_61623',
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
for species_full_name in bacteroides:
    data_dir = os.path.join(config.analysis_directory, "closely_related")
    raw_data = pd.read_pickle(os.path.join(data_dir, 'third_pass', species_full_name + '.pickle'))
    if 'vulgatus' in species_full_name:
        x, y1, y2, _ = close_pair_utils.prepare_HMM_results_for_B_vulgatus(save_path=config.B_vulgatus_data_path,
            cf_cutoff=config.clonal_fraction_cutoff, mode='count', cache_intermediate=False)
        x_ = x[x > 0]
        y_ = y1[x > 0] + y2[x > 0]
        rates = y_ / x_
    else:
        x, y = close_pair_utils.prepare_x_y(raw_data, mode='count')
        x_ = x[x > 0]
        y_ = y[x > 0]
        rates = y_ / x_
    means.append(rates.mean())
    var.append(rates.var())
    all_rates.append(rates)
bacteroides_df = pd.DataFrame({"species": bacteroides, 'mean rates': means, 'variance rates': var})

sample_stats_df = parallel_utils.compute_good_sample_stats()
qp_fracs = sample_stats_df['num_qp_samples'] / sample_stats_df['num_high_coverage_samples'].astype(float)
qp_dict = dict(zip(sample_stats_df['species_name'], qp_fracs))
bacteroides_df['QP frac'] = bacteroides_df['species'].apply(qp_dict.get)

concat_rates = np.concatenate(all_rates)
concat_qp = np.concatenate([np.ones(len(all_rates[i])) * qp_dict[x] for i, x in enumerate(bacteroides)])

fig, axes = plt.subplots(1, 1, figsize=(3, 2))
# plt.subplots_adjust(wspace=0.5)
# axes[0].plot(concat_qp, concat_rates, '.')
# axes[0].set_ylim([0, 100000])
# # axes[0].set_yscale('log')
# axes[0].set_xlabel('Fraction of QP samples')
# axes[0].set_ylabel('Rates')
# print("Correlation using all points is:")
# print(spearmanr(concat_qp, concat_rates))

axes.plot(bacteroides_df['QP frac'], bacteroides_df['mean rates'], '.')
axes.set_ylim([None, 100000])
axes.set_yscale('log')
axes.set_xlabel('Fraction of QP samples')
axes.set_ylabel('Averaged transfer / divergence')
print("Correlation using only the mean is:")
print(spearmanr(bacteroides_df['QP frac'], bacteroides_df['mean rates']))

plt.savefig(os.path.join(config.figure_directory, 'supp', 'supp_qp_frac_rate_corr.pdf'), bbox_inches='tight')
