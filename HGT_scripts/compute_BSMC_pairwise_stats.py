import numpy as np
import os
import matplotlib.pyplot as plt
from utils import BSMC_utils
import config
from HGT_scripts.plot_clonal_frac_pairwise_div_joint import plot_one_species


def find_fit(x, y, x_threshold):
    xfit = x[x>=x_threshold]
    yfit = y[x>=x_threshold]
    params = np.polyfit(xfit, yfit, 2)
    return params


if __name__ == "__main__":
    all_cf_mats = []
    all_pd_mats = []
    data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'for_pileup')

    for i in range(40):
        sim_data = BSMC_utils.load_data(os.path.join(data_dir, 'b_vulgatus', '%d.txt')%i)

        genome_len = 280000
        cf_mat = BSMC_utils.get_pairwise_clonal_fraction_matrix(sim_data, genome_len)
        pd_mat = BSMC_utils.get_pairwise_distance_matrix(sim_data, genome_len)
        all_cf_mats.append(cf_mat)
        all_pd_mats.append(pd_mat)
        # cf_mat = all_cf_mats[i]
        # pd_mat = all_pd_mats[i]

        x = cf_mat[np.triu_indices(cf_mat.shape[0], 1)]
        y = pd_mat[np.triu_indices(pd_mat.shape[0], 1)]
        theta = np.mean(y[x < 0.01])
        f, axes = plot_one_species(x, y)
        f.savefig(os.path.join(config.analysis_directory, 'misc', 'BSMC_joint_plot', 'unscaled', '%d.pdf' % i))
        plt.close()

        f, axes = plot_one_species(x, y / theta, asexual_line=False)
        params = find_fit(x, y / theta, 0.2)
        trend_x = np.linspace(0, 1)
        trend_y = params[0] * trend_x ** 2 + params[1] * trend_x + params[2]
        axes[0].plot(trend_x, trend_y, color='tab:orange', linestyle='--')
        axes[0].plot(trend_x, 1 - trend_x, color='tab:green', linestyle='--')
        f.savefig(os.path.join(config.analysis_directory, 'misc', 'BSMC_joint_plot', 'scaled', '%d.pdf' % i))
        plt.close()