import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import BSMC_utils
import config
from scripts.plot_clonal_frac_pairwise_div_joint import plot_one_species


def find_fit(x, y, x_threshold):
    xfit = x[x>=x_threshold]
    yfit = y[x>=x_threshold]
    params = np.polyfit(xfit, yfit, 2)
    return params


if __name__ == "__main__":
    data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'for_pileup')
    df = pd.read_csv(os.path.join(data_dir, 'b_vulgatus', 'experiments.txt'), sep=' ')

    close_fractions_2 = []
    close_fractions_4 = []
    close_fractions_6 = []
    close_fractions_8 = []
    for i in range(100):
        pd_cached_dir = os.path.join(data_dir, 'b_vulgatus', 'pd', 'pd_%d.csv'%i)
        cf_cached_dir = os.path.join(data_dir, 'b_vulgatus', 'cf', 'cf_%d.csv'%i)
        if os.path.exists(pd_cached_dir):
            pd_mat = np.loadtxt(pd_cached_dir)
            cf_mat = np.loadtxt(cf_cached_dir)
        else:
            sim_data = BSMC_utils.load_data(os.path.join(data_dir, 'b_vulgatus', '%d.txt')%i)

            genome_len = 280000
            cf_mat = BSMC_utils.get_pairwise_clonal_fraction_matrix(sim_data, genome_len)
            pd_mat = BSMC_utils.get_pairwise_distance_matrix(sim_data, genome_len)
            np.savetxt(os.path.join(data_dir, 'b_vulgatus', 'pd', 'pd_%d.csv'%i), pd_mat)
            np.savetxt(os.path.join(data_dir, 'b_vulgatus', 'cf', 'cf_%d.csv'%i), cf_mat)
        # cf_mat = all_cf_mats[i]
        # pd_mat = all_pd_mats[i]

        x = cf_mat[np.triu_indices(cf_mat.shape[0], 1)]
        y = pd_mat[np.triu_indices(pd_mat.shape[0], 1)]
        theta = np.mean(y[x < 0.01])

        close_fractions_2.append(np.sum(x > 0.2))
        close_fractions_4.append(np.sum(x > 0.4))
        close_fractions_6.append(np.sum(x > 0.6))
        close_fractions_8.append(np.sum(x > 0.8))

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

    df['close counts 0.2'] = close_fractions_2
    df['close counts 0.4'] = close_fractions_4
    df['close counts 0.6'] = close_fractions_6
    df['close counts 0.8'] = close_fractions_8
    df.to_csv(os.path.join(config.analysis_directory, 'misc', 'BSMC_joint_plot', 'statistics.txt'))
