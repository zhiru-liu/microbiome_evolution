import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
from utils import close_pair_utils, BSMC_utils, pileup_utils
import config

# data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'for_pileup', 'b_vulgatus')
data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'hotspots_rbymu_2')
# ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'b_vulgatus')
ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'hotspots_rbymu2')
DEBUG=False
t = 0.00825
genome_len = 280000
surprise_index = np.array([15, 20, 25])
thresholds = surprise_index / t
# thresholds = [1500, 2000, 2500, 3000, 3500, 4000, 4500]
np.savetxt(os.path.join(ckpt_dir, 'thresholds.txt'), thresholds)


def process_sim(filename, save_path, genome_len, close_pair_cutoff=0.95):
    sim_data = BSMC_utils.load_data(filename)
    cf_mat = BSMC_utils.get_pairwise_clonal_fraction_matrix(sim_data, genome_len)
    cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(1 - cf_mat, threshold=1 - close_pair_cutoff)
    # pd_mat = BSMC_utils.get_pairwise_distance_matrix(sim_data, genome_len)
    # cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(pd_mat, threshold=0.001)
    print("Sim has {} close clusters".format(len(cluster_dict)))
    f = lambda x, y, z: pileup_utils.get_event_start_end_BSMC(sim_data, genome_len, x, y, z)
    cumu_runs = pileup_utils.compute_pileup_for_clusters(cluster_dict, f, genome_len, thresholds)
    # save cumu_runs
    np.savetxt(save_path, cumu_runs)
    return np.std(cumu_runs, axis=0) / np.mean(cumu_runs, axis=0)


for i in range(1, 11):
    file_path = os.path.join(data_dir, 'rep%d.txt'%i)
    cv = process_sim(file_path, os.path.join(ckpt_dir, 'rep%d.txt'%i), genome_len)
    print(cv)


# Codes for processing multiple files

# find the simulation id from the metadata df
# df = pd.read_csv(os.path.join(data_dir, 'experiments.txt'), delimiter=' ')

# rbymus = [0.1, 0.5, 1, 1.5, 2]
# lambdas = [500, 1000, 2000, 5000, 10000]
# cvs = []
# for i in range(len(rbymus)):
#     rbymu = rbymus[i]
#     for j in range(len(lambdas)):
#         l = lambdas[j]
#         for sim_id in np.array(BSMC_utils.get_simulation_ids(df, rbymu, l)):
#             filename = '%d.txt' % sim_id
#             path = os.path.join(data_dir, filename)
#             save_path = os.path.join(ckpt_dir, filename)
#             cv = process_sim(path, save_path, genome_len)
#             cvs.append(cv)
#
#             if DEBUG:
#                 break
#         if DEBUG:
#             break
#     if DEBUG:
#         break
#
# np.savetxt(os.path.join(ckpt_dir, 'cv.csv'), cvs)
