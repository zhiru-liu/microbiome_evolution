import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
from utils import close_pair_utils, BSMC_utils, pileup_utils
import config

data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'for_pileup', 'b_vulgatus')
ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'b_vulgatus')
t = 0.00825
genome_len = 280000
# surprise_index = np.array([15, 20, 25, 30, 35])
# thresholds = surprise_index / t
thresholds = [1500, 2000, 2500, 3000, 3500, 4000, 4500]

# find the simulation id from the metadata df
df = pd.read_csv(os.path.join(data_dir, 'experiments.txt'), delimiter=' ')


rbymus = [0.1, 0.5, 1, 1.5, 2]
lambdas = [5000, 10000]
cvs = []
for i in range(len(rbymus)):
    rbymu = rbymus[i]
    for j in range(len(lambdas)):
        l = lambdas[j]
        for sim_id in np.array(BSMC_utils.get_simulation_ids(df, rbymu, l)):
            filename = '%d.txt' % sim_id
            sim_data = BSMC_utils.load_data(os.path.join(data_dir, filename))
            pd_mat = BSMC_utils.get_pairwise_distance_matrix(sim_data, genome_len)
            cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(pd_mat, threshold=0.001)
            print("Sim #{} has {} close clusters".format(sim_id, len(cluster_dict)))
            f = lambda x, y, z: pileup_utils.get_event_start_end_BSMC(sim_data, genome_len, x, y, z)
            cumu_runs = pileup_utils.compute_pileup_for_clusters(cluster_dict, f, genome_len, thresholds)
            # save cumu_runs
            np.savetxt(os.path.join(ckpt_dir, filename), cumu_runs)
            cvs.append(np.std(cumu_runs, axis=0) / np.mean(cumu_runs, axis=0))

np.savetxt(os.path.join(ckpt_dir, 'cv.csv'), cvs)
