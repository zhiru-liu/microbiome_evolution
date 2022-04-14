import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
from utils import close_pair_utils, BSMC_utils, pileup_utils
import config


if __name__ == "__main__":
    # data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'for_pileup', 'b_vulgatus')
    data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'r_scan', 'fastsimbac_rbymu_1')
    # ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'b_vulgatus')
    ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'r_scan', 'fastsimbac_rbymu_1')
    DEBUG=False
    t = 0.00825
    genome_len = 280000
    surprise_index = np.arange(20, 35, 2)
    thresholds = surprise_index / t
    # thresholds = [1500, 2000, 2500, 3000, 3500, 4000, 4500]
    np.savetxt(os.path.join(ckpt_dir, 'thresholds.txt'), thresholds)


    for i in range(400, 410):
        file_path = os.path.join(data_dir, '%d.txt'%i)
        cv = pileup_utils.process_BSMC_sim(file_path, os.path.join(ckpt_dir, '%d.txt'%i), genome_len, thresholds)
        # print(cv)


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
