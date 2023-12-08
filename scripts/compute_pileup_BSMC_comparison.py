import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
from utils import pileup_utils
import config

data_dir = os.path.join(config.analysis_directory, 'fastsimbac_data', 'r_scan')

# for comparing with real data

# parameters of BSMC sim
t = 0.00825
genome_len = 280000

# analysis 1: compute a scan over threshold length to determine the length that matches B. vulgatus mean sharing fraction
def scan_length():
    surprise_index = np.arange(10, 35, 2)
    thresholds = surprise_index / t
    ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'r_scan', 'length_scan')
    medians = []
    cvs = []
    for i in range(400, 500):
        file_path = os.path.join(data_dir, '%d.txt'%i)
        cumu_runs = pileup_utils.process_BSMC_sim(file_path, os.path.join(ckpt_dir, '%d.txt'%i), genome_len, thresholds, return_res=True)
        medians.append(np.median(cumu_runs, axis=0))
        cv = np.std(cumu_runs, axis=0) / np.mean(cumu_runs, axis=0)
        cvs.append(cv)
    return medians, cvs

# analysis 2: compute pileup for the threshold for all simulation of the same r/mu=1
def pileup_for_reps():
    thresholds = np.array([3400]) #TODO: fill the threshold number. 3400: matching the median
    ckpt_dir = os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'r_scan', 'fastsimbac_rbymu_1')
    for i in range(400, 500):
        # This range corresponds to r/mu = 1
        file_path = os.path.join(data_dir, '%d.txt'%i)
        pileup_utils.process_BSMC_sim(file_path, os.path.join(ckpt_dir, '%d.txt'%i), genome_len, thresholds)

# analysis 3: compute pileup for a range of r to demonstrate it only decreases the mean, but won't change the CV
def compute_all_cv(debug=False):
    thresholds = np.array([3400])
    median_fracs = []
    cvs = []
    for i in range(700):
        file_path = os.path.join(data_dir, '%d.txt'%i)
        cumu_runs = pileup_utils.process_BSMC_sim(file_path, None, genome_len, thresholds, return_res=True)
        median_share_frac = np.median(cumu_runs, axis=0)
        cv = np.std(cumu_runs, axis=0) / np.mean(cumu_runs, axis=0)
        # save those numbers
        median_fracs.append(median_share_frac)
        cvs.append(cv)
        if debug:
            break

    return median_fracs, cvs


if __name__=="__main__":
    # median_fracs, cvs = compute_all_cv(debug=False)
    # np.savetxt(os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'r_scan_statistics', 'median_frac.csv'), median_fracs)
    # np.savetxt(os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'r_scan_statistics', 'cvs.csv'), cvs)
    median_scan, cvs_scan = scan_length()
    np.savetxt(os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'length_scan_median.csv'), median_scan)
    np.savetxt(os.path.join(config.analysis_directory, 'sharing_pileup', 'simulated', 'length_scan_cvs.csv'), cvs_scan)
