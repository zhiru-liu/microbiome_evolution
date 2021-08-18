import numpy as np
import os
import sys
sys.path.append("..")
from utils import pileup_utils, parallel_utils
import config


def compute_between_host(species_name, thresholds):
    if 'vulgatus' in species_name:
        # only B vulgatus need to pick out the dominant clade
        clade_cutoff = 0.03
    else:
        clade_cutoff = None

    ph = pileup_utils.Pileup_Helper(species_name, clade_cutoff=clade_cutoff)
    genome_len = np.sum(ph.dh.general_mask)
    cumu_runs = pileup_utils.compute_pileup_for_clusters(
        ph.cluster_dict, ph.get_event_start_end, genome_len, thresholds=thresholds)
    ckpt_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    np.savetxt(os.path.join(ckpt_path, 'cutoff_0.001.csv'), cumu_runs)
    np.savetxt(os.path.join(ckpt_path, 'thresholds.txt'), thresholds)


def compute_within_host(species_name, thresholds):
    dh = parallel_utils.DataHoarder(species_name, mode='within')
    cumu_runs = pileup_utils.compute_pileup_for_within_host(dh, thresholds)
    if cumu_runs is None:
        print("No within host samples for {}".format(species_name))
        return
    ckpt_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
    np.savetxt(os.path.join(ckpt_path, 'within_host_cutoff_0.001.csv'), cumu_runs)
    np.savetxt(os.path.join(ckpt_path, 'thresholds.txt'), thresholds)

species_name = 'Bacteroides_vulgatus_57955'
thresholds = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
compute_between_host(species_name, thresholds)
compute_within_host(species_name, thresholds)
