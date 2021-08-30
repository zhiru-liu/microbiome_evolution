import numpy as np
import os
import sys
import json
sys.path.append("..")
from utils import pileup_utils, parallel_utils, typical_pair_utils
import config


def compute_between_host(species_name, thresholds):
    print("Processing %s" % species_name)
    if 'vulgatus' in species_name:
        # only B vulgatus need to pick out the dominant clade
        clade_cutoff = 0.03
    else:
        clade_cutoff = None

    ckpt_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
    if os.path.exists(ckpt_path):
        print("%s already processed. Skipping" % species_name)
        return
    ph = pileup_utils.Pileup_Helper(species_name, clade_cutoff=clade_cutoff)
    genome_len = np.sum(ph.dh.general_mask)
    cumu_runs = pileup_utils.compute_pileup_for_clusters(
        ph.cluster_dict, ph.get_event_start_end, genome_len, thresholds=thresholds)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    np.savetxt(os.path.join(ckpt_path, 'between_host.csv'), cumu_runs)
    np.savetxt(os.path.join(ckpt_path, 'between_host_thresholds.txt'), thresholds)


def compute_B_vulgatus_between_clade(thresholds):
    species_name = 'Bacteroides_vulgatus_57955'
    ckpt_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', '%s_between'%species_name)
    ph = pileup_utils.Pileup_Helper(species_name, clade_cutoff=0.03)
    genome_len = np.sum(ph.dh.general_mask)
    cumu_runs = pileup_utils.compute_pileup_for_cluster_between_clades(
        ph.cluster_dict, ph.minor_cluster_dict, ph.get_event_start_end, genome_len, thresholds=thresholds)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    np.savetxt(os.path.join(ckpt_path, 'between_host.csv'), cumu_runs)
    np.savetxt(os.path.join(ckpt_path, 'between_host_thresholds.txt'), thresholds)


def compute_within_host(species_name, thresholds, b_vulgatus_between_clade=False):
    dh = parallel_utils.DataHoarder(species_name, mode='within')
    ckpt_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
    if 'vulgatus' in species_name and b_vulgatus_between_clade:
        # ugly but whatever
        cumu_runs = pileup_utils.compute_pileup_for_within_host(dh, thresholds, clade_cutoff=0.03, within_clade=False)
        ckpt_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', "%s_between" % species_name)
    elif 'vulgatus' in species_name:
        cumu_runs = pileup_utils.compute_pileup_for_within_host(dh, thresholds, clade_cutoff=0.03)
    else:
        cumu_runs = pileup_utils.compute_pileup_for_within_host(dh, thresholds)
    if cumu_runs is None:
        print("No within host samples for {}".format(species_name))
        return
    np.savetxt(os.path.join(ckpt_path, 'within_host.csv'), cumu_runs)
    np.savetxt(os.path.join(ckpt_path, 'within_host_thresholds.txt'), thresholds)


if __name__ == '__main__':
    contig_data = json.load(open(os.path.join(config.data_directory, 'contig_counts.json'), 'r'))
    # thresholds = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    # thresholds = [250, 350, 500, 750, 1000, 1250, 1500]
    # species_name = 'Bacteroides_vulgatus_57955'
    # species_name = 'Eubacterium_rectale_56927'

    # compute_within_host(species_name, thresholds, b_vulgatus_between_clade=True)
    # compute_B_vulgatus_between_clade(thresholds)

    for species_name in contig_data:
        if contig_data[species_name] > 50:
            continue
        theta = typical_pair_utils.compute_theta(species_name)
        thresholds = np.array([10, 20, 30, 40]) / theta
        compute_between_host(species_name, thresholds)
        compute_within_host(species_name, thresholds)
