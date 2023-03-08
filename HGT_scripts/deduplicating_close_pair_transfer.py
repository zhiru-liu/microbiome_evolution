import os
import numpy as np
import sys
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.style import context
from matplotlib import cm
import pandas as pd
os.chdir('/Users/Device6/Documents/Research/bgoodlab/microbiome_evolution/')
from utils import parallel_utils
import config


def find_overlap(range1, range2):
    # Sort the ranges in ascending order
    range1.sort()
    range2.sort()

    # Check if the ranges overlap
    if range1[0] <= range2[0] <= range1[1]:
        return range2[0], min(range1[1], range2[1])
    elif range2[0] <= range1[0] <= range2[1]:
        return range1[0], min(range1[1], range2[1])
    else:
        return None



def process_one_species(species_name):
    dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])

    syn_core_mask = dh.general_mask
    core_mask = parallel_utils.get_general_site_mask(species_name, allowed_variants=['1D', '2D', '3D', '4D'])
    full_to_syn_mask = syn_core_mask[core_mask]  # core -> 4D core

    syn_core_length = syn_core_mask.sum()
    full_coord_to_syn_coord = np.empty(shape=full_to_syn_mask.shape)
    full_coord_to_syn_coord[full_to_syn_mask] = np.arange(syn_core_length).astype(int)
    full_coord_to_syn_coord[~full_to_syn_mask] = -1  # sites not in syn core

    sample_to_idx = {y:x for x, y in list(enumerate(dh.good_samples))}
    def compute_event_similarity(row1, row2):
        start1, end1 = row1['Core genome start loc'], row1['Core genome end loc']
        start2, end2 = row2['Core genome start loc'], row2['Core genome end loc']
        overlap = find_overlap([start1, end1], [start2, end2])
        if overlap is None:
            return None
        else:
            overlap_len = overlap[-1] - overlap[0] + 1
            overlap_frac1 = overlap_len / float(end1 - start1 + 1)
            overlap_frac2 = overlap_len / float(end2 - start2 + 1)
            if max(overlap_frac1, overlap_frac2) < 0.5:
                # overlap is too small
                return None
        alleles = np.zeros((syn_core_length, 2))

        sample1 = row1['Sample 1']
        sample2 = row1['Sample 2']
        pair1 = sample_to_idx[sample1], sample_to_idx[sample2]
        snp_vec1, covered1 = dh.get_snp_vector(pair1)
        alleles[covered1, 0] = snp_vec1

        sample1 = row2['Sample 1']
        sample2 = row2['Sample 2']
        pair2 = sample_to_idx[sample1], sample_to_idx[sample2]
        snp_vec2, covered2 = dh.get_snp_vector(pair2)
        alleles[covered2, 1] = snp_vec2

        syn_core_start = int(full_coord_to_syn_coord[overlap[0]])
        syn_core_end = int(full_coord_to_syn_coord[overlap[1]])
        mask = (covered1 & covered2)[syn_core_start:syn_core_end + 1]
        if np.sum(mask) == 0:
            return None
        overlap_snp = alleles[syn_core_start:syn_core_end + 1][mask]

        shared_snps = 2 * np.sum((overlap_snp[:, 0] == 1) & (overlap_snp[:, 1] == 1))
        total_snps = np.sum(overlap_snp)
        if total_snps == 0:
            return None
        similarity = shared_snps / float(total_snps)  # check this because there appears to be divided by zero error

        same = np.sum(overlap_snp[:, 0] == overlap_snp[:, 1])
        identity = same / float(overlap_snp.shape[0])
        mut = np.sum(overlap_snp[:, 0] != overlap_snp[:, 1])  # number of mutational differences
        return identity, mut, similarity

    # loading the computed transfers
    tab_path = os.path.join(config.figure_directory, 'supp_table', 'all_transfers', '{}.csv'.format(species_name))
    transfer_df = pd.read_csv(tab_path)
    # filter only within clade events used in main text
    within_events = transfer_df[transfer_df['Shown in Fig3?']]

    num_events = within_events.shape[0]
    tot = 0
    identity_mat = np.zeros((num_events, num_events))
    similarity_mat = np.zeros((num_events, num_events))
    muts_mat = np.ones((num_events, num_events)) * 1e8
    print("Start at at {}".format(datetime.now()))

    for i, j in itertools.combinations(xrange(num_events), 2):
        row1 = within_events.iloc[i, :]
        row2 = within_events.iloc[j, :]
        res = compute_event_similarity(row1, row2)
        if res is not None:
            identity, num_snps, similarity = res
        else:
            continue
        identity_mat[i, j] = identity
        identity_mat[j, i] = identity
        muts_mat[i, j] = num_snps
        muts_mat[j, i] = num_snps
        similarity_mat[i, j] = similarity
        similarity_mat[j, i] = similarity

        tot += 1
        if tot %10000 == 0:
            print("Finished {} at {}".format(tot, datetime.now()))

    print("Finished all at {}".format(datetime.now()))
    savepath = os.path.join(config.analysis_directory, 'misc', 'dedup', species_name)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    np.savetxt(os.path.join(savepath, 'identity.csv'), identity_mat)
    np.savetxt(os.path.join(savepath, 'muts.csv'), muts_mat)
    np.savetxt(os.path.join(savepath, 'similarity.csv'), similarity_mat)


for filename in os.listdir(os.path.join(config.figure_directory, 'supp_table', 'all_transfers')):
    species_name = filename.split('.')[0]
    if os.path.exists(os.path.join(config.analysis_directory, 'misc', 'dedup', species_name, 'identity.csv')):
        print("{} already processed".format(species_name))
        continue
    else:
        print("Processing {}".format(species_name))
    process_one_species(species_name)
