import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
import config
import json
from utils import snp_data_utils, close_pair_utils, pileup_utils


def load_highlight_samples(start, end):
    species_name = 'Bacteroides_vulgatus_57955'
    save_path = os.path.join(config.plotting_intermediate_directory, 'haplotype_highlight_samples.csv')
    if os.path.exists(save_path):
        highlight_pairs = np.loadtxt(save_path)
    else:
        ph = pileup_utils.Pileup_Helper(species_name, clade_cutoff=0.03)
        base_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'cached',
                                 'B_vulgatus_between_host_within_clade')
        good_pairs = []
        for cache_file in os.listdir(base_path):
            id1, id2 = cache_file.split('.')[0].split('_')
            sample1, sample2 = ph.cluster_id_to_sample_id(int(id1)), ph.cluster_id_to_sample_id(int(id2))
            fullpath = os.path.join(base_path, cache_file)
            cached_runs = json.load(open(fullpath, 'r'))
            for s, e in cached_runs[0][0]:
                if (s < start) and (e > end):
                    good_pairs.append((sample1, sample2))
        highlight_pairs = np.unique(np.concatenate(good_pairs))
        np.savetxt(save_path, highlight_pairs)
    return highlight_pairs


def prepare_and_plot_haplotypes(highlight_pairs, allowed_variants=['1D', '2D', '3D', '4D']):
    species_name = 'Bacteroides_vulgatus_57955'
    within_dh = snp_data_utils.DataHoarder(species_name, mode='within', allowed_variants=allowed_variants)
    between_dh = snp_data_utils.DataHoarder(species_name, allowed_variants=allowed_variants)

    genes_to_plot = ['435590.9.peg.1739', '435590.9.peg.1740', '435590.9.peg.1741']  # only the efflux pumps
    region_mask = np.isin(within_dh.gene_names[within_dh.general_mask], genes_to_plot)
    locs = np.where(region_mask)[0]
    start, end = locs[0], locs[-1]
    print(start, end)

    within_snps = within_dh.naive_haplotype[start:end, 92].reshape((-1, 1))
    between_snps, between_covered = between_dh.get_local_haplotype(start, end, only_snps=False, if_polarize=False)

    polarized_snps = np.zeros(between_snps.shape)
    for i in range(between_snps.shape[0]):
        for j in range(between_snps.shape[1]):
            if not between_covered[i, j]:
                polarized_snps[i, j] = 0
            elif between_snps[i, j]!=within_snps[i]:
                polarized_snps[i, j] = 1
            else:
                polarized_snps[i, j] = 0

    has_snps = polarized_snps.sum(axis=1) > 0
    only_snps = polarized_snps[has_snps, :]
    if_covered = between_covered[has_snps, :]
    num_samples = polarized_snps.shape[1]
    pw_mat = np.zeros((num_samples, num_samples))
    for i, j in itertools.combinations(range(num_samples), 2):
        covered = np.logical_and(if_covered[:, i], if_covered[:, j])
        snp_vec = only_snps[:, i] != only_snps[:, j]
        pw_mat[i, j] = np.sum(snp_vec[covered])
        pw_mat[j, i] = np.sum(snp_vec[covered])

    clusters = close_pair_utils.get_clusters_from_pairwise_matrix(pw_mat, threshold=1)
    cluster_means = []
    for key in clusters:
        cluster_means.append(only_snps[:, clusters[key]].mean())
    cluster_order = np.array(clusters.keys())[np.argsort(cluster_means)]
    sample_order = np.concatenate([clusters[c] for c in cluster_order])

    good_samples = np.load(os.path.join(config.plotting_intermediate_directory, 'Bv_within_clade_samples.npy'))
    sample_mask = np.isin(sample_order, good_samples)
    filtered_sample_order = sample_order[sample_mask]

    reordered_snps = only_snps[:, filtered_sample_order]
    reordered_covered = if_covered[:, filtered_sample_order]

    final_has_snps = reordered_snps.sum(axis=1)>0  # after filtering samples finally

    # finally building the RGB matrix
    haplotype = np.zeros((only_snps.shape[0], only_snps.shape[1], 3))
    for i in range(only_snps.shape[0]):
        for j in range(only_snps.shape[1]):
            if not if_covered[i, j]:
                haplotype[i, j, :] = 1
            elif only_snps[i, j] == 1:
                haplotype[i, j, 0] = 1
            else:
                haplotype[i, j, 2] = 1

    for i in highlight_pairs:
        i = int(i)
        mask = haplotype[:, i, 1] != 1
        haplotype[mask, i, 1] = 0.3

    final_haplotype = haplotype[final_has_snps, :][:, filtered_sample_order]
    return final_haplotype


if __name__ == "__main__":
    highlight_samples = load_highlight_samples(0, 1)  # start end should use 4D region coordinates
    final_haplotype = prepare_and_plot_haplotypes(highlight_samples, allowed_variants=['4D'])
    final_haplotype = np.swapaxes(final_haplotype, 0, 1)
    plt.figure(dpi=600)
    plt.imshow(final_haplotype)
    plt.ylabel('Samples')
    plt.xlabel('SNVs')
    # plt.savefig(os.path.join(config.figure_directory, 'supp', 'supp_within_sweep_haplotype.pdf'),
    #             bbox_inches='tight')
    plt.close()

    final_haplotype = prepare_and_plot_haplotypes(highlight_samples)
    final_haplotype = np.swapaxes(final_haplotype, 0, 1)
    np.save(os.path.join(config.figure_data_directory, 'figS32', 'Bv_within_sweep_haplotype_all_sites'), final_haplotype, delimiter=',')
    plt.figure(dpi=600)
    plt.imshow(final_haplotype)
    plt.ylabel('Samples')
    plt.xlabel('SNVs')
    plt.savefig(os.path.join(config.figure_directory, 'supp', 'S32_supp_within_sweep_haplotype_all_sites.pdf'),
                bbox_inches='tight')
    plt.close()
