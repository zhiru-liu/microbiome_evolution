import numpy as np
import os
import sys
import pandas as pd
sys.path.append("..")
from utils import close_pair_utils, parallel_utils
import config


def compute_one_species(species_name, debug=False):
    save_path = os.path.join(
            config.analysis_directory, "pairwise_divergence", "between_hosts", "%s.csv" % species_name)
    if os.path.exists(save_path):
        print('{} has already been processed'.format(species_name))
        return
    dh = parallel_utils.DataHoarder(species_name)
    num_samples = dh.snp_arr.shape[1] if not debug else 10
    div_mat = np.zeros((num_samples, num_samples))
    clonal_frac_mat = np.ones((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            snp_vec, _ = dh.get_snp_vector((i, j))
            div = np.sum(snp_vec) / float(len(snp_vec))

            snp_blocks = close_pair_utils.to_block(snp_vec, config.first_pass_block_size)
            nonzeros = np.sum(snp_blocks > 0)
            clonal_fraction = 1 - float(nonzeros) / len(snp_blocks)

            div_mat[i, j] = div
            div_mat[j, i] = div
            clonal_frac_mat[i, j] = clonal_fraction
            clonal_frac_mat[j, i] = clonal_fraction
    np.savetxt(save_path, div_mat, delimiter=',')

    clonal_frac_path = os.path.join(
        config.analysis_directory, "pairwise_clonal_fraction", "between_hosts", "%s.csv" % species_name)
    np.savetxt(clonal_frac_path, clonal_frac_mat, delimiter=',')
    del dh


def compute_one_species_within_host(species_name):
    save_path = os.path.join(
            config.analysis_directory, "pairwise_divergence", "within_hosts", "%s.csv" % species_name)
    if os.path.exists(save_path):
        print('{} has already been processed'.format(species_name))
        return
    dh = parallel_utils.DataHoarder(species_name, mode="within")
    num_samples = dh.snp_arr.shape[1]
    div_arr = np.zeros(num_samples)
    clonal_frac_arr = np.zeros(num_samples)
    for i in range(num_samples):
        snp_vec, _ = dh.get_snp_vector(i)
        div = np.sum(snp_vec) / float(len(snp_vec))

        snp_blocks = close_pair_utils.to_block(snp_vec, config.first_pass_block_size)
        nonzeros = np.sum(snp_blocks > 0)
        clonal_fraction = 1 - float(nonzeros) / len(snp_blocks)

        div_arr[i] = div
        clonal_frac_arr[i] = clonal_fraction
    np.savetxt(save_path, div_arr, delimiter=',')

    clonal_frac_path = os.path.join(
        config.analysis_directory, "pairwise_clonal_fraction", "within_hosts", "%s.csv" % species_name)
    np.savetxt(clonal_frac_path, clonal_frac_arr, delimiter=',')
    del dh


def compute_one_species_isolate(species_name, debug=False):
    save_path = os.path.join(
        config.analysis_directory, "pairwise_divergence", "isolates", "%s.csv" % species_name)
    if os.path.exists(save_path):
        print('{} has already been processed'.format(species_name))
        return
    dh = parallel_utils.DataHoarder(species_name, mode='isolates')
    num_samples = dh.snp_arr.shape[1] if not debug else 10
    div_mat = np.zeros((num_samples, num_samples))
    clonal_frac_mat = np.ones((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            snp_vec, _ = dh.get_snp_vector((i, j))
            div = np.sum(snp_vec) / float(len(snp_vec))

            snp_blocks = close_pair_utils.to_block(snp_vec, config.first_pass_block_size)
            nonzeros = np.sum(snp_blocks > 0)
            clonal_fraction = 1 - float(nonzeros) / len(snp_blocks)

            div_mat[i, j] = div
            div_mat[j, i] = div
            clonal_frac_mat[i, j] = clonal_fraction
            clonal_frac_mat[j, i] = clonal_fraction
    np.savetxt(save_path, div_mat, delimiter=',')

    clonal_frac_path = os.path.join(
        config.analysis_directory, "pairwise_clonal_fraction", "isolates", "%s.csv" % species_name)
    np.savetxt(clonal_frac_path, clonal_frac_mat, delimiter=',')
    del dh


def main():
    # base_dir = 'zarr_snps'
    # for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
    #     if species_name.startswith('.'):
    #         continue
    #     print("processing %s" % species_name)
    #     compute_one_species(species_name)
    isolate_metadata = pd.read_csv(os.path.join(config.isolate_directory, 'isolate_info.csv'), index_col='MGnify_accession')
    for species_name, row in isolate_metadata.iterrows():
        compute_one_species_isolate(species_name)


if __name__ == "__main__":
    main()
