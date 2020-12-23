import numpy as np
import os
import sys
sys.path.append("..")
from utils import core_gene_utils, diversity_utils, HGT_utils, parallel_utils
import config


def compute_one_species(species_name, debug=True):
    save_path = os.path.join(
            config.analysis_directory, "pairwise_divergence", "between_hosts", "%s.csv" % species_name)
    if os.path.exists(save_path):
        print('{} has already been processed'.format(species_name))
        return
    dh = parallel_utils.DataHoarder(species_name)
    num_samples = dh.snp_arr.shape[1] if not debug else 10
    div_mat = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            snp_vec, _ = parallel_utils.get_two_QP_sample_snp_vector(
                    dh.snp_arr, dh.covered_arr, (i, j))
            div = np.sum(snp_vec) / float(len(snp_vec))
            div_mat[i, j] = div
            div_mat[j, i] = div
    np.savetxt(save_path, div_mat, delimiter=',')


def main():
    base_dir = 'zarr_snps'
    for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
        if species_name.startswith('.'):
            continue
        print("processing %s" % species_name)
        compute_one_species(species_name, debug=False)


if __name__ == "__main__":
    main()
