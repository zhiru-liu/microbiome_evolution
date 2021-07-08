'''
Lots of helper functions for parsing fastsimbac(BSMC) simulation results
'''
import numpy as np
import itertools


def load_data(path):
    f = open(path, 'r')
    data = []
    for line in f.read().splitlines()[:-5]:
        parts = line.split('\t')
        if len(parts) != 5:
            continue
        line_data = list(map(int, parts[-1]))
        line_data.append(float(parts[2]))
        data.append(line_data)
    f.close()
    return np.array(data)


def get_simulation_ids(metadata_df, rbymu, l):
    """
    Handy function to use the metadata df (loaded by reading metadata csv) to filter desired sim ids
    """
    return metadata_df[(metadata_df['rbymu']==rbymu) & (metadata_df['lambda']==l)]['sim_id']


def compare_two_samples(idx1, idx2, data, genome_len):
    locations = data[:, -1][np.nonzero(data[:, idx1] != data[:, idx2])]
    runs = locations[1:] - locations[:-1]
    runs = runs * genome_len
    return locations, runs.astype(int)


def get_snp_vector(idx1, idx2, data):
    return data[:, idx1] != data[:, idx2]


def get_full_snp_vector(idx1, idx2, data, genome_len):
    v = np.zeros(int(genome_len)).astype(bool)
    snps = data[:, idx1] != data[:, idx2]
    locs = (data[:, -1] * genome_len).astype(int)
    v[locs] = snps
    return v


def get_all_haplotypes(data, genome_len):
    haps = np.zeros((int(genome_len), data.shape[1]-1)).astype(bool)
    locs = (data[:, -1] * genome_len).astype(int)
    haps[locs, :] = data[:, :-1]
    return haps


def get_block_snp_vector(idx1, idx2, data, genome_len=1e6, block_len=1e3):
    # coarse graining snp vector into blocks
    bins = np.arange(0, genome_len + 1, block_len)
    locations = data[:, -1][np.nonzero(data[:, idx1] != data[:, idx2])] * genome_len
    snp_vec, _ = np.histogram(locations, bins)
    return snp_vec


def get_pairwise_distance_matrix(sim_data, genome_len):
    """
    Produce a pairwise divergence matrix
    :param sim_data: output of load_data, an array of shape (num snps, num samples+1)
    :param genome_len: genome length parameter for the simulation, needed for divergence calculation
    :return: an array of shape (num samples, num samples)
    """
    sample_size = sim_data.shape[1] - 1
    pd_mat = np.zeros((sample_size, sample_size))
    genome_len = float(genome_len)
    for i, j in itertools.combinations(range(sample_size), 2):
        snp_locs, run_lens = compare_two_samples(i, j, sim_data, genome_len)
        snp_count = len(snp_locs) + 1
        div = snp_count / genome_len
        pd_mat[i, j] = div
        pd_mat[j, i] = div
    return pd_mat