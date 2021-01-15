import numpy as np
import pandas as pd


def to_block(bool_array, block_size):
    # coarse-graining the bool array (snp array) into blocks
    num_blocks = len(bool_array) / int(block_size) + 1
    bins = np.arange(0, num_blocks * block_size + 1, block_size)
    counts, _ = np.histogram(np.nonzero(bool_array), bins)
    return counts

def process_close_pairs(dh, idxs, block_size):
    # dh: DataHoarder instance; see parallel_util.py
    # idxs: indices of closely-related pairs
    singles = []
    nonzeros = []
    snps = []
    num_blocks = []
    for pair in idxs:
        snp_vec, snp_mask = dh.get_snp_vector(pair)
        blocks = to_block(snp_vec, block_size)
        snps.append(np.sum(snp_vec))
        singles.append(sum(blocks == 1))
        nonzeros.append(sum(blocks >= 1))
        num_blocks.append(len(blocks))
    df = pd.DataFrame()
    df['single_snp_blocks'] = singles
    df['snp_blocks'] = nonzeros
    df['num_snps'] = snps
    df['num_total_blocks'] = num_blocks
    df['pair_idxs'] = idxs
    return df
