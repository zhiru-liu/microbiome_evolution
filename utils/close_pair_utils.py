import numpy as np
import pandas as pd


def to_block(bool_array, block_size):
    """
    Converting a boolean array into blocks of True counts. The last
    block could be shorter than block_size
    :param bool_array:
    :param block_size:
    :return: An array of counts of Trues in blocks
    """
    # coarse-graining the bool array (snp array) into blocks
    num_blocks = len(bool_array) / int(block_size) + 1
    bins = np.arange(0, num_blocks * block_size + 1, block_size)
    counts, _ = np.histogram(np.nonzero(bool_array), bins)
    return counts


def process_close_pairs(dh, idxs, block_size):
    """
    :param dh: DataHoarder instance; see parallel_util.py
    :param idxs: Pairs to be processed. Should be sufficiently close
    :param block_size: Block length to use when coarse-graining the genome into blocks
    :return: A pandas dataframe holding various useful statistics
    """
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


def find_segments(states, target_state=None):
    """
    Find the continuous segments of a target state. By default, all
    nonzero segments will be found.
    :param states: array of non-negative integers, output of HMM
    :param target_state: the state of the segments, if None, find nonzero segments
    :return: start and end indices of segments, *inclusive*
    """
    if target_state is not None:
        states = states == target_state
        states = states.astype(int)
    else:
        for n in np.unique(states):
            if n not in [0, 1]:
                import warnings
                warnings.warn(
                    "Treating all nonzero states as recombined regions", RuntimeWarning)
                states = states.copy()
                states[states != 0] = 1
                break
    # padding to take care of end points
    padded = np.empty(len(states) + 2)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = states
    diff = padded[1:] - padded[:-1]
    ups = np.nonzero(diff == 1)[0]
    downs = np.nonzero(diff == -1)[0]
    return ups, downs - 1

def prepare_x_y(df):
    # Multiple choice of x&y to plot
    clonal_fraction = 1 - df['transfer_len'] / df['num_total_blocks'].mean()
    exp_snps = df['num_clonal_snps'] / clonal_fraction
    x = exp_snps.to_numpy()
#    y = df['transfer_len'].to_numpy()
    y = df['num_transfers'].to_numpy()
    return x, y
