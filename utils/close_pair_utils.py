import numpy as np
import pandas as pd


def find_close_pairs(cutoff, div_mat, good_idxs):
    """
    helper function for picking out pairs to process
    :param cutoff: cutoff divergence
    :param div_mat: matrix of pairwise divergences. Must be sorted the
    same order as DataHoarder order
    :param good_idxs: list of indices corresponding to distinct subjects
    Can be computed with dh.get_single_subject_idxs
    :return:
    """
    masked_div_mat = div_mat[good_idxs, :][:, good_idxs]
    idxs = np.nonzero((masked_div_mat < cutoff) & (masked_div_mat != 0))
    pairs = zip(list(idxs[0]), list(idxs[1]))
    pairs = [(good_idxs[pair[0]], good_idxs[pair[1]])
             for pair in pairs if pair[0] < pair[1]]
    return pairs


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


def process_close_pairs_first_pass(dh, idxs, block_size):
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


def _fit_and_count_transfers_iterative(sequence, model, block_size, iters=3):
    """
    Use a HMM to fit the sequence and eventually compute the number of runs, as well as
    and estimate for walll clock T. The program will iteratively determine the divergence
    in the clonal region. Only distinguishes clonal vs non-clonal.
    :param sequence: The sequence in sites, a snp vector
    :param model: The hidden markov model
    :param block_size: Size of the coarse-grained blocks
    :param iters: Number of iterations
    :return: triplet of start and end indices (inclusive) as well as wall clock estimate
    """
    init_means = model.init_means
    starts = []
    ends = []
    T_approx = 0
    blk_seq = to_block(sequence, block_size).reshape((-1, 1))
    for i in range(iters):
        model.fit(blk_seq)
        _, states = model.decode(blk_seq)
        starts, ends = find_segments(states)
        T_approx = np.sum(blk_seq[states == 0]) / (float(block_size) * np.sum(states == 0))
        model.init_means[0] = T_approx * block_size  # update the clonal divergence
    model.init_means = init_means
    return starts, ends, T_approx


def fit_and_count_transfers_all_chromosomes(snp_vec, chromosomes, model, block_size, iters=3):
    """
    Accumulate the results of above function for all contigs
    :param snp_vec: The full snp vector for a given pair of QP samples
    :param chromosomes: Array of same length as snp_vec, containing the chromosome of each site
    :param model: HMM model
    :param block_size: same as above
    :param iters: same as above
    :return: triplet of number of transfers, total transfer length, and wall clock estimate
    """
    num_transfers = []
    transfer_lens = []
    T_approxs = []
    for chromo in np.unique(chromosomes):
        # iterate over contigs; similar to run length dist calculation
        subvec = snp_vec[chromosomes==chromo]
        starts, ends, T_approx = _fit_and_count_transfers_iterative(
            subvec, model, block_size, iters=iters)
        num_transfers.append(len(starts))
        transfer_lens.append(np.sum(ends-starts+1))
        T_approxs.append(T_approx)
    num_transfer = np.sum(num_transfers)
    transfer_len = np.sum(transfer_lens)
    T_approx = np.mean(T_approxs)
    return num_transfer, transfer_len, T_approx


def prepare_x_y(df):
    # Multiple choice of x&y to plot
    clonal_fraction = 1 - df['transfer_len'] / df['num_total_blocks'].mean()
    exp_snps = df['num_clonal_snps'] / clonal_fraction
    x = exp_snps.to_numpy()
#    y = df['transfer_len'].to_numpy()
    y = df['num_transfers'].to_numpy()
    return x, y
