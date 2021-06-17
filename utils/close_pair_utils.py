import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hierarchy


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
    num_blocks = (len(bool_array) + block_size - 1) // block_size
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


def find_segments(states, target_state=None, target_range=None):
    """
    Find the continuous segments of a target state. By default, all
    nonzero segments will be found.
    :param states: array of non-negative integers, output of HMM
    :param target_state: the state of the segments, if None and no target_range,
    find nonzero segments
    :param target_range: the range of accepted states, left inclusive
    :return: start and end indices of segments, *inclusive*
    """
    if target_state is not None:
        states = states == target_state
        states = states.astype(int)
    elif target_range is not None:
        if len(target_range) != 2:
            raise ValueError("Please supply the desired range of states as [min, max]")
        states = (states >= target_range[0]) & (states < target_range[1])
        states = states.astype(int)
    else:
        # find all non zero segments
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


def _decode_and_count_transfers(sequence, model, block_size, need_fit=True, clade_cutoff_bin=None,
                                index_offset=0):
    """
    Use a HMM to decode the sequence and eventually compute the number of runs, as well as
    and estimate for wall clock T. Can distinguish different types of transfers using clade_cutoff_bin.
    Does not fit every sequence!
    :param sequence: The sequence in blocks
    :param model: The hidden markov model
    :param block_size: Size of the coarse-grained blocks
    :param clade_cutoff_bin: For determining whether transfer is within clade or between clade
    based on the inferred bin in the empirical distribution. Within clade range from state 1 to
    state clade_cutoff_bin, inclusive.
    :return: triplet of start and end indices (inclusive) as well as wall clock estimate
    """
    if need_fit:
        model.fit(sequence)
    _, states = model.decode(sequence)
    T_approx = np.sum(sequence[states == 0]) / (float(block_size) * np.sum(states == 0))

    if clade_cutoff_bin is not None:
        # compute segments lengths for desire states
        starts = []
        ends = []
        for limits in [[1, clade_cutoff_bin], [clade_cutoff_bin, np.inf]]:
            tmp_starts, tmp_ends = find_segments(states, target_state=None, target_range=limits)
            if len(tmp_starts) != 0:
                # for taking care of contigs
                tmp_starts += index_offset
                tmp_ends += index_offset
            starts.append(tmp_starts)
            ends.append(tmp_ends)
    else:
        tmp_starts, tmp_ends = find_segments(states)
        if len(tmp_starts) != 0:
            tmp_starts += index_offset
            tmp_ends += index_offset
        # put in [] for compatibility with the above case
        starts = [tmp_starts]
        ends = [tmp_ends]
    return starts, ends, T_approx


def _fit_and_count_transfers_iterative(sequence, model, block_size, desired_states=[], iters=3):
    """
    Use a HMM to fit the sequence and eventually compute the number of runs, as well as
    and estimate for walll clock T. The program will iteratively determine the divergence
    in the clonal region. Only distinguishes clonal vs non-clonal.
    :param sequence: The sequence in blocks
    :param model: The hidden markov model
    :param block_size: Size of the coarse-grained blocks
    :param desired_states: a list of states that should be detected. If not provided, return
    all runs of nonzero states
    :param iters: Number of iterations
    :return: triplet of start and end indices (inclusive) as well as wall clock estimate
    """
    init_means = model.init_means
    starts = []
    ends = []
    T_approx = 0
    for i in range(iters):
        model.fit(sequence)
        _, states = model.decode(sequence)
        starts, ends = find_segments(states)
        T_approx = np.sum(sequence[states == 0]) / (float(block_size) * np.sum(states == 0))
        model.init_means[0] = T_approx * block_size  # update the clonal divergence
    model.init_means = init_means

    if len(desired_states) > 0:
        # compute segments lengths for desire states
        starts = []
        ends = []
        for state in desired_states:
            tmp_starts, tmp_ends = find_segments(states, state)
            starts.append(tmp_starts)
            ends.append(tmp_ends)

    return starts, ends, T_approx


def fit_and_count_transfers_all_chromosomes(snp_vec, chromosomes, model, block_size):
    """
    Accumulate the results of above function for all contigs
    :param snp_vec: The full snp vector for a given pair of QP samples
    :param chromosomes: Array of same length as snp_vec, containing the chromosome of each site
    :param model: HMM model
    :param block_size: size of the block
    :return: triplet of starts and ends of transfers, and wall clock estimate
    """
    all_starts = []
    all_ends = []
    T_approxs = []
    index_offset = 0
    for chromo in np.unique(chromosomes):
        # iterate over contigs; similar to run length dist calculation
        subvec = snp_vec[chromosomes==chromo]
        blk_seq = to_block(subvec, block_size).reshape((-1, 1))
        # to reduce effect of correlated mutation over short distances
        blk_seq = (blk_seq > 0).astype(float)
        if np.sum(blk_seq) == 0:
            # some time will have an identical contig
            # have to skip otherwise will mess up hmm
            starts = [np.array([])]
            ends = [np.array([])]
            T_approx = 0
        else:
            starts, ends, T_approx = _decode_and_count_transfers(
                blk_seq, model, block_size, index_offset=index_offset)
        all_starts.append(starts)
        all_ends.append(ends)
        T_approxs.append(T_approx)
        model.reinit_emission_and_transfer_rates()
        index_offset += len(blk_seq)

    # group transfers of the same type together over contigs
    num_types = len(all_starts[0])
    starts = []
    ends = []
    for i in xrange(num_types):
        starts.append(np.concatenate([s[i] for s in all_starts]))
        ends.append(np.concatenate([s[i] for s in all_ends]))
    T_approx = np.mean(T_approxs)
    return starts, ends, T_approx


def merge_and_filter_transfers_one_pair(starts, ends, merge_threshold=100, filter_threshold=10):
    # clean up raw data
    df_transfers = pd.DataFrame()
    df_transfers['starts'] = np.concatenate(starts)
    df_transfers['ends'] = np.concatenate(ends)
    df_transfers['types'] = np.concatenate(
        [np.repeat(i, len(x)) for i, x in enumerate(starts)])
    df_transfers = df_transfers.sort_values('starts')

    # merging
    new_starts = []
    new_ends = []
    new_types = []
    curr_start = None
    curr_end = None
    curr_type = None
    for _, row in df_transfers.iterrows():
        if curr_start is None:
            curr_start = row['starts']
            curr_end = row['ends']
            curr_type = row['types']
            continue
        if (row['starts'] - curr_end) < merge_threshold:
            curr_len = curr_end - curr_start + 1
            next_len = row['ends'] - row['starts'] + 1
            curr_end = row['ends']
            curr_type = curr_type if curr_len >= next_len else row['types']
        else:
            new_starts.append(curr_start)
            new_ends.append(curr_end)
            new_types.append(curr_type)
            curr_start = row['starts']
            curr_end = row['ends']
            curr_type = row['types']
    new_starts.append(curr_start)
    new_ends.append(curr_end)
    new_types.append(curr_type)
    new_df = pd.DataFrame(zip(new_starts, new_ends, new_types), columns=['starts', 'ends', 'types'])
    new_df['lengths'] = new_df['ends'] - new_df['starts'] + 1

    if filter_threshold:
        new_df = new_df[new_df['lengths'] >= filter_threshold]
    return new_df


def merge_and_filter_transfers(data, separate_clade=False):
    """
    process the output of stage 2 (HMM detection) by merging and filtering transfers
    to reduce bioinformatic errors
    :param data: dict with intermediate stats
    :param separate_clade: whether report within/between clade transfer separately
    :return: array of number of transfer per pair; dataframe of every single transfer's info
    """
    all_dfs = []
    num_pairs = len(data['starts'])
    counts = []
    if separate_clade:
        between_counts = []
        within_counts = []
    for i in range(num_pairs):
        merged_df = merge_and_filter_transfers_one_pair(data['starts'][i], data['ends'][i],
                                                        merge_threshold=100,
                                                        filter_threshold=10)
        merged_df['pairs'] = [data['pairs'][i] for x in range(merged_df.shape[0])]  # record pair information
        counts.append(len(merged_df))
        if separate_clade:
            between_counts.append(np.sum(merged_df['types'] == 1))
            within_counts.append(len(merged_df) - between_counts[-1])
        all_dfs.append(merged_df)
    counts = np.array(counts)
    full_df = pd.concat(all_dfs)
    if separate_clade:
        return within_counts, between_counts, full_df
    else:
        return counts, full_df


def prepare_x_y(df):
    # Multiple choice of x&y to plot
    clonal_fraction = 1 - df['transfer_len'] / df['num_total_blocks'].mean()
    exp_snps = df['num_clonal_snps'] / clonal_fraction
    x = exp_snps.to_numpy()
#    y = df['transfer_len'].to_numpy()
    y = df['num_transfers'].to_numpy()
    return x, y


def _fclusters_to_dict(clusters):
    """
    organize scipy.hierarchy.fcluster output into a dictionary
    :param clusters: a array of length num_samples, values are the cluster id
    :return: dictionary of {cluster_id: [sample_idxs]}
    """
    d = dict()
    for i, c in enumerate(clusters):
        if c in d:
            d[c].append(i)
        else:
            d[c] = [i]
    return d


def get_clusters_from_pairwise_matrix(pd_mat, threshold=1e-3):
    """
    Clustering samples to avoid overcounting close pairs
    Method is linkage average clustering
    :param pd_mat: pairwise divergence matrix
    :param threshold: the distance cutoff for clustering
    :return: a dictionary of {cluster id: [sample idx]}
    """
    uptri = np.triu_indices(pd_mat.shape[0], 1)
    divergences = pd_mat[uptri]
    Z = hierarchy.linkage(divergences, 'average')
    clusters = hierarchy.fcluster(Z, t=threshold, criterion='distance')
    d = _fclusters_to_dict(clusters)
    return d


def get_empirical_div_dist(local_divs, genome_divs, num_bins, separate_clades, clade_cutoff=0.03):
    # both local divs and genome divs are obtained by sampling QP pairs; call DH function
    # the center of the bin is returned
    # separate_clades and clade_cutoff are used to separate within/between clade donors
    bins = np.linspace(0, max(local_divs), num_bins + 1)
    divs = (bins[:-1] + bins[1:]) / 2
    if separate_clades:
        within_counts, _ = np.histogram(local_divs[genome_divs <= clade_cutoff], bins=bins)
        between_counts, _ = np.histogram(local_divs[genome_divs > clade_cutoff], bins=bins)
        divs = np.concatenate([divs, divs])
        counts = np.concatenate([within_counts, between_counts])
    else:
        counts, _ = np.histogram(local_divs, bins=bins)
    return divs, counts