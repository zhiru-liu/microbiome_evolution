import os
import numpy as np
import pandas as pd
import random
import config
from utils import parallel_utils, close_pair_utils
import cphmm.cphmm as hmm
from datetime import datetime
"""
Improving the location controlled simulation even further by passing the simulated genomes through cphmm
"""


def init_hmm(species_name, genome_len, block_size):
    # initialize the hmm with default params
    # clonal emission and transfer rate will be fitted per sequence later in the pipeline
    num_blocks = genome_len / block_size
    transfer_counts = 20.
    clonal_div = 5e-5
    transfer_length = 1000.

    transfer_rate = transfer_counts / num_blocks
    transfer_length = transfer_length / block_size
    clonal_emission = clonal_div * block_size
    cphmm = hmm.ClosePairHMM(species_name=species_name, block_size=block_size,
                             transfer_rate=transfer_rate, clonal_emission=clonal_emission,
                             transfer_length=transfer_length, n_iter=5)
    return cphmm


def generate_simulated_genome(dh, focal_pair, events):
    def sample_strain():
        return random.sample(dh.get_single_subject_idxs(), 1)[0]

    snp_vec, coverage_vec = dh.get_snp_vector(focal_pair)
    block_size = config.second_pass_block_size

    # for transforming between indicies (blk_seq, snp_vec, and the full 4D core genome vec)
    good_chromo = dh.chromosomes[dh.general_mask][coverage_vec]
    contig_lengths = parallel_utils.get_contig_lengths(good_chromo)
    snp_vec_to_core = np.where(coverage_vec)[0]

    clonal_div = events.iloc[0, :]['clonal divergence']
    simulated_genome = np.random.binomial(1, clonal_div, len(coverage_vec))
    for _, row in events.iterrows():
        blk_start = row['starts']
        blk_end = row['ends']
        start = close_pair_utils.block_loc_to_genome_loc(blk_start, contig_lengths, block_size, left=True)
        end = close_pair_utils.block_loc_to_genome_loc(blk_end, contig_lengths, block_size, left=False)

        core_start, core_end = snp_vec_to_core[int(start)], snp_vec_to_core[int(end) - 1]  # core end is inclusive now

        # choose random strain from the focal pair as the recipient
        focal_strain = random.sample(focal_pair, 1)[0]
        donor_strain = sample_strain()

        new_snp_vec, new_coverage_vec = dh.get_snp_vector((focal_strain, donor_strain))
        core_snp_vec = new_coverage_vec.copy()
        core_snp_vec[new_coverage_vec] = new_snp_vec
        transfer_genotype = core_snp_vec[core_start:core_end + 1]

        simulated_genome[core_start:core_end + 1] = transfer_genotype
    return simulated_genome


def find_event_divergences(starts, ends, blk_seq):
    """
    Taking the output from cphmm to decode the divergence of each of the transfer
    :param starts:
    :param ends:
    :param blk_seq:
    :return:
    """
    divs = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        length = float((end - start + 1) * config.second_pass_block_size)
        div = np.sum(blk_seq[start:end + 1]) / length
        divs.append(div)
    return divs


def generate_fake_transfer_dist(species_name, total_reps=5):
    # only processing B. vulgatus as two clades
    if 'vulgatus' in species_name:
        clade_cutoff_bin = config.empirical_histogram_bins
    else:
        clade_cutoff_bin = None

    dh = parallel_utils.DataHoarder(
        species_name, mode='QP', allowed_variants=['4D'])
    L = dh.general_mask.sum()
    cphmm = init_hmm(species_name, L, config.second_pass_block_size)

    # first loading all transfer location
    save_path = os.path.join(config.analysis_directory,
                             "closely_related", "third_pass", "{}_all_transfers.pickle".format(species_name))
    cf_cutoff = config.clonal_fraction_cutoff
    run_df = pd.read_pickle(save_path)
    full_df = run_df[run_df['clonal fraction'] > cf_cutoff]
    num_pairs = len(pd.unique(full_df['pairs']))
    print("Simulating {} random transfers in {} pairs".format(full_df.shape[0], num_pairs))
    all_sim_divs = []
    num_processed = 0
    for focal_pair, events in full_df.groupby('pairs'):
        for rep in range(total_reps):
            g = generate_simulated_genome(dh, focal_pair, events)
            blk_seq = close_pair_utils.to_block(g, config.second_pass_block_size).reshape((-1, 1))
            blk_seq_fit = (blk_seq > 0).astype(float)
            if np.sum(blk_seq) == 0:
                continue
            inferred_starts, inferred_ends, _ = close_pair_utils._decode_and_count_transfers(
                blk_seq_fit, cphmm, sequence_with_snps=blk_seq, clade_cutoff_bin=clade_cutoff_bin)
            inferred_starts = np.concatenate(inferred_starts)
            inferred_ends = np.concatenate(inferred_ends)  # no distinction between transfer types
            divs = find_event_divergences(inferred_starts, inferred_ends, blk_seq)
            all_sim_divs.append(divs)
        num_processed += 1
        if (num_processed % 100) == 0:
            print("Finished {} pairs at {}".format(num_processed, datetime.now()))
    all_sim_divs = np.concatenate(all_sim_divs)
    np.savetxt(os.path.join(config.analysis_directory, 'closely_related', 'simulated_transfers_cphmm', '{}.csv'.format(species_name)), all_sim_divs)


if __name__ == "__main__":
    second_pass_dir = os.path.join(config.analysis_directory, "closely_related", "second_pass")
    for filename in os.listdir(second_pass_dir):
        if filename.startswith('.'):
            continue
        species_name = filename.split('.')[0]
        if 'Lachnospiraceae' in species_name:
            continue
        print("Processing {}".format(species_name))
        generate_fake_transfer_dist(species_name)
        break
