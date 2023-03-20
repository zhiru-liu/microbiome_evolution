import sys
import os
import random
import numpy as np
import pandas as pd
import pickle
sys.path.append("..")
import config
from utils import parallel_utils, close_pair_utils, typical_pair_utils
import cphmm.cphmm as hmm


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


def test_species(species_name, species_params={}):
    dh = parallel_utils.DataHoarder(
        species_name, mode='QP', allowed_variants=['4D'])
    good_idxs = dh.get_single_subject_idxs()
    pd_mat = typical_pair_utils.load_pairwise_div_mat(species_name)
    if 'vulgatus' in species_name:
        clade_cutoff_bin = config.empirical_histogram_bins
        clade_cutoff_div = 0.03
    else:
        clade_cutoff_bin = None
        clade_cutoff_div = None

    def sample_strain():
        return random.sample(good_idxs, 1)[0]

    def get_random_transfer(focal_strain, lamb):
        # l must be an integer
        donor_strain = sample_strain()
        snp_vec, _ = dh.get_snp_vector((focal_strain, donor_strain))
        genome_len = len(snp_vec)
        l = int(np.random.exponential(lamb))
        start_idx = np.random.randint(0, genome_len - l + 1)
        return start_idx, l, snp_vec[start_idx:start_idx + l], donor_strain

    def generate_fake_genome(T, lamb, rbymu, L):
        genome = np.random.binomial(1, T, L)
        mask = np.ones_like(genome)
        focal_strain = sample_strain()
        num_transfers = np.random.binomial(L, T * rbymu)
        transfer_starts = []
        transfer_lens = []
        transfer_origin = [] # true = between clade transfer
        for i in xrange(num_transfers):
            while True:
                start, l, seq, donor = get_random_transfer(focal_strain, lamb)
                if start > L:
                    continue
                if np.sum(seq) <= 1:
                    # ignore transfers that are indistinguishable from mutation
                    continue
                # ignore transfer seq beyond genome end
                genome[start:start + l] = seq[:min(l, L - start)]
                mask[start:start + l] = 0
                transfer_starts.append(start)
                transfer_lens.append(min(l, L - start))
                if clade_cutoff_div is not None:
                    transfer_origin.append(pd_mat[donor, focal_strain] > clade_cutoff_div)
                break
        mask = mask.astype(bool)
        true_div = np.sum(genome[mask]) / float(np.sum(mask))
        true_clonal_len = np.sum(mask)
        return genome, true_div, true_clonal_len, transfer_starts, transfer_lens, transfer_origin

    def sample_pair():
        return random.sample(good_idxs, 2)

    def get_transfer(l):
        # l must be an integer
        pair = sample_pair()
        snp_vec, _ = dh.get_snp_vector(pair)
        start_idx = np.random.randint(0, len(snp_vec) - l)
        return snp_vec[start_idx:start_idx + l]

    def generate_fake_genome_old(T, lamb, rbymu, L):
        genome = np.random.binomial(1, T, L)
        num_transfers = np.random.binomial(L, T * rbymu)
        transfer_starts = np.random.randint(0, L, size=num_transfers)
        transfer_lens = np.random.exponential(lamb, num_transfers).astype(int)

        for i, x in enumerate(transfer_starts):
            l = transfer_lens[i]
            genome[x:x + l] = get_transfer(min(l, len(genome) - x))
        return genome, transfer_starts, transfer_lens

    dat = dict()
    dat['starts'] = []
    dat['ends'] = []
    dat['T est'] = []
    dat['naive T est'] = []
    dat['true counts'] = []
    dat['true T'] = []
    dat['true div'] = []
    dat['true clonal fraction'] = []
    dat['true lengths'] = []
    dat['true between clade counts'] = []
    dat['clonal fraction'] = []

    BLOCK_SIZE = config.second_pass_block_size
    L = species_params['genome length']
    mean_transfer_len = species_params['transfer len']
    rbymu = species_params['rbymu']

    if 'putredinis' in species_name:
        # load T distribution inferred with EM
        real_dist = pd.read_csv(os.path.join(config.analysis_directory, 'HMM_validation', 'Ap_rate_weights.csv'))
        n = 1849  # number of pairs for A putredinis
        Ts = np.random.choice(real_dist['Rates'], n, p=real_dist['Weights'])
        num_reps = 1
    else:
        Ts = np.arange(1, 16)*1e-5
        num_reps = species_params['reps']
    # [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 10e-5, 11e-5, 12e-5, 13e-5, 14e-5, 15e-5]
    # Ts = [6e-5, 7e-5, 8e-5, 9e-5, 1e-4]
    # Ts = [1e-4+1e-5, 1e-4+2e-5, 1e-4+3e-5, 1e-4+4e-5, 1e-4+5e-5]

    cphmm = init_hmm(species_name, L, BLOCK_SIZE)

    for T in Ts:
        for idx in xrange(num_reps):
            g, div, clonal_len, sim_starts, sim_lens, sim_origins = generate_fake_genome(
                T, mean_transfer_len, rbymu, int(L))
            blk_seq = close_pair_utils.to_block(g, BLOCK_SIZE).reshape((-1, 1))
            blk_seq_fit = (blk_seq > 0).astype(float)
            if np.sum(blk_seq) == 0:
                continue
            starts, ends, clonal_seq = close_pair_utils._decode_and_count_transfers(
                blk_seq_fit, cphmm, sequence_with_snps=blk_seq, clade_cutoff_bin=clade_cutoff_bin)
            naive_div, est_div = close_pair_utils.estimate_clonal_divergence(clonal_seq)

            # TODO: can measure the measured diverged of these locations
            start = close_pair_utils.block_loc_to_genome_loc(row['starts'], contig_lengths, block_size, left=True)
            end = close_pair_utils.block_loc_to_genome_loc(row['ends'], contig_lengths, block_size, left=False)
            div = np.sum(g[int(start):int(end)]) / float(end - start)
            divs.append(div)

            # saving data
            dat['true counts'].append(len(sim_starts))
            dat['true lengths'].append(sim_lens)
            dat['true T'].append(T)
            dat['true div'].append(div)
            dat['true clonal fraction'].append(clonal_len / float(L))
            dat['true between clade counts'].append(np.sum(sim_origins))

            dat['starts'].append(starts)
            dat['ends'].append(ends)
            dat['T est'].append(est_div)
            dat['naive T est'].append(naive_div)
            dat['clonal fraction'].append(len(clonal_seq)*BLOCK_SIZE / L)

            if (idx % 20) == 0:
                print("Finished {} reps".format(idx))

    save_path = os.path.join(
        config.analysis_directory, 'HMM_validation', '%s.pickle' % (species_name+species_params['suffix']))

    # data format will be almost identical to real pipeline, except also includes ground truth
    pickle.dump(dat, open(save_path, 'wb'))

# all_species = ['Alistipes_putredinis_61533',
#                'Bacteroides_vulgatus_57955',
#                'Eubacterium_rectale_56927']

all_species = ['Bacteroides_vulgatus_57955', 'Alistipes_putredinis_61533']
all_species_params = [
    # {'species': 'Bacteroides_vulgatus_57955', 'suffix': '', 'genome length':2.8e5, 'transfer len': 2600, 'rbymu': 0.65, 'reps':16, 'clade cutoff': 0.03},  # matched post hoc
    # {'species': 'Alistipes_putredinis_61533', 'suffix': '_EM_025', 'genome length':2.5e5, 'transfer len': 800, 'rbymu': 5.2, 'reps':100, 'clade cutoff': None},  # estimated using transfer/divergence
    # {'species': 'Alistipes_putredinis_61533', 'suffix': '_EM_050', 'genome length':2.5e5, 'transfer len': 800, 'rbymu': 3.4, 'reps':100, 'clade cutoff': None},  # estimated using transfer/divergence
    # {'species': 'Alistipes_putredinis_61533', 'suffix': '_EM_100', 'genome length':2.5e5, 'transfer len': 800, 'rbymu': 2.1, 'reps':100, 'clade cutoff': None},  # estimated using transfer/divergence
    {'species': 'Alistipes_putredinis_61533', 'suffix': '_EM_large', 'genome length':2.5e5, 'transfer len': 800, 'rbymu': 10, 'reps':100, 'clade cutoff': None},  # estimated using transfer/divergence
]

for d in all_species_params:
    test_species(d['species'], species_params=d)
