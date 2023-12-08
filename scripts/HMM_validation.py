import sys
import os
import random
import numpy as np
import pickle
sys.path.append("..")
import config
from utils import snp_data_utils, close_pair_utils, hmm

""" Old code for naive HMM validation. Updated code in CPHMM_validation.py """


def test_species(species_name, transfer_div=None, debug=False):
    dh = snp_data_utils.DataHoarder(
        species_name, mode='QP', allowed_variants=['1D', '2D', '3D', '4D'])
    good_idxs = dh.get_single_subject_idxs()

    def sample_pair():
        return random.sample(good_idxs, 2)

    def get_transfer(l):
        # l must be an integer
        pair = sample_pair()
        snp_vec, _ = dh.get_snp_vector(pair)
        start_idx = np.random.randint(0, len(snp_vec) - l)
        return snp_vec[start_idx:start_idx + l]

    def generate_fake_genome(T, lamb, rbymu, L):
        genome = np.random.binomial(1, T, L)
        num_transfers = np.random.binomial(L, T * rbymu)
        transfer_starts = np.random.randint(0, L, size=num_transfers)
        transfer_lens = np.random.exponential(lamb, num_transfers).astype(int)

        for i, x in enumerate(transfer_starts):
            l = transfer_lens[i]
            genome[x:x + l] = get_transfer(min(l, len(genome) - x))
        return genome, transfer_starts, transfer_lens

    all_data = []
    BLOCK_SIZE = 2000
    num_reps = 1 if debug else 100
    L = 1e6
    mean_transfer_len = 5000
    rbymu = 1
    Ts = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]

    if transfer_div:
        transfer_mean = transfer_div * BLOCK_SIZE
        hmm_init_means = [0.5, transfer_mean]
    else:
        hmm_init_means = [0.5, 5]

    num_states = 2
    phmm = hmm.PoissonHMM(init_means=hmm_init_means,
                          n_components=num_states, params='st')
    for T in Ts:
        true_counts = []
        counts = []
        T_approxs = []
        for i in range(num_reps):
            g, sim_starts, sim_lens = generate_fake_genome(
                T, mean_transfer_len, rbymu, int(L))
            seq = close_pair_utils.to_block(g, BLOCK_SIZE).reshape((-1, 1))
            starts, ends, T_approx = close_pair_utils._fit_and_count_transfers_iterative(
                seq, phmm, BLOCK_SIZE, iters=3)
            phmm.init_means = hmm_init_means
            true_counts.append(len(sim_starts))
            counts.append(len(starts))
            T_approxs.append(T_approx)
        all_data.append((true_counts, counts, T_approxs))
    save_path = os.path.join(
        config.analysis_directory, 'HMM_validation', '%s.pickle' % species_name)
    pickle.dump(all_data, open(save_path, 'wb'))

all_species = ['Alistipes_putredinis_61533',
               'Bacteroides_vulgatus_57955',
               'Eubacterium_rectale_56927']
transfer_divs = [0.007, 0.003, 0.015]

for i, species in enumerate(all_species):
    test_species(species, transfer_div=transfer_divs[i], debug=False)