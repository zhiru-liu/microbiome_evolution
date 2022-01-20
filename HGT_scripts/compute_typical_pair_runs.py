import os
import pickle
import sys
import collections
import itertools
import numpy as np
sys.path.append("..")
from utils import parallel_utils, typical_pair_utils, close_pair_utils
from parsers import parse_HMP_data
import config


def process_species(species_name, skip_between=False):
    print("\nProcessing {}".format(species_name))
    within_dh = parallel_utils.DataHoarder(species_name, mode='within', allowed_variants=['4D'])
    within_pairs = typical_pair_utils.generate_within_sample_idxs(within_dh)

    print("Processing within host pairs")
    within_runs_data = typical_pair_utils.compute_runs(within_dh, within_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'within_hosts', '{}.pickle'.format(species_name))
    pickle.dump(within_runs_data, open(save_path, 'wb'))

    if skip_between:
        return
    between_dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])
    between_pairs = typical_pair_utils.generate_between_sample_idxs(
        between_dh, num_pairs=5000)
    print("Processing between host pairs")
    between_runs_data = typical_pair_utils.compute_runs(between_dh, between_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'between_hosts', '{}.pickle'.format(species_name))
    pickle.dump(between_runs_data, open(save_path, 'wb'))


def process_all_species(skip_between=False):
    sample_df = parallel_utils.compute_good_sample_stats()
    sample_df = sample_df[sample_df['num_good_within_samples'] > 5]
    for species in sample_df['species_name']:
        process_species(species, skip_between=skip_between)


def process_E_rectale():
    species_name = 'Eubacterium_rectale_56927'
    within_dh = parallel_utils.DataHoarder(species_name, mode='within', allowed_variants=['4D'])

    print("Processing within host pairs")
    within_pairs = typical_pair_utils.generate_within_sample_idxs(within_dh)
    within_runs_data = typical_pair_utils.compute_runs(within_dh, within_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'within_hosts',
                             '{}.pickle'.format(species_name))
    pickle.dump(within_runs_data, open(save_path, 'wb'))

    country_counts_dict = typical_pair_utils.get_E_rectale_within_host_countries(within_dh)

    print("Processing between host pairs")
    between_dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])
    country_pairs = typical_pair_utils.generate_between_sample_idxs_control_country(between_dh, country_counts_dict, num_pairs=3000)
    between_pairs = list(itertools.chain.from_iterable(country_pairs.values()))

    between_runs_data = typical_pair_utils.compute_runs(between_dh, between_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'between_hosts',
                             '{}.pickle'.format(species_name))
    pickle.dump(between_runs_data, open(save_path, 'wb'))



def process_B_vulgatus(skip_between_host=False):
    species_name = 'Bacteroides_vulgatus_57955'
    within_dh = parallel_utils.DataHoarder(species_name, mode='within', allowed_variants=['4D'])
    within_same_clade_pairs, within_diff_clade_pairs = typical_pair_utils.generate_within_sample_idxs(
        within_dh, clade_cutoff=0.03, clonal_frac_cutoff=0.1)

    print("Processing within host, within clade pairs")
    within_runs_data = typical_pair_utils.compute_runs(within_dh, within_same_clade_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'within_hosts', '{}_same_clade.pickle'.format(species_name))
    pickle.dump(within_runs_data, open(save_path, 'wb'))

    print("Processing within host, between clade pairs")
    within_runs_data = typical_pair_utils.compute_runs(within_dh, within_diff_clade_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'within_hosts', '{}_diff_clade.pickle'.format(species_name))
    pickle.dump(within_runs_data, open(save_path, 'wb'))

    if skip_between_host:
        return
    between_dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])
    # compute all qualified pairs
    between_same_clade_pairs, between_diff_clade_pairs = typical_pair_utils.generate_between_sample_idxs(
        between_dh, num_pairs=1e9, clade_cutoff=0.03, clonal_frac_cutoff=0.1)

    print("Processing between host, between clade pairs")
    between_runs_data = typical_pair_utils.compute_runs(between_dh, between_same_clade_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'between_hosts', '{}_same_clade.pickle'.format(species_name))
    pickle.dump(between_runs_data, open(save_path, 'wb'))

    print("Processing between host, between clade pairs")
    between_runs_data = typical_pair_utils.compute_runs(between_dh, between_diff_clade_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'between_hosts', '{}_diff_clade.pickle'.format(species_name))
    pickle.dump(between_runs_data, open(save_path, 'wb'))


if __name__ == "__main__":
    # process_B_vulgatus(skip_between_host=True)
    # process_all_species(skip_between=True)
    process_E_rectale()
    # process_species("Eubacterium_rectale_56927", skip_between=True)
