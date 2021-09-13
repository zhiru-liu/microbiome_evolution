import os
import pickle
import sys
sys.path.append("..")
from utils import parallel_utils, typical_pair_utils
import config


def process_species(species_name):
    print("\nProcessing {}".format(species_name))
    within_dh = parallel_utils.DataHoarder(species_name, mode='within', allowed_variants=['4D'])
    between_dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])
    within_pairs = typical_pair_utils.generate_within_sample_idxs(within_dh)
    between_pairs = typical_pair_utils.generate_between_sample_idxs(
        between_dh, num_pairs=5000)

    within_runs_data = typical_pair_utils.compute_runs(within_dh, within_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'within_hosts', '{}.pickle'.format(species_name))
    pickle.dump(within_runs_data, open(save_path, 'wb'))

    between_runs_data = typical_pair_utils.compute_runs(between_dh, between_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'between_hosts', '{}.pickle'.format(species_name))
    pickle.dump(between_runs_data, open(save_path, 'wb'))


def process_all_species():
    sample_df = parallel_utils.compute_good_sample_stats()
    sample_df = sample_df[sample_df['num_good_within_samples'] > 5]
    for species in sample_df['species_name']:
        process_species(species)


def process_B_vulgatus():
    species_name = 'Bacteroides_vulgatus_57955'
    within_dh = parallel_utils.DataHoarder(species_name, mode='within', allowed_variants=['4D'])
    between_dh = parallel_utils.DataHoarder(species_name, mode='QP', allowed_variants=['4D'])
    within_same_clade_pairs, within_diff_clade_pairs = typical_pair_utils.generate_within_sample_idxs(
        within_dh, clade_cutoff=0.03)
    # compute all qualified pairs
    between_same_clade_pairs, between_diff_clade_pairs = typical_pair_utils.generate_between_sample_idxs(
        between_dh, num_pairs=1e9, clade_cutoff=0.03)

    within_runs_data = typical_pair_utils.compute_runs(within_dh, within_same_clade_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'within_hosts', '{}_same_clade.pickle'.format(species_name))
    pickle.dump(within_runs_data, open(save_path, 'wb'))

    within_runs_data = typical_pair_utils.compute_runs(within_dh, within_diff_clade_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'within_hosts', '{}_diff_clade.pickle'.format(species_name))
    pickle.dump(within_runs_data, open(save_path, 'wb'))

    between_runs_data = typical_pair_utils.compute_runs(between_dh, between_same_clade_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'between_hosts', '{}_same_clade.pickle'.format(species_name))
    pickle.dump(between_runs_data, open(save_path, 'wb'))

    between_runs_data = typical_pair_utils.compute_runs(between_dh, between_diff_clade_pairs)
    save_path = os.path.join(config.analysis_directory, 'typical_pairs', 'runs_data', 'between_hosts', '{}_diff_clade.pickle'.format(species_name))
    pickle.dump(between_runs_data, open(save_path, 'wb'))


if __name__ == "__main__":
    process_B_vulgatus()
    # process_all_species()
