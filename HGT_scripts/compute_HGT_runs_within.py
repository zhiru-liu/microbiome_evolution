import time
import pickle
import os
import sys
import numpy as np
sys.path.append("..")
import config
from parsers import parse_midas_data
from utils import core_gene_utils, diversity_utils, HGT_utils


def get_passed_site_vector(passed_site_map, all_genes, id1, id2):
    # Used global all_genes to map to a vector
    # assuming only 4D sites
    all_num_sites = np.zeros(len(all_genes))
    for i in xrange(len(all_genes)):
        if all_genes[i] in passed_site_map:
            all_num_sites[i] = passed_site_map[all_genes[i]
                                               ]['4D']['sites'][id1, id2]
        else:
            all_num_sites[i] = 0
    return all_num_sites


def process_one_species(species_name):
    t0 = time.time()
    # load data first; computed from parse_snps_to_pickle.py
    all_genes = core_gene_utils.get_sorted_core_genes(species_name)
    data_dir = os.path.join(config.analysis_directory,
                            "within_hosts_checkpoints/")

    if os.path.exists("{}{}/all_runs_map.pickle".format(data_dir, species_name)):
        print('{} already processed'.format(species_name))
        return
    _, sfs_map = parse_midas_data.parse_within_sample_sfs(
            species_name, allowed_variant_types=set(['4D']))
    allele_counts_map = pickle.load(
        open("{}{}/allele_counts_map.pickle".format(data_dir, species_name), 'rb'))
    found_samples = pickle.load(
        open("{}{}/found_samples.pickle".format(data_dir, species_name), 'rb'))
    passed_sites_map = pickle.load(
        open("{}{}/passed_sites_map.pickle".format(data_dir, species_name), 'rb'))
    print("Finish loading data for {} at {} min".format(
        species_name, (time.time() - t0)/60))

    counts_map = dict()
    runs_map = dict()
    for sample_idx in xrange(len(found_samples)):
        sample_id = found_samples[sample_idx]
        gene_snp_map = HGT_utils.find_single_host_relative_snps(
                sample_idx, found_samples, allele_counts_map, sfs_map)
        if gene_snp_map is None:
            print("Sample {} has no clear peak".format(sample_id))
            continue
        all_gene_counts = HGT_utils.get_gene_snp_vector(
                gene_snp_map, all_genes)
        counts_map[sample_idx] = sum(all_gene_counts)
        runs, starts, ends = HGT_utils.find_runs(all_gene_counts)

        # Now count the number of passed sites for each run
        passed_site_vec = get_passed_site_vector(
                passed_sites_map, all_genes, sample_idx, sample_idx)
        site_counts = np.array([sum(passed_site_vec[start:end+1])
                                for (start, end) in zip(starts, ends)])
        # Now count the number of anolamous events
        runs_map[sample_idx] = (runs, starts, ends, site_counts)

    # save data
    pickle.dump(runs_map, open(
        "{}{}/all_runs_map.pickle".format(data_dir, species_name), 'wb'))
    pickle.dump(counts_map, open(
        "{}{}/snp_counts_map.pickle".format(data_dir, species_name), 'wb'))
    print("Finish saving data for {} at {} min".format(
        species_name, (time.time() - t0)/60))


checkpoint_dir = os.path.join(config.analysis_directory, 'within_hosts_checkpoints')
for species_name in os.listdir(checkpoint_dir):
    if species_name.startswith('.'):
        continue
    process_one_species(species_name)
