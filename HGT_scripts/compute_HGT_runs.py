import sys
import pickle
import os
import time
import numpy as np
from functools import partial
from multiprocessing import Pool, Manager
import itertools
sys.path.append("..")
from utils import core_gene_utils, diversity_utils, HGT_utils
import config

'''
    Script for computing between host sharing blocks/runs. Mainly
    producing all_runs_map and snp_counts_map.

    all_runs_map: for each pair (idx1, idx2), save tuble of four arrays:
    (runs, starts, ends, site_counts), where runs are run lengths in genes,
    starts ands ends are gene indices, site_counts are number of sites in
    runs.

    snp_counts_map: total number of snps for a pair
'''

# define them as global variables so that can be shared by multiprocessing
all_genes = None
allele_counts_map = None
passed_sites_map = None
found_samples = None


def get_core_gene_vector(species_name):
    core_genes = core_gene_utils.parse_core_genes(species_name)
    # sort the genes
    all_genes = np.array(list(core_genes))
    gene_indices = np.array(
        list(map(lambda name: int(name.split('.')[-1]), all_genes)))
    all_genes = all_genes[np.argsort(gene_indices)]
    return all_genes


def get_passed_site_vector(passed_site_map, id1, id2):
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


def get_total_passed_sites(passed_site_map, id1, id2):
    # only 4D!
    return sum(get_passed_site_vector(passed_site_map, id1, id2))


def process_one_pair(counts_map, runs_map, pair_idx):
    # For parallel processing
    idx1 = pair_idx[0]
    idx2 = pair_idx[1]
    if idx1 >= idx2:
        # Save only half of the comparisons
        return
    gene_snp_map, _ = HGT_utils.get_two_sample_SNP_genes(
        [idx1, idx2], allele_counts_map)
    # Get a vector of all the snp counts ordered according to the order of genes
    all_gene_counts = np.zeros(len(all_genes))

    for gene in gene_snp_map:
        all_gene_counts[np.where(all_genes == gene)[0]] = gene_snp_map[gene]

    counts_map[(idx1, idx2)] = sum(all_gene_counts)
    runs, starts, ends = HGT_utils.find_runs(all_gene_counts)

    # Now count the number of passed sites for each run
    passed_site_vec = get_passed_site_vector(passed_sites_map, idx1, idx2)
    site_counts = np.array([sum(passed_site_vec[start:end+1])
                            for (start, end) in zip(starts, ends)])
    # Now count the number of anolamous events
    runs_map[(idx1, idx2)] = (runs, starts, ends, site_counts)


def compute_one_species(species_name):
    t0 = time.time()
    # load data first; computed from parse_snps_to_pickle.py
    global all_genes
    all_genes = get_core_gene_vector(species_name)
    data_dir = os.path.join(config.analysis_directory,
                            "between_hosts_checkpoints/")
    if os.path.exists("{}{}/all_runs_map.pickle".format(data_dir, species_name)):
        print('{} already processed'.format(species_name))
        return
    global allele_counts_map, found_samples, passed_sites_map
    allele_counts_map = pickle.load(
        open("{}{}/allele_counts_map.pickle".format(data_dir, species_name), 'rb'))
    found_samples = pickle.load(
        open("{}{}/found_samples.pickle".format(data_dir, species_name), 'rb'))
    passed_sites_map = pickle.load(
        open("{}{}/passed_sites_map.pickle".format(data_dir, species_name), 'rb'))
    print("Finish loading data for {} at {} min".format(
        species_name, (time.time() - t0)/60))

    manager = Manager()
    snp_counts = manager.dict()
    all_runs = manager.dict()

    num_samples = len(found_samples)

    func = partial(process_one_pair, snp_counts, all_runs)
    job_list = list(itertools.product(
        xrange(num_samples), xrange(num_samples)))
    pool = Pool()
    _ = pool.map(func, job_list)

    print("Finish runs computation for {} at {} min".format(
        species_name, (time.time() - t0)/60))

    all_runs_map = all_runs._getvalue()
    snp_counts_map = snp_counts._getvalue()

    pickle.dump(all_runs_map, open(
        "{}{}/all_runs_map.pickle".format(data_dir, species_name), 'wb'))
    pickle.dump(snp_counts_map, open(
        "{}{}/snp_counts_map.pickle".format(data_dir, species_name), 'wb'))
    print("Finish saving data for {} at {} min".format(
        species_name, (time.time() - t0)/60))


if __name__ == "__main__":
    desired_species = ['Bacteroides_vulgatus_57955',
                       'Bacteroides_uniformis_57318',
                       'Bacteroides_stercoris_56735',
                       'Bacteroides_caccae_53434',
                       'Bacteroides_ovatus_58035',
                       'Bacteroides_thetaiotaomicron_56941',
                       'Bacteroides_xylanisolvens_57185',
                       'Bacteroides_massiliensis_44749',
                       'Bacteroides_cellulosilyticus_58046',
                       'Bacteroides_fragilis_54507',
                       'Bacteroides_eggerthii_54457',
                       'Bacteroides_coprocola_61586',
                       'Prevotella_copri_61740',
                       'Barnesiella_intestinihominis_62208',
                       'Alistipes_putredinis_61533',
                       'Alistipes_shahii_62199',
                       'Alistipes_finegoldii_56071',
                       'Bacteroidales_bacterium_58650',
                       'Ruminococcus_bromii_62047',
                       'Eubacterium_siraeum_57634',
                       'Coprococcus_sp_62244',
                       'Roseburia_intestinalis_56239',
                       'Roseburia_inulinivorans_61943',
                       'Dialister_invisus_61905',
                       'Escherichia_coli_58110',
                       'Faecalibacterium_cf_62236']

    for species in desired_species:
        compute_one_species(species)
