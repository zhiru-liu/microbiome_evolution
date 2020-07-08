import sys
sys.path.append("..")
import pickle
import config
from utils import core_gene_utils, diversity_utils
from parsers.parse_midas_data import parse_snps
import os
import time


# zhiru: use this to save intermediate results of parse_snps() to pickle files
def main(between_host):
    # Parse and save all the snps between QP hosts
    t0 = time.time()
    intermediate_file_path = os.path.join(config.analysis_directory, 'within_hosts_checkpoints')
    for species_name in desired_species:
        print("Start processing {}".format(species_name))
        core_genes = core_gene_utils.parse_core_genes(species_name)
        desired_samples = get_desired_samples(species_name, between_host)
        if desired_samples is None or len(desired_samples) == 0:
            print("{} has no qualified samples".format(species_name))
            continue
        found_samples, allele_counts_map, passed_sites_map, final_line_number = parse_snps(
            species_name, allowed_samples=desired_samples, allowed_genes=core_genes, allowed_variant_types=['4D'])
        pickle_path = intermediate_file_path + species_name
        if not os.path.exists(pickle_path):
            os.mkdir(pickle_path)
        pickle.dump(allele_counts_map, open(
            pickle_path + '/allele_counts_map.pickle', 'wb'))
        pickle.dump(found_samples, open(
            pickle_path + '/found_samples.pickle', 'wb'))
        pickle.dump(passed_sites_map, open(
            pickle_path + '/passed_sites_map.pickle', 'wb'))
        print("Done processing {} at {} min".format(
            species_name, (time.time() - t0) / 60))


def get_desired_samples(species_name, between_hosts=False):
    highcoverage_samples = set(
        diversity_utils.calculate_highcoverage_samples(species_name))
    if between_hosts:
        QP_samples = set(
            diversity_utils.calculate_haploid_samples(species_name))
        return QP_samples & highcoverage_samples
    else:
        single_peak_dir = os.path.join(config.analysis_directory, 'allele_freq', species_name, '1')
        if not os.path.exists(single_peak_dir):
            print("Please plot sfs by peaks first for {}".format(species_name))
            return None
        desired_samples = set([f.split('.')[0] for f in os.listdir(single_peak_dir) if not f.startswith('.')])
        return desired_samples & highcoverage_samples

desired_species = ['Bacteroides_caccae_53434',
                   'Bacteroides_thetaiotaomicron_56941',
                   'Bacteroides_xylanisolvens_57185',
                   'Prevotella_copri_61740',
                   'Bacteroidales_bacterium_58650',
                   'Bacteroides_eggerthii_54457',
                   'Dialister_invisus_61905',
                   'Bacteroides_vulgatus_57955',
                   'Bacteroides_uniformis_57318',
                   'Barnesiella_intestinihominis_62208',
                   'Bacteroides_fragilis_54507']

if __name__ == "__main__":
    main(False)
