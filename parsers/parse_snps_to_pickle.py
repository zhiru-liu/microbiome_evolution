import pickle
from utils import core_gene_utils, diversity_utils
from parsers.parse_midas_data import parse_snps
import os


def main():
    # Parse and save all the snps between QP hosts
    for species_name in desired_species:
        core_genes = core_gene_utils.parse_core_genes(species_name)
        QP_samples = set(diversity_utils.calculate_haploid_samples(species_name))
        highcoverage_samples = set(diversity_utils.calculate_highcoverage_samples(species_name))
        desired_samples = QP_samples & highcoverage_samples
        found_samples, allele_counts_map, passed_sites_map, final_line_number = parse_snps(
            species_name, allowed_samples=desired_samples, allowed_genes=core_genes, allowed_variant_types=['4D'])
        pickle_path = intermediate_file_path + species_name
        if not os.path.exists(pickle_path):
            os.mkdir(pickle_path)
        pickle.dump(allele_counts_map, open(pickle_path + '/allele_counts_map.pickle', 'wb'))
        pickle.dump(found_samples, open(pickle_path + '/found_samples.pickle', 'wb'))


desired_species = ['Bacteroides_caccae_53434', 'Bacteroides_thetaiotaomicron_56941', 'Bacteroides_xylanisolvens_57185', 
        'Prevotella_copri_61740', 'Bacteroidales_bacterium_58650', 'Bacteroides_eggerthii_54457', 'Dialister_invisus_61905']
intermediate_file_path = '/Users/Device6/Documents/Research/bgoodlab/microbiome_evolution/outputs/between_hosts_checkpoints/'
if __name__ == "__main__":
    main()
