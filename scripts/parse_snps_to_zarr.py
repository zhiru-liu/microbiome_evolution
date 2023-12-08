import sys
sys.path.append("..")
import pickle
import config
from utils import core_gene_utils, diversity_utils, snp_data_utils
from parsers.parse_midas_data import parse_snps
import os
import time


# picked by hand
# desired_species = ['Bacteroides_vulgatus_57955',
#                    'Bacteroides_uniformis_57318',
#                    'Bacteroides_stercoris_56735',
#                    'Bacteroides_caccae_53434',
#                    'Bacteroides_ovatus_58035',
#                    'Bacteroides_thetaiotaomicron_56941',
#                    'Bacteroides_xylanisolvens_57185',
#                    'Bacteroides_massiliensis_44749',
#                    'Bacteroides_cellulosilyticus_58046',
#                    'Bacteroides_fragilis_54507',
#                    'Bacteroides_eggerthii_54457',
#                    'Bacteroides_coprocola_61586',
#                    'Prevotella_copri_61740',
#                    'Barnesiella_intestinihominis_62208',
#                    'Alistipes_putredinis_61533',
#                    'Alistipes_shahii_62199',
#                    'Alistipes_finegoldii_56071',
#                    'Bacteroidales_bacterium_58650',
#                    'Ruminococcus_bromii_62047',
#                    'Eubacterium_siraeum_57634',
#                    'Coprococcus_sp_62244',
#                    'Roseburia_intestinalis_56239',
#                    'Roseburia_inulinivorans_61943',
#                    'Dialister_invisus_61905',
#                    'Escherichia_coli_58110',
#                    'Faecalibacterium_cf_62236',
#                    'Eubacterium_rectale_56927',
#                    'Bacteroides_plebeius_61623',
#                    'Bacteroides_finegoldii_57739',
#                    'Parabacteroides_merdae_56972',
#                    'Odoribacter_splanchnicus_62174',
#                    'Parabacteroides_distasonis_56985',
#                    'Alistipes_onderdonkii_55464',
#                    'Oscillibacter_sp_60799',
#                    'Akkermansia_muciniphila_55290']


def parse_site_counts():
    line_count_file = os.path.join(
            config.analysis_directory, "annotated_snps_line_counts.txt")
    with open(line_count_file, 'r') as f:
        lines = f.read().splitlines()
        pairs = [x.split() for x in lines]
        # the second number is the total number of lines
        # minus one to exclude the header line
        count_map = {x[0]: int(x[1]) - 1 for x in pairs}
    return count_map


# Parse and save all the snps into a giant zarr array
def main():
    # get the total number of snps in file for all species
    species_site_count_map = parse_site_counts()
    with open(os.path.join(config.data_directory, 'plosbio_fig2_species.txt'), 'r') as f:
        desired_species = f.read().splitlines()
        print('in total %d species to process' % len(desired_species))

    for species_name in desired_species:
        site_counts = species_site_count_map[species_name]
        snp_file_path = os.path.join(
                config.data_directory, "snps", species_name, "annotated_snps.txt.bz2")
        zarr_path = os.path.join(
                config.data_directory, 'zarr_snps', species_name)

        print("Start processing {}".format(species_name))
        print("{} has {} sites to be processed".format(species_name, site_counts))
        if not os.path.exists(zarr_path):
            print('{} has not been processed'.format(species_name))
            os.mkdir(zarr_path)
        else:
            print('{} already processed'.format(species_name))
            continue
        parallel_utils.parse_annotated_snps_to_zarr(snp_file_path, zarr_path, site_counts)


if __name__ == "__main__":
    main()
