import os
import sys
import time
sys.path.append("..")
import config
from utils import snp_data_utils


def main():
    t0 = time.time()
    base_dir = 'zarr_snps'
    for species_name in os.listdir(os.path.join(config.analysis_directory, base_dir)):
        if species_name.startswith('.'):
            continue
        print("Saving for {} at {} min".format(species_name, (time.time()-t0)/60))

        dh = snp_data_utils.DataHoarder(species_name)
        path_to_file = os.path.join('/Users/Device6/Documents/Research/bgoodlab/',
                                    'fineSTRUCTURE', 'microbiome', species_name)
        dh.save_haplotype_fs(path_to_file)

if __name__ == "__main__":
    main()
