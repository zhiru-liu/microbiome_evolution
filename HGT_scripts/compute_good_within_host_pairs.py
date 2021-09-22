import numpy as np
import os
import sys
import csv
sys.path.append("..")
from utils import close_pair_utils, parallel_utils
import config


base_dir = 'zarr_snps'
csv_file_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'sample_stats.csv')
csv_file = open(csv_file_dir, 'w')
writer = csv.writer(csv_file)
writer.writerow(['species_name', 'num_samples', 'num_qp_samples',
                   'num_good_within_samples'])

for species_name in os.listdir(os.path.join(config.data_directory, 'zarr_snps')):
    if species_name.startswith('.'):
        continue
    print("processing %s" % species_name)
    qp_mask, _ = parallel_utils.get_QP_sample_mask(species_name)
    good_within_mask, _, _, _ = parallel_utils.get_single_peak_sample_mask(species_name)

    writer.writerow([species_name, len(qp_mask), np.sum(qp_mask), np.sum(good_within_mask)])

csv_file.close()
