import numpy as np
import os
import pandas as pd
from utils import parallel_utils, core_gene_utils
import config

base_dir = 'zarr_snps'
all_species = []
all_syn_counts = []
all_counts = []
for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
    if species_name.startswith('.'):
        continue
    snp_info = parallel_utils.get_snp_info(species_name)
    core_genes = core_gene_utils.get_sorted_core_genes(species_name)
    allowed_variants = ['4D']
    mask = parallel_utils._get_general_site_mask(snp_info[2], snp_info[3], snp_info[4], core_genes, allowed_variants=allowed_variants)
    syn_site_count = np.sum(mask)

    allowed_variants = ['1D', '2D', '3D', '4D']
    mask = parallel_utils._get_general_site_mask(snp_info[2], snp_info[3], snp_info[4], core_genes, allowed_variants=allowed_variants)
    all_site_count = np.sum(mask)
    all_species.append(species_name)
    all_syn_counts.append(syn_site_count)
    all_counts.append(all_site_count)
d = {0: all_species, 1: all_counts, 2: all_syn_counts}
df = pd.DataFrame(data=d)
df.columns = ['Species names', 'Core genome length', '4D synonymous site counts']
df['Core to syn ratio'] = df['Core genome length'].astype(float) / df['4D synonymous site counts']
df.to_csv(os.path.join(config.analysis_directory, 'misc', 'genome_length.csv'))
