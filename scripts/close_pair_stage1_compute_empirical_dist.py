import sys
import os
import random
import numpy as np
sys.path.append("..")
import config
from utils import snp_data_utils
from utils.close_pair_utils import sample_blocks


def get_empirical_div_dist(local_divs, genome_divs, num_bins, separate_clades=True, clade_cutoff=0.03):
    # Prepare empirical distribution for Tc of transferred blocks
    # Both local divs and genome divs are obtained by sampling QP pairs
    # The center of the bin is returned
    # For B. vulgatus and others that need to classify the transfer, separate_clades=True
    # then the dist of within-clade and between-clade will be concat together
    bins = np.linspace(0, max(local_divs), num_bins + 1)
    divs = (bins[:-1] + bins[1:]) / 2
    if separate_clades:
        within_counts, _ = np.histogram(local_divs[genome_divs <= clade_cutoff], bins=bins)
        between_counts, _ = np.histogram(local_divs[genome_divs > clade_cutoff], bins=bins)
        divs = np.concatenate([divs, divs])
        counts = np.concatenate([within_counts, between_counts])
    else:
        counts, _ = np.histogram(local_divs, bins=bins)
    return divs, counts


base_dir = 'zarr_snps'
for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
    if species_name.startswith('.'):
        continue
    print('Processing ' + species_name)
    save_path = os.path.join(config.hmm_data_directory, species_name + '.csv')
    if os.path.exists(save_path):
        print("%s already processed" % species_name)
        continue
    if species_name == 'Bacteroides_vulgatus_57955':
        separate_clades = True
        clade_cutoff = 0.03
    elif species_name == 'Alistipes_shahii_62199':
        separate_clades = True
        clade_cutoff = 0.04
    else:
        separate_clades = False
        clade_cutoff = None
    dh = snp_data_utils.DataHoarder(species_name, mode="QP")
    local_divs, genome_divs = sample_blocks(dh)
    divs, counts = get_empirical_div_dist(local_divs, genome_divs,
                                          num_bins=40, separate_clades=separate_clades,
                                          clade_cutoff=clade_cutoff)
    save_path = os.path.join(config.hmm_data_directory, species_name + '.csv')
    np.savetxt(save_path, np.vstack([divs, counts]))
