import sys
import json
import config
import os
from utils import sample_utils, diversity_utils, species_phylogeny_utils
from parsers import parse_midas_data, parse_HMP_data

# adapted from PLOS Bio plot_figure_2.py
min_sample_size = config.between_host_min_sample_size  # 46 gives at least 1000 pairs, 33 gives at least 500 (actually 528)
subject_sample_map = parse_HMP_data.parse_subject_sample_map()
good_species_list = parse_midas_data.parse_good_species_list()
good_species = []
sample_sizes = {}
for species_name in good_species_list:
    sys.stderr.write("Loading haploid samples...\n")
    # Only plot samples above a certain depth threshold that are "haploids"
    snp_samples = diversity_utils.calculate_haploid_samples(species_name, debug=False)

    if len(snp_samples) < min_sample_size:
        sys.stderr.write("Not enough haploid samples!\n")
        continue

    sys.stderr.write("Calculating unique samples...\n")
    # Only consider one sample per person
    snp_samples = snp_samples[sample_utils.calculate_unique_samples(subject_sample_map, sample_list=snp_samples)]

    if len(snp_samples) < min_sample_size:
        sys.stderr.write("Not enough unique samples!\n")
        continue
    good_species.append(species_name)
    sample_sizes[species_name] = len(snp_samples)


species_names = []
num_samples = []

for species_name in good_species:
    species_names.append(species_name)

    if species_name == 'Bacteroides_vulgatus_57955':
        num_samples.append(-1000)
    else:
        num_samples.append(-sample_sizes[species_name])

sorted_species_names = species_phylogeny_utils.sort_phylogenetically(good_species,
                                                                     first_entry='Bacteroides_vulgatus_57955',
                                                                     second_sorting_attribute=num_samples)
species_priority = dict([(b, a) for a, b in enumerate(sorted_species_names)])


json.dump(species_priority, open(os.path.join(config.analysis_directory, 'species_plotting_priority.json'), 'w'))