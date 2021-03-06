import sys
import numpy
from numpy.random import normal
from utils import diversity_utils, parse_midas_data, stats_utils
from parsers import parse_patric

################################################################################
#
# Standard header to read in argument information
#
################################################################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("species_name", help="name of species to process")
args = parser.parse_args()

species_name = args.species_name
################################################################################

# get a list of genome_ids:
genome_ids= parse_midas_data.get_ref_genome_ids(species_name)

#write new genome features file:
for genome_id in genome_ids:
    parse_patric.new_genome_features_file(genome_id)
