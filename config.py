###############################################################################
#
# Set up default source and output directories
#
###############################################################################
import os.path 
from math import log10

data_directory = os.path.expanduser("/Volumes/Botein/microbiome_data/")
#data_directory = os.path.expanduser("~/ben_nandita_hmp_data_071518/")
#data_directory = os.path.expanduser("~/ben_nandita_hmp_data/")
analysis_directory = os.path.expanduser("/Volumes/Botein/zhiru_analysis/")
plotting_intermediate_directory = os.path.expanduser("/Volumes/Botein/plotting_intermediate_files/")
figure_directory = os.path.expanduser("/Volumes/Botein/figs/")
scripts_directory = os.path.expanduser("/Users/Device6/Documents/Research/bgoodlab/microbiome_evolution/")
patric_directory = os.path.expanduser("/Volumes/Botein/microbiome_data/patric_db/")
midas_directory = os.path.expanduser("/Volumes/Botein/microbiome_data/midas_db/")
hmm_data_directory = os.path.expanduser("~/Documents/Research/bgoodlab/microbiome_evolution/cphmm/dat/")

# We use this one to debug because it was the first one we looked at
debug_species_name = 'Bacteroides_uniformis_57318'

good_species_min_coverage = 10
good_species_min_prevalence = 10

min_median_coverage = 20

consensus_lower_threshold = 0.2
consensus_upper_threshold = 0.8
fixation_min_change = consensus_upper_threshold-consensus_lower_threshold
fixation_log10_depth_ratio_threshold = log10(3)

threshold_within_between_fraction = 0.1
threshold_pi = 1e-03

min_opportunities = 100000

modification_difference_threshold = 20
replacement_difference_threshold = 500

twin_modification_difference_threshold = 1000
twin_replacement_difference_threshold = 1000

gainloss_max_absent_copynum = 0.05
gainloss_min_normal_copynum = 0.6
gainloss_max_normal_copynum = 1.2

core_genome_min_copynum = 0.3
core_genome_max_copynum = 3 # BG: should we use a maximum for "core genome"? I'm going to go w/ yes for now
core_genome_min_prevalence = 0.9
shared_genome_min_copynum = 3

# Default parameters for pipe snps
# (Initial filtering for snps, done during postprocessing)
pipe_snps_min_samples=4
pipe_snps_min_nonzero_median_coverage=5
pipe_snps_lower_depth_factor=0.3
pipe_snps_upper_depth_factor=3

parse_snps_min_freq = 0.05

between_host_min_sample_size = 33
between_host_ld_min_sample_size = 10
within_host_min_sample_size = 3
within_host_min_haploid_sample_size = 10

between_low_divergence_threshold = 2e-04

# for close pair analysis
clonal_fraction_cutoff = 0.8
first_pass_block_size = 1000
second_pass_block_size = 10
empirical_histogram_bins = 40
B_vulgatus_data_path = os.path.join(analysis_directory,
                         "closely_related", "debug", "{}_two_clades.pickle".format('Bacteroides_vulgatus_57955'))

# for typical pair analysis
typical_clonal_fraction_cutoff = 0.1

# for within host pipeline
coverage_zscore_cutoff = 2
sample_zscore_cutoff = 2

# color convention
within_host_color = '#57C398'
between_host_color = '#2c7fb8'

# Comment this out
#from parsers.parse_HMP_data import *
# and uncomment this
#from parse_simulated_data import *
# for isolate data
