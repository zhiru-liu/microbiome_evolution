import matplotlib  
matplotlib.use('Agg')
import pylab
import sys
import numpy

from utils import diversity_utils, gene_diversity_utils, stats_utils
from parsers import parse_patric, parse_midas_data

import matplotlib.colors as colors
import matplotlib.cm as cmx
from math import log10,ceil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.random import randint
import pandas
import os.path
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

#inFN=('%s/%s_within_host_gene_changes_A_put_only.txt' % (parse_midas_data.analysis_directory,species_name))
inFN=('%s/%s_within_host_gene_changes_B_unif_only.txt' % (parse_midas_data.analysis_directory, species_name))
inFile=open(inFN,'r')

gene_changes_species_only=[]
for line in inFile:
    gene_changes_species_only.append(line.strip())

inFN=('%s/%s_within_host_gene_changes.txt' % (parse_midas_data.analysis_directory, species_name))
inFile=open(inFN,'r')

outFN=('%s/%s_within_host_gene_changes_database_diffs.txt' % (parse_midas_data.analysis_directory, species_name))
outFile=open(outFN,'w')

gene_changes_full_db=[]
for line in inFile:   
    gene=line.strip()
    gene_changes_full_db.append(gene)
    if gene not in gene_changes_species_only:
        outFile.write(gene + '\tunique_to_full_db_midas\n') 

for gene in gene_changes_species_only:
    if gene not in gene_changes_full_db:
        outFile.write(gene + '\tunique_to_species_only_midas\n')


