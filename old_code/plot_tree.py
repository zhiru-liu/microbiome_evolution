import matplotlib 
matplotlib.use('Agg')
import pylab
import sys
import numpy
from utils import diversity_utils
from parsers import parse_midas_data
import os
species_name=sys.argv[1]   


os.system('python ~/projectBenNandita/print_distance_matrix.py ' + species_name)    

os.system('Rscript ~/projectBenNandita/plot_tree.R ' + species_name)
