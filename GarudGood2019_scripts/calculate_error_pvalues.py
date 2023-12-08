import sys
import os
import bz2
from parsers import parse_midas_data

if len(sys.argv) > 1:
    species_name=sys.argv[1]
else:
    species_name= parse_midas_data.debug_species_name

sys.stderr.write("Calculating pvalues for %s...\n" % species_name)

output_filename = "%ssnps/%s/annotated_snps.txt.bz2" % (parse_midas_data.data_directory, species_name)

os.system('python %spipe_midas_data.py %s | %sannotate_pvalue --disabled | bzip2 -c > %s' % (
    parse_midas_data.metadata_directory, species_name, parse_midas_data.metadata_directory, output_filename))
 
