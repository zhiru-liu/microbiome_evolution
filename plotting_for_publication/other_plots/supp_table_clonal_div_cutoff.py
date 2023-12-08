import json
import numpy as np
import os
import config
import pandas as pd

species_cutoff_dict = json.load(open(os.path.join(config.plotting_intermediate_directory, 'clonal_div_cutoff.json'), 'r'))
species_cutoff_dict['Bacteroides_vulgatus_57955'] = config.Bv_clonal_div_cutoff

all_species, all_cutoffs = [], []
for species in species_cutoff_dict:
    if species_cutoff_dict[species] is not None:
        all_species.append(species)
        all_cutoffs.append(species_cutoff_dict[species])
df = pd.DataFrame(data={'Species': all_species, 'Clonal divergence cutoff': all_cutoffs})
df = df[['Species', 'Clonal divergence cutoff']]
df.to_csv(os.path.join(config.figure_directory, 'supp_table', 'cd_cutoffs.csv'))
