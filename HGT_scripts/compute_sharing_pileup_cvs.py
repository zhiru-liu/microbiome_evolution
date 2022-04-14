import numpy as np
import pandas as pd
import os
import json
import config

contig_data = json.load(open(os.path.join(config.data_directory, 'contig_counts.json'), 'r'))
species = []
mean_frac = []
cv = []
for species_name in contig_data:
    if contig_data[species_name] > 50:
        continue
    ckpt_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
    cumu_runs = np.loadtxt(os.path.join(ckpt_path, 'between_host.csv'))
    species.append(species_name)
    mean_frac.append(cumu_runs[:, 0].mean())
    cv.append(np.std(cumu_runs[:, 0]) / np.mean(cumu_runs[:, 0]))

# vulgatus between clade
species_name = 'Bacteroides_vulgatus_57955_between'
ckpt_path = os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical', species_name)
cumu_runs = np.loadtxt(os.path.join(ckpt_path, 'between_host.csv'))
species.append(species_name)
mean_frac.append(cumu_runs[:, 0].mean())
cv.append(np.std(cumu_runs[:, 0]) / np.mean(cumu_runs[:, 0]))

df = pd.DataFrame({'Species name': species, 'Mean sharing fraction': mean_frac, 'CV': cv})
df.to_csv(os.path.join(config.analysis_directory, 'sharing_pileup', 'species_summary.csv'))
