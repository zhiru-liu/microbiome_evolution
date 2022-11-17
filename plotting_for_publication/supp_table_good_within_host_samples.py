import os
import numpy as np
import pandas as pd
import config


max_run_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs')
plot_dir = os.path.join(config.analysis_directory, 'typical_pairs', 'max_runs_plots')
files = [
    'Alistipes_putredinis_61533',
    'Alistipes_shahii_62199',
    'Bacteroides_stercoris_56735',
    'Bacteroides_thetaiotaomicron_56941',
    'Bacteroides_vulgatus_57955_diff_clade',
    'Bacteroides_vulgatus_57955_same_clade',
    'Eubacterium_rectale_56927',
    'Parabacteroides_distasonis_56985',
    'Parabacteroides_merdae_56972'
]
all_species = []
all_nums = []
for species_name in files:
    within_host_max_runs = np.loadtxt(os.path.join(max_run_dir, species_name + '_within.txt'), ndmin=1)
    all_nums.append(len(within_host_max_runs))
    items = species_name.split('_')
    species = '_'.join(items[:3])
    if 'vulgatus' in species_name:
        species += ' ({} {})'.format(items[-2], items[-1])
    all_species.append(species)
df = pd.DataFrame(data={'Species': all_species, 'High-quality dual-colonization samples': all_nums})
df.sort_values(by=['Species'], inplace=True)
df = df[['Species', 'High-quality dual-colonization samples']]
df.to_csv(os.path.join(config.figure_directory, 'supp_table', 'high_quality_within_host_samples.csv'))
