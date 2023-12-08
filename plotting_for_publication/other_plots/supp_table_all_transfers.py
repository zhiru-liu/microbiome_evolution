import os
import json
import numpy as np
import pandas as pd
import config
from utils import snp_data_utils

fig3_species = json.load(open(os.path.join(config.plotting_intermediate_directory, 'fig3_species.json'), 'r'))
species_cutoff_dict = json.load(open(os.path.join(config.plotting_intermediate_directory, 'clonal_div_cutoff.json'), 'r'))
species_cutoff_dict['Bacteroides_vulgatus_57955'] = config.Bv_clonal_div_cutoff

dfs = []
for species_name in fig3_species:
    transfer_df_path = os.path.join(config.analysis_directory, "closely_related", 'third_pass',
                                    species_name + '_all_transfers.pickle')
    if not os.path.exists(transfer_df_path):
        raise RuntimeError("Please run stage 3a scripts to obtain full genome length and divergences")
    transfer_df = pd.read_pickle(transfer_df_path)
    # transfer_df = transfer_df[transfer_df['clonal fraction >80%']]  # TODO: should not filter CF, but report all; instead, add a "shown in fig3?" column

    good_df = transfer_df
    sample_mask, sample_names = snp_data_utils.get_QP_sample_mask(species_name)
    good_samples = sample_names[sample_mask]
    good_df['Species name'] = species_name
    good_df['Sample 1'] = [good_samples[pair[0]] for pair in good_df['pairs']]
    good_df['Sample 2'] = [good_samples[pair[1]] for pair in good_df['pairs']]
    if ('vulgatus' in species_name) or ('shahii' in species_name):
        good_df['between clade?'] = [['N', 'Y'][i] for i in good_df['types']]
    else:
        good_df['between clade?'] = 'NA'

    clonal_div_cutoff = species_cutoff_dict.get(species_name, 1)
    if clonal_div_cutoff is None:
        clonal_div_cutoff = 1
    good_df['in fig3'] = (good_df['clonal divergence'] <= clonal_div_cutoff) & (good_df['clonal fraction'] >= config.clonal_fraction_cutoff)
    good_df.drop(columns=['starts', 'ends', 'types', 'lengths', 'pairs'], inplace=True)
    df_to_save = good_df[
        ['Species name', 'Sample 1', 'Sample 2', 'clonal divergence', 'clonal fraction', 'between clade?', 'in fig3',
         'synonymous divergences', 'divergences', 'core genome starts', 'core genome ends', 'transfer lengths (core genome)',
         'contigs', 'reference genome starts', 'reference genome ends']]
    df_to_save.columns = ['Species name', 'Sample 1', 'Sample 2', 'Clonal divergence', 'Clonal fraction', 'between clade?', 'Shown in Fig3?',
                          'Transfer divergence (synonymous)', 'Transfer divergence',
                          'Core genome start loc', 'Core genome end loc', 'Transfer length (# covered sites on core genome)',
                          'Reference contig', 'Reference genome start loc', 'Reference genome end loc']
    df_to_save.to_csv(os.path.join(config.figure_directory, 'supp_table', 'all_transfers', '{}.csv'.format(species_name)))
    dfs.append(df_to_save)

big_df = pd.concat(dfs)
# make sure there is no mixture of str and numbers for sample names
big_df['Sample 1'] = big_df['Sample 1'].astype(str)
big_df['Sample 2'] = big_df['Sample 2'].astype(str)

# now adding dedup information
dfs = []
for species in fig3_species:
    save_path = os.path.join(config.analysis_directory, "misc", "dedup", species, "unique_events.csv")
    dedup_events = pd.read_csv(save_path)
    dfs.append(dedup_events)
big_unique_df = pd.concat(dfs)
big_unique_df['Sample 1'] = big_unique_df['Sample 1'].astype(str)
big_unique_df['Sample 2'] = big_unique_df['Sample 2'].astype(str)
unique_events = zip(big_unique_df['Species name'], big_unique_df['Sample 1'], big_unique_df['Sample 2'], big_unique_df['Core genome start loc'], big_unique_df['Core genome end loc'])
event_keys = zip(big_df['Species name'], big_df['Sample 1'], big_df['Sample 2'], big_df['Core genome start loc'], big_df['Core genome end loc'])
unique_set = set(unique_events)
if_unique = []
for tup in event_keys:
    if_unique.append(tup in unique_set)

big_df['Potential duplicate of other events?'] = ~np.array(if_unique)

big_df.to_csv(os.path.join(config.figure_directory, 'supp_table', 'all_transfers.csv'))
