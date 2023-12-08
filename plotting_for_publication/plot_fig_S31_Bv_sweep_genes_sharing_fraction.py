''' To use this package, need to use python 3 '''
import dna_features_viewer
from dna_features_viewer import GraphicFeature, GraphicRecord
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
import config

dat = pd.read_csv(os.path.join(config.plotting_intermediate_directory, 'Bv_annotated_genes.csv'))
features = []
dat.fillna('', inplace=True)
for index, row in dat.iterrows():
    start = row['start']
    end = row['end']
    strand = row['strand']
    if 'flux' not in row['label']:
        label = None
    else:
        label = row['label']
#     label = row['label'] if len(row['label'])>0 else None
    mean_sharing = row['mean sharing']  # mean sharing is normalized by the biggest peak
    color = cm.coolwarm(mean_sharing / 0.6)  # scale color by the RND pump region, which is about 60% of the biggest sharing peak
    if mean_sharing < 0:
        # Non core
        color = 'tab:grey'
    features.append(GraphicFeature(start=start, end=end, strand=strand, label=label, color=color))

# roughly the sweep region plus a bit more to the left
zoom_start = 2137208
zoom_end = 2182437

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'

record = GraphicRecord(sequence_length=5163189, features=features)
fig, axes = plt.subplots(1, 1, figsize=(12, 3), dpi=600)
cropped_record = record.crop((zoom_start, zoom_end))
_ = cropped_record.plot(ax=axes)
fig.savefig(os.path.join(config.figure_directory, 'supp', 'S31_supp_Bv_sweep_genes.pdf'), bbox_inches='tight')