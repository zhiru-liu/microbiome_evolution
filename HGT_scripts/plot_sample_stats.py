import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
sys.path.append("..")
from utils import parallel_utils
import config


fig, haploid_axis = plt.subplots(figsize=(1.5, 3))
fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

haploid_color = '#08519c'
light_haploid_color = '#6699CC'
good_witin_color = '#ef8a62'

sample_df = parallel_utils.compute_good_sample_stats()
sample_df = sample_df.sort_values('num_high_coverage_samples', ascending=False)
num_qp_samples = sample_df['num_qp_samples']
num_samples = sample_df['num_high_coverage_samples']
num_within_samples = sample_df['num_good_within_samples']

ys = 0-np.arange(0,len(num_qp_samples))

haploid_axis.barh(ys+0.5, num_qp_samples,color=haploid_color,linewidth=0,label='QP',zorder=1)
haploid_axis.barh(ys+0.5, num_samples,color=light_haploid_color,linewidth=0,label='hard non-QP',zorder=0)
haploid_axis.barh(ys+0.5, num_within_samples,left=num_qp_samples,color=good_witin_color,linewidth=0,label='simple non-QP')
haploid_axis.set_xlim([0,800])
haploid_axis.set_xticks([0,200,400,600,800])

haploid_axis.yaxis.tick_right()
haploid_axis.xaxis.tick_bottom()

haploid_axis.set_yticks(ys+0.5)
haploid_axis.set_yticklabels(sample_df['species_name'],fontsize=4)
haploid_axis.set_ylim([-1*len(num_qp_samples)+0.5,1.5])

haploid_axis.tick_params(axis='y', direction='out',length=3,pad=1)

haploid_axis.legend(loc='lower right',frameon=False)

fig.savefig('test_sample_stats.pdf', bbox_inches='tight')