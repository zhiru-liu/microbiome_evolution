import os
import numpy as np
import matplotlib.pyplot as plt
from utils import snp_data_utils
import config


for species_name in os.listdir(os.path.join(config.analysis_directory, 'sharing_pileup', 'empirical')):
    if species_name.startswith('.') or species_name.endswith('zip'):
        continue
    save_path = os.path.join(config.analysis_directory, 'sharing_pileup',
                             'empirical', species_name)
    thresholds = np.loadtxt(os.path.join(save_path, 'between_host_thresholds.txt'))
    cumu_runs = np.loadtxt(os.path.join(save_path, 'between_host.csv'))

    if 'between' in species_name:
        species_name = "Bacteroides_vulgatus_57955"
    print("processing %s" % species_name)
    contig_lengths = snp_data_utils.get_core_genome_contig_lengths(species_name)
    contig_ends = np.cumsum(contig_lengths)

    real_cv = np.std(cumu_runs, axis=0) / np.mean(cumu_runs, axis=0)
    print(real_cv)

    fig, ax = plt.subplots(figsize=(5, 2))
    # idx_to_plot = [0, 1, 2, 3]
    for i in range(cumu_runs.shape[1]):
        dat = cumu_runs[:, i]
        ax.plot(dat, linewidth=1, label="threshold={:.0f}, cv={:.2f}".format(thresholds[i], real_cv[i]))

    for end in contig_ends:
        ax.axvline(x=end, ls='--', c='0.5', linewidth=1, zorder=0)
    if len(contig_ends) > 1:
        ax.axvline(x=-100, ls='--', c='0.5', linewidth=1, label='contig boundary')

    ax.set_ylim([0, 0.35])
    ax.set_xlim([0, cumu_runs.shape[0]])
    ax.set_title(species_name)
    ax.legend(fontsize='small', bbox_to_anchor=(1, 1))
    ax.set_xlabel('4D core genome location')
    ax.set_ylabel('sharing fraction')
    fig.savefig(os.path.join(config.analysis_directory, "sharing_pileup", "plots", species_name + '.pdf'),
                bbox_inches='tight')
    plt.close()
