import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
import sys
sys.path.append("..")
import config


def plot_for_one_species(species_name, plot_same_clade):
    checkpoint_path = os.path.join(
        config.analysis_directory, 'between_hosts_checkpoints', species_name, 'snp_counts_map.pickle')
    fig_path = os.path.join(config.analysis_directory,
                            'divergence_distributions', species_name + '.pdf')

    snp_counts = pickle.load(open(checkpoint_path, 'rb'))
    plt.figure()
    counts = snp_counts.values()
    ax = sns.distplot(counts, bins=100, kde=False)
    if plot_same_clade:
        cutoff_dict = json.load(open('./same_clade_snp_cutoffs.json', 'r'))
        cutoffs = cutoff_dict[species_name]
        lower = cutoffs[0] or min(counts) # hacky way of assigning value to None
        upper = cutoffs[1] or max(counts)
        ax.axvspan(lower, upper, alpha=0.1, color='red', label='Same clade')
        plt.legend()
    ax.set_xlabel("Synonymous SNP counts")
    ax.set_ylabel("Bin counts")
    plt.savefig(fig_path)

plot_same_clade = True
for species_name in os.listdir(os.path.join(config.analysis_directory, 'between_hosts_checkpoints')):
    if species_name.startswith('.'):
        continue
    plot_for_one_species(species_name, plot_same_clade)
