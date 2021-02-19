import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import sys
sys.path.append("..")
import config


def plot_for_one_species(species_name, checkpoint_dir, fig_dir, plot_same_clade, annotate=True):
    div_dir = os.path.join(
        checkpoint_dir, "between_hosts", "{}.csv".format(species_name))

    fig_path = os.path.join(fig_dir, species_name + '.pdf')

    div_mat = np.loadtxt(div_dir, delimiter=',')
    div_dist = div_mat[np.triu_indices(div_mat.shape[0])]
    if annotate:
        cutoff_dict = json.load(open('./same_clade_div_cutoffs.json', 'r'))
        if species_name in cutoff_dict:
            cutoffs = cutoff_dict[species_name]
        else:
            cutoffs = (None, None)
        plt.ion()
        while True:
            plt.figure()
            ax = sns.distplot(div_dist, bins=100, kde=False)
            ax.set_title(species_name)
            plt.show()
            lower = cutoffs[0] or min(div_dist)  # hacky way of assigning value to None
            upper = cutoffs[1] or max(div_dist)
            ax.axvspan(lower, upper, alpha=0.1, color='red', label='Same clade')
            plt.legend()
            try:
                if_okay = input("Cutoffs okay? Enter True to continue")
                if if_okay is True:
                    break
                else:
                    cutoffs = input("Please enter cutoffs as lower,upper: ")
            except NameError:
                print("Please enter True or False as an python expression")
            plt.close()
        cutoff_dict[species_name] = cutoffs
        json.dump(cutoff_dict, open('./same_clade_div_cutoffs.json', 'w'))
        plt.ioff()

        # save the limits for easy compare with within host
        xlim_dict = json.load(open('./div_plot_limits.json', 'r'))
        xlim_dict[species_name] = ax.get_xlim()
        json.dump(xlim_dict, open('./div_plot_limits.json', 'w'))
    else:
        plt.figure()
        ax = sns.distplot(div_dist, bins=100, kde=False)
        if plot_same_clade:
            if len(div_dist) != 0:
                cutoff_dict = json.load(open('./same_clade_div_cutoffs.json', 'r'))
                cutoffs = cutoff_dict[species_name]
                lower = cutoffs[0] or min(div_dist) # hacky way of assigning value to None
                upper = cutoffs[1] or max(div_dist)
                ax.axvspan(lower, upper, alpha=0.1, color='red', label='Same clade')
                plt.legend()
    ax.set_xlabel("Synonymous SNP divergence")
    ax.set_ylabel("Bin counts")
    plt.savefig(fig_path)
    plt.close()


def plot_for_one_species_both(species_name, checkpoint_dir, fig_dir):
    fig_path = os.path.join(fig_dir, species_name + '.pdf')

    div_dir = os.path.join(
        checkpoint_dir, "between_hosts", "{}.csv".format(species_name))
    div_mat = np.loadtxt(div_dir, delimiter=',')
    between_dist = div_mat[np.triu_indices(div_mat.shape[0])]

    div_dir = os.path.join(
        checkpoint_dir, "within_hosts", "{}.csv".format(species_name))
    within_dist = np.loadtxt(div_dir, delimiter=',')

    cp = sns.color_palette()
    fig, host = plt.subplots()
    host.set_xlabel("Pairwise synonymous divergence")
    host.set_ylabel("between counts")
    par = host.twinx()
    par.set_ylabel("within counts")
    sns.distplot(between_dist, bins=100, kde=False, ax=host, label="between", color=cp[0])
    sns.distplot(within_dist, bins=100, kde=False, ax=par, color=cp[1], label="within")
    host.figure.legend()
    fig.savefig(fig_path)
    plt.close()


between_host = False
checkpoint_dir = os.path.join(config.analysis_directory, 'pairwise_divergence')
if between_host:
    fig_dir = os.path.join(config.analysis_directory, 'divergence_distributions',
                           'synonymous_divergences', 'between')
else:
    # Plotting both
    fig_dir = os.path.join(config.analysis_directory, 'divergence_distributions',
                           'synonymous_divergences', 'both')

for filename in os.listdir(os.path.join(checkpoint_dir, "between_hosts")):
    if filename.startswith('.'):
        continue
    species_name = filename.split('.')[0]
    if between_host:
        plot_for_one_species(species_name, checkpoint_dir, fig_dir, plot_same_clade=True, annotate=True)
    else:
        plot_for_one_species_both(species_name, checkpoint_dir, fig_dir)
