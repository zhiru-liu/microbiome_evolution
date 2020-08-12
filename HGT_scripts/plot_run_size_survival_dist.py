import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle
import json
import sys
import time
import seaborn as sns
from scipy import stats
sys.path.append("..")
from utils import core_gene_utils, diversity_utils, HGT_utils
import config

'''
    Need to know the cutoffs for same clade snp counts
    Currently, manually saved to a json file
'''

'''
    Use Seaborn color blind pallete
'''
color_list = sns.color_palette(palette='colorblind')
between_color = color_list[0]
within_color = color_list[4]
null_geometric_color = color_list[3]
null_simulated_color = color_list[2]
fit_color = color_list[5]


def _convert_ticks(ticks, conversion, log=False):
    # helper function for secondary axis plotting
    max_new_tick = conversion * max(ticks)
    if log:
        new_tick_labels = np.power(10, np.arange(0, np.log10(max_new_tick), 1))
        new_ticks = new_tick_labels / conversion
    else:
        new_tick_labels = np.arange(0, max_new_tick, 1)
        new_ticks = new_tick_labels / conversion
    return new_ticks, new_tick_labels


def get_passed_site_vector(all_genes, passed_sites_map, id1, id2):
    all_num_sites = np.zeros(len(all_genes))
    for i in xrange(len(all_genes)):
        if all_genes[i] in passed_sites_map:
            all_num_sites[i] = passed_sites_map[all_genes[i]
                                               ]['4D']['sites'][id1, id2]
        else:
            all_num_sites[i] = 0
    return all_num_sites


def get_total_passed_sites(all_genes, passed_sites_map, id1, id2):
    return sum(get_passed_site_vector(all_genes, passed_sites_map, id1, id2))


def prepare_divergence_dict(species_name, passed_sites_map, checkpoint_path, if_between_host):
    # load lots of data
    all_genes = core_gene_utils.get_sorted_core_genes(species_name)
    snp_counts = pickle.load(
        open(os.path.join(checkpoint_path, 'snp_counts_map.pickle'), 'rb'))
    found_samples = pickle.load(
        open(os.path.join(checkpoint_path, 'found_samples.pickle'), 'rb'))
    num_samples = len(found_samples)
    divergence_dict = dict()
    if if_between_host:
        for i in xrange(num_samples):
            for j in xrange(i + 1, num_samples):
                divergence_dict[(i, j)] = snp_counts[(i, j)] / \
                    get_total_passed_sites(all_genes, passed_sites_map, i, j)
    else:
        for i in snp_counts:
            divergence_dict[i] = snp_counts[i] / \
                get_total_passed_sites(all_genes, passed_sites_map, i, i)

    return divergence_dict


def plot_for_one_species(ax, species_name, if_between_host, if_limit=True,
        subset=True, collapse=True, normalization=True):
    # loading data
    base_dir = 'between_hosts_checkpoints' if if_between_host else 'within_hosts_checkpoints'
    checkpoint_path = os.path.join(
        config.analysis_directory, base_dir, species_name)

    if not os.path.exists(checkpoint_path):
        print('No data found for {}'.format(species_name))
        return

    all_runs_dict = pickle.load(
        open(os.path.join(checkpoint_path, 'all_runs_map.pickle'), 'rb'))
    snp_counts = pickle.load(
        open(os.path.join(checkpoint_path, 'snp_counts_map.pickle'), 'rb'))
    passed_sites_map = pickle.load(
        open(os.path.join(checkpoint_path, 'passed_sites_map.pickle'), 'rb'))

    if len(snp_counts.values()) == 0:
        return

    divergence_dict = prepare_divergence_dict(
            species_name, passed_sites_map, checkpoint_path, if_between_host)
    # load same clade snp cutoff
    # TODO eventually want to use only divergence
    cutoffs = json.load(open('./same_clade_snp_cutoffs.json', 'r'))
    lower_cutoff = cutoffs[species_name][0] or min(
        snp_counts.values())  # hacky way of assigning value to None
    upper_cutoff = cutoffs[species_name][1] or max(snp_counts.values())

    # since we loaded snps_counts here, we'll compute the average # snps of this species
    all_snp_counts = np.array(snp_counts.values())
    average_snp_count = np.mean(
            all_snp_counts[(all_snp_counts < upper_cutoff) & (all_snp_counts > lower_cutoff)])

    print("Finish loading for {}".format(species_name))
    print("{} has {} snps on average".format(species_name, average_snp_count))

    ax = plt.gca()
    ax.set_yscale('log')

    ax.set_xlabel('Normalized site counts')
    ax.set_ylabel('Survival Probability')
    if if_limit:
        ax.set_xlim((0, 20))

    if subset:
        all_pairs = all_runs_dict.keys()
        num_to_plot = min(len(all_pairs), 1000)
        pairs = random.sample(all_pairs, num_to_plot)
    else:
        pairs = all_runs_dict.keys()
    
    num_runs = []
    data_for_fit = []
    num_curves = len(pairs)
    for pair in pairs:
        div = divergence_dict[pair]
        if (snp_counts[pair] < lower_cutoff) or (snp_counts[pair] > upper_cutoff):
            continue
        if not collapse:
            div = 0.01 # use wrong collapse
        if not if_between_host:
            # choose appropriate alpha
            plot_color = within_color
        else:
            plot_color = between_color
        # normalize by multiplying div
        data = all_runs_dict[pair][3] * div
        plot_range = (0, max(data))
        plot_alpha = min(20./num_curves, 1)
        num_runs.append(len(all_runs_dict[pair][0]))
        histo = ax.hist(data, range=plot_range, normed=normalization, cumulative=-1, bins=50, histtype='step', log=True,
                    color=plot_color, alpha=plot_alpha)
        data_for_fit.append(histo)
    return average_snp_count, int(np.nan_to_num(np.mean(num_runs))), passed_sites_map, data_for_fit


def plot_only_between_or_within():
    t0 = time.time()
    if_between_host = False
    subset = False
    collapse = True
    base_dir = 'between_hosts_checkpoints' if if_between_host else 'within_hosts_checkpoints'

    for species_name in os.listdir(os.path.join(config.analysis_directory, base_dir)):
        if species_name.startswith('.'):
            continue

        print("plotting for {} at {} min".format(species_name, (time.time()-t0)/60))

        fig = plt.figure()
        fig.set_size_inches(8, 6)
        ax = fig.gca()

        plot_for_one_species(
            ax, species_name, if_between_host, subset=subset, collapse=collapse)

        if if_between_host:
            fig_base_path = os.path.join(
                config.analysis_directory, 'run_size_survival_distributions', 'between')
        else:
            fig_base_path = os.path.join(
                config.analysis_directory, 'run_size_survival_distributions', 'within')

        if not collapse:
            fig_base_path = os.path.join(fig_base_path, 'no_collapse')

        if subset:
            fig_path = os.path.join(
                fig_base_path, '{}_subset.pdf'.format(species_name))
        else:
            fig_path = os.path.join(fig_base_path, '{}.pdf'.format(species_name))

        plt.savefig(fig_path)
        plt.close()

def plot_geometric_null(ax, normalization=False, num_runs=1000, num_reps=1000):
    for i in range(num_reps):
        null = np.random.geometric(p=0.01, size=num_runs)
        _ = ax.hist(null * 0.01, normed=normalization, cumulative=-1, bins=50,
                    histtype='step', color=null_geometric_color, alpha=20./num_reps)


def plot_simulated_null(ax, species_name, all_genes, passed_sites_map, normalization=False, num_simulations=100, num_reps=5):
    num_samples = passed_sites_map[passed_sites_map.keys()[0]]['4D']['sites'].shape[0]
    sample_ids = [i for i in range(num_samples)]
    
    for i in range(num_simulations):
        pair = random.sample(sample_ids, 2)
        if pair[1] < pair[0]:
            pair = [pair[1], pair[0]]
        passed_sites_vec = get_passed_site_vector(all_genes, passed_sites_map, pair[0], pair[1])
        passed_sites_vec = passed_sites_vec.astype(int)
        for i in range(num_reps):
            simulated_snps = np.random.binomial(passed_sites_vec, 0.01)
            runs, starts, ends = HGT_utils.find_runs(simulated_snps) 
            site_counts = np.array([sum(passed_sites_vec[start:end+1]) for (start, end) in zip(starts, ends)])
            div = float(sum(simulated_snps)) / sum(passed_sites_vec)
            data = div * site_counts
            bin_range = (0, max(data))
            _ = ax.hist(data, normed=normalization, cumulative=-1, bins=50, range=bin_range,
                        histtype='step', color=null_simulated_color, alpha=20./(num_reps*num_simulations))


def plot_linear_fit(ax, data_for_fit):
    ys = []
    xs = []
    for dat in data_for_fit:
        arr = np.array([(dat[1][i] + dat[1][i+1])/2 for i in range(len(dat[1])-1)])
        ys.append(np.array(dat[0])[arr < 6]) # TODO: 6 is a eyeballed cutoff
        xs.append(arr[arr < 6])
    xs = [x for l in xs for x in l]
    ys = [y for l in ys for y in l]
    ys = np.log(ys)
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    x_plot = np.linspace(0, 15, 100)
    ax.plot(x_plot, np.exp(x_plot * slope + intercept), label='Slope = %.2f' % slope, color=fit_color)
    return slope, intercept, r_value, p_value, std_err


def plot_both():
    t0 = time.time()
    base_dir = 'between_hosts_checkpoints'
    fig_base_path = os.path.join(
        config.analysis_directory, 'run_size_survival_distributions', 'test')
    slope_file = open(os.path.join(fig_base_path, 'slopes.csv'), 'w')

    for species_name in os.listdir(os.path.join(config.analysis_directory, base_dir)):
        if species_name.startswith('.'):
            continue

        print("plotting for {} at {} min".format(species_name, (time.time()-t0)/60))

        fig = plt.figure()
        fig.set_size_inches(8, 6)
        ax = fig.gca()

        average_snp, average_num_runs, passed_sites, data_for_fit = plot_for_one_species(
            ax, species_name, if_between_host=True, subset=True, normalization=True)

        plot_for_one_species(
            ax, species_name, if_between_host=False, subset=False, normalization=True)

        # now add a secondary axis for genome distance
        gene_len_dict = core_gene_utils.parse_gene_lengths(species_name)
        all_genes = core_gene_utils.get_sorted_core_genes(species_name)
        core_gene_lengths = [gene_len_dict[gene][0] for gene in all_genes]
        distance_between_snps = sum(core_gene_lengths) / average_snp
        new_ticks, new_tick_labels = _convert_ticks(ax.get_xticks(), distance_between_snps/1000)

        plot_geometric_null(ax, num_runs=average_num_runs, normalization=True)

        plot_simulated_null(ax, species_name, all_genes, passed_sites, normalization=True)

        fitted_params = plot_linear_fit(ax, data_for_fit)
        slope_file.write("{},{}\n".format(species_name, fitted_params[0]))

        # additional x axis
        ax2 = ax.twiny()
        ax2.set_xticks(new_ticks)
        ax2.set_xlim(ax.get_xlim())
        _ = ax2.set_xticklabels(new_tick_labels)
        ax2.set_xlabel('Expected genome distance (kbp)')
        
        # additional y axis
        ax.set_ylim([0.0005, 1.2])
        new_yticks, new_ytick_labels = _convert_ticks(ax.get_yticks(), average_num_runs, log=True)
        ax3 = ax.twinx()
        ax3.set_yscale('log')
        ax3.set_yticks(new_yticks)
        ax3.set_ylim(ax.get_ylim())
        _ = ax3.set_yticklabels(new_ytick_labels)
        ax3.set_ylabel('Expected counts')
        ax3.minorticks_off()

        fig_path = os.path.join(fig_base_path, '{}.pdf'.format(species_name))

        # set labels
        ax.plot([], [], color=null_geometric_color, label='Null (Geometric)')
        ax.plot([], [], color=null_simulated_color, label='Null (Simulated)')
        ax.plot([], [], color=between_color, label='Data (Between)')
        ax.plot([], [], color=within_color, label='Data (Within)')
        ax.legend()


        plt.savefig(fig_path)
        plt.close()
    slope_file.close()

plot_both()
