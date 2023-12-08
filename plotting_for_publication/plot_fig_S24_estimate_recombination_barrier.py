import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import random
import config

from utils import figure_utils
from sklearn.linear_model import LogisticRegression

def load_transfer_data(species_name):
    save_path = os.path.join(config.analysis_directory,
                                 "closely_related", "third_pass", "{}_all_transfers.pickle".format(species_name))
    run_df = pd.read_pickle(save_path)
    data_dir = os.path.join(config.analysis_directory, "closely_related")
    raw_df = pd.read_pickle(os.path.join(data_dir, 'third_pass', species_name + '.pickle'))

    cf_cutoff = config.clonal_fraction_cutoff
    good_pairs = raw_df[raw_df['clonal fractions'] > cf_cutoff]['pairs']
    mask = run_df['pairs'].isin(good_pairs)
    full_df = run_df[mask]

    # sim_transfers = np.loadtxt(os.path.join(
    #     config.analysis_directory, 'closely_related', 'simulated_transfers', species_name+'.csv'))
    sim_transfers = np.loadtxt(os.path.join(
        config.analysis_directory, 'closely_related', 'simulated_transfers_cphmm', species_name+'.csv'))
    sim_transfers = sim_transfers[~np.isnan(sim_transfers)]
    obs_transfers = full_df['synonymous divergences']
    return sim_transfers, obs_transfers


def fit_logistic(species_name):
    sim, obs = load_transfer_data(species_name)
    # for dropping the noisy tail end of data
    rank = int(0.01 * len(sim))
    upper_cutoff = sim[np.argsort(sim)[-rank]]
    lower_cutoff = sim[np.argsort(sim)[rank]]

    # make regression data
    sim_reps = np.repeat(sim, 20)
    X = np.hstack([sim_reps, obs])
    mask = (X < upper_cutoff) & (X > lower_cutoff)
    y = np.hstack([np.zeros(len(sim_reps)), np.ones(len(obs))])
    #     print(div_cutoff)

    clf = LogisticRegression(random_state=42, penalty='l1').fit(X[mask].reshape(-1, 1), y[mask])
    beta1, beta0 = clf.coef_.squeeze(), clf.intercept_.squeeze()

    # finding the divergence of simulated transfer at 25% quantile
    rank = int(0.25 * len(sim))
    upper_cutoff = sim[np.argsort(sim)[-rank]]
    lower_cutoff = sim[np.argsort(sim)[rank]]
    return beta0, beta1, upper_cutoff - lower_cutoff

files_to_plot = os.listdir(os.path.join(config.analysis_directory, 'closely_related', 'simulated_transfers'))
files_to_plot = list(filter(lambda x: not x.startswith('.'), files_to_plot))

fit_params = {}
for i in range(len(files_to_plot)):
    species_name = files_to_plot[i].split('.')[0]
    if 'Lachnospiraceae' in species_name:
        continue
    fit_params[species_name] = fit_logistic(species_name)

ks_df = pd.read_csv(os.path.join(config.plotting_intermediate_directory, 'transfer_distribution_ks_test.csv'), index_col=0)
ks_df.columns = ['Species name', 'ks stat', 'p val']
ks_df.set_index('Species name', inplace=True)
ks_df = ks_df.sort_values(by='ks stat', ascending=True)

ks_df['beta1'] = map(lambda x: fit_params[x][1], ks_df.index)
ks_df['delta div'] = map(lambda x: fit_params[x][2], ks_df.index)

fig, axbottom = plt.subplots(1, 1, figsize=(5, 1.8), dpi=300)

xs = np.arange(ks_df.shape[0])
ys = np.exp(ks_df['beta1'].astype(float) * ks_df['delta div'].astype(float))
ks_df['IQR_reduction'] = ys
ks_df.to_csv(os.path.join(config.figure_data_directory, 'figS24', 'recombination_barrier_estimates.csv'))
axbottom.bar(xs, ys, linewidth=0.7, facecolor='grey', edgecolor='k', fill=True, label=None)
axbottom.axhline(1, color='k', linestyle='--')

axbottom.set_xticks(xs)
axbottom.set_xlim([-1, xs.max()+1])
axbottom.spines['top'].set_visible(False)
axbottom.spines['right'].set_visible(False)
species_names = map(lambda x: figure_utils.get_pretty_species_name(x, manual=True), ks_df.index.to_numpy())
axbottom.set_xticklabels(species_names,fontsize=5, rotation = 90)
axbottom.set_xlim(xmin=xs[0]-1, xmax=xs[-1]+1)
axbottom.set_ylabel('IQR recombination rate reduction', fontsize=6)

fig.savefig(os.path.join(config.figure_directory, 'supp', 'S24_supp_estimate_recombination_barrier.pdf'), bbox_inches='tight')