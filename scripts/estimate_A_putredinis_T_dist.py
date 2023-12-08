import numpy as np
import os
import pandas as pd
from scipy.stats import poisson
import config

"""
Use EM to infer the distribution of true divergence time for A. putredinis, based on the distribution of clonal divergence
and Poisson model of SNV accumulation
"""

def q_is(xi, li, s, Ts, lamb):
    denom = np.sum([poisson.pmf(xi, mu=Ts[s]*li) * lamb[s] for s in range(len(Ts))])
    num = poisson.pmf(xi, mu=Ts[s]*li) * lamb[s]
    return num / denom


def new_est(all_x, all_l, lamb, Ts):
    Qis = np.empty(shape=(len(all_x), len(Ts)))
    for i in xrange(len(all_x)):
        for s in xrange(len(Ts)):
            Qis[i, s] = q_is(all_x[i], all_l[i], s, Ts, lamb)
    return Qis.sum(axis=0) / Qis.sum()

data_dir = os.path.join(config.analysis_directory, "closely_related")
raw_data = pd.read_pickle(os.path.join(data_dir, 'third_pass', 'Alistipes_putredinis_61533.pickle'))

cf = raw_data['clonal fractions']
cf_cutoff = 0.8
rates = raw_data['clonal divs'].to_numpy()[cf>=cf_cutoff]
ls = raw_data['clonal lengths'].to_numpy()[cf>=cf_cutoff]

ls = ls[rates<2e-4]
rates = rates[rates<2e-4]
Xs = (rates*ls).astype(int)

Ts = np.arange(1, 21) * 1e-5

lamb = np.ones(20) / 20.
for i in range(5):
    print(lamb)
    lamb = new_est(Xs, ls, lamb, Ts)

df = pd.DataFrame({'Rates': Ts, 'Weights': lamb})
df.to_csv(os.path.join(config.analysis_directory, 'HMM_validation', 'Ap_rate_weights.csv'))