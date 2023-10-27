import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import config
import json

TcTm_df = pd.read_csv(os.path.join(config.figure_directory, 'supp_table', 'TcTm_estimation.csv'))
alpha_dict = json.load(open(os.path.join(config.analysis_directory, 'misc', 'partial_recomb_alpha.json'), 'r'))

TcTm_df['alpha'] = TcTm_df['Species'].apply(alpha_dict.get)

mpl.rcParams['font.size'] = 5
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'
plt.figure(figsize=(3, 2), dpi=300)

mpl.rcParams['font.size'] = 5
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'small'
plt.figure(figsize=(3, 2), dpi=300)

xs = np.linspace(0, 100)
plt.plot(xs, xs, color='k', linestyle='--')

Tcs = TcTm_df['d(Tc)'].copy()
xvals = Tcs / TcTm_df['d(Tm)']
yvals = 1000 * Tcs / (1 / TcTm_df['alpha'] - 1)
r2 = np.corrcoef(xvals, yvals)[0, 1]
plt.plot(xvals, yvals, '.')

# quick script to compute correlation p val
rs = []
for i in range(10000):
    np.random.shuffle(Tcs)

    xvals = Tcs / TcTm_df['d(Tm)']
    yvals = 1000 * Tcs / (1 / TcTm_df['alpha'] - 1)
    rs.append(np.corrcoef(xvals, yvals)[0, 1])
rs = np.array(rs)
p_val = np.sum(rs > r2) / 10000.

# plt.plot(1/TcTm_df['d(Tm)'], 1 /(1/TcTm_df['alpha'] - 1), '.')

plt.text(0, 85, '$R^2={:.2f}$\np-value = {:.3f}'.format(r2, p_val), fontsize=7, color='k')

plt.xlabel(r'$\frac{T_{mrca}}{T_{mosaic}}$, proxy for $r/m$')
plt.ylabel(r'$\frac{\ell \cdot \theta}{1/\alpha - 1} \approx \frac{r}{m}$')
plt.savefig(os.path.join(config.figure_directory, 'supp', 'supp_rbym_correlation.pdf'), bbox_inches='tight')