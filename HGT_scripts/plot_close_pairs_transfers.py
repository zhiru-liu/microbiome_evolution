import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import warnings
import scipy
import statsmodels.api as sm
from statsmodels.nonparametric import _smoothers_lowess as sml

sys.path.append("..")
import config

def bootstrap(x, y, x_p, bs_frac=0.5):
    lowess = sm.nonparametric.lowess
    samples = np.random.choice(len(y), int(bs_frac*len(y)), replace=True)
    x_bs = x[samples]
    y_bs = y[samples]
    y_p = lowess(y_bs, x_bs, xvals=x_p, return_sorted = True).T
    return y_p


def lowess_summary(x, y, num_plot=100, if_bs=False, bs_frac=0.5, bs_size=200):
    x_plot = np.linspace(np.min(x), np.max(x), num_plot)
    lowess = sm.nonparametric.lowess
    sm_y = lowess(y, x, frac=0.33, xvals=x_plot, return_sorted=True).T

    if if_bs:
        bs_res = np.stack([bootstrap(x, y, x_plot) for i in range(bs_size)]).T
        bs_std = np.nanstd(bs_res, axis=1)
        bs_mean = np.nanmean(bs_res, axis=1)
    else:
        bs_mean = None
        bs_std = None
    return x_plot, sm_y, bs_mean, bs_std


def tricube(x):
    return (1 - x**3)**3


def find_weighted_residue_dist(x, y_res, x_obs, k=25, qs=[0.66]):    
    if len(qs) == 0:
        warnings.warn("No desired quantiles supplied")
        return None
    for q in qs:
        if q < 0 or q > 1:
            raise ValueError("Quantile must be a value in [0, 1]")
    if k <= 0:
        k = 1
    # getting neighboring points and their weights
    lid, rid, radius = sml.update_neighborhood(x.astype(float), x_obs, len(x), 0, k)
    dist = np.abs(x[lid:rid] - x_obs) / radius
    weights = tricube(dist)
    weights = weights / np.sum(weights)

    # generating numerical CDF for residues
    vals = np.abs(y_res[lid:rid])
    order = vals.argsort()
    vals = vals[order]
    weights = weights[order]
    cum_weights = np.cumsum(weights)
    f_int = scipy.interpolate.interp1d(cum_weights, vals, fill_value="extrapolate")
    return f_int(qs)

trend_line = True
data_dir = os.path.join(config.analysis_directory, "closely_related", "second_pass")
for filename in os.listdir(data_dir):
    if filename.startswith('.'):
        continue
    species_name = filename.split('.')[0]
    print("Processing {}".format(species_name))
    df = pd.read_csv(os.path.join(data_dir, filename), index_col=0)
    if df.shape[0] < 10:
        print("Skipping; Only {} pairs".format(df.shape[0]))
        continue

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    # we have different choices of x&y values to plot
    # x: number of clonal snps; or the expected number of total snps
    # y: number of detected transfers; or the total length of transfer regions
    clonal_fraction = 1 - df['transfer_len'] / df['num_total_blocks'].mean()
    exp_snps = df['num_clonal_snps'] / clonal_fraction
    mean_transfer_len = np.nan_to_num(df['transfer_len'] / df['num_transfers'])
    x = exp_snps.to_numpy()
#    y = df['transfer_len'].to_numpy()
    y = df['num_transfers'].to_numpy()

    im = ax.scatter(x, y, s=2, c=clonal_fraction, label=None)
    cbar = plt.colorbar(im)
    cbar.set_label('Clonal fraction', labelpad=10, rotation=270)

    ax.set_title(species_name)
    ax.set_xlabel("Expected clonal snps")
    ax.set_ylabel("Num detected transfers")
    if trend_line:
        x_plot, y_plot, bs_mean, bs_std = lowess_summary(x, y)

        ax.plot(x_plot, y_plot)

        k = int(len(x)*0.1)  # number of neighbor points to choose
        if k >= 7:
            lowess = sm.nonparametric.lowess
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            y_fit = lowess(y, x, return_sorted=False).T
            sigmas = []
            for x_obs in x_plot:
                sigmas.append(find_weighted_residue_dist(x, y_fit - y, x_obs, k=k))
            sigmas = np.array(sigmas).squeeze()
            ax.fill_between(x_plot, y_plot - sigmas,
                            y_plot + sigmas, alpha=0.25)
            df_save = pd.DataFrame(data={'x':x_plot, 'y':y_plot, 'sigma':sigmas})
            df_save.to_csv(os.path.join(config.analysis_directory,
                "closely_related", "third_pass", "{}.csv".format(species_name)))

    fig.tight_layout()
    fig.savefig(os.path.join(config.analysis_directory,
                             "closely_related", "wall_clock_v2.2",
                             "{}.pdf".format(species_name)), dpi=300)
    plt.close()
