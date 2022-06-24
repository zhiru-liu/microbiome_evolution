#!/usr/bin/env python3
# awkwardly need to use python3 for this single file because of the package statsmodels!

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
import utils.close_pair_utils

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
        bs_res = np.stack([bootstrap(x, y, x_plot, bs_frac=bs_frac) for i in range(bs_size)]).T
        bs_std = np.nanstd(bs_res, axis=1)
        bs_mean = np.nanmean(bs_res, axis=1)
    else:
        bs_mean = None
        bs_std = None
    return x_plot, sm_y, bs_mean, bs_std


def tricube(x):
    return (1 - x**3)**3


def find_weighted_residual_dist(x, y_res, x_obs, k=25, qs=[0.66]):
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

    # generating numerical CDF for residuals
    vals = np.abs(y_res[lid:rid])
    order = vals.argsort()
    vals = vals[order]
    weights = weights[order]
    cum_weights = np.cumsum(weights)
    f_int = scipy.interpolate.interp1d(cum_weights, vals, fill_value="extrapolate")
    return f_int(qs)


def prepare_trend_line(x, y):
    x_plot, y_plot, bs_mean, bs_std = lowess_summary(x, y)

    k = int(len(x)*0.2)  # number of neighbor points to choose
    if k >= 7:
        lowess = sm.nonparametric.lowess
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        y_fit = lowess(y, x, return_sorted=False).T
        sigmas = []
        for x_obs in x_plot:
            sigmas.append(find_weighted_residual_dist(x, y_fit - y, x_obs, k=k))
        sigmas = np.array(sigmas).squeeze()
    else:
        sigmas = None
    return x_plot, y_plot, sigmas


if __name__ == "__main__":
    trend_line = True
    colored = True  # whether to use clonal fraction for color code
    if_counts = False
    second_pass_dir = os.path.join(config.analysis_directory, "closely_related", "second_pass")
    data_dir = os.path.join(config.analysis_directory, "closely_related", "third_pass")

    for filename in os.listdir(second_pass_dir):
        if filename.startswith('.'):
            continue
        species_name = filename.split('.')[0]
        print("Processing {}".format(species_name))
        filepath = os.path.join(data_dir, "%s.pickle" % species_name)
        if not os.path.exists(filepath):
            print("Intermediate file not found for {}, skipping".format(species_name))
            continue
        df = pd.read_pickle(filepath)
        if df.shape[0] < 10:
            print("Skipping; Only {} pairs".format(df.shape[0]))
            continue

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        # we have different choices of x&y values to plot
        # x: number of clonal snps; or the expected number of total snps; or the clonal divergence
        # y: number of detected transfers; or the total length of transfer regions
        x = df['clonal divs'].to_numpy()
        div_cutoff = 3.e-4
        cf = df['clonal fractions'][x < div_cutoff]
        if if_counts:
            y = df['normalized transfer counts'].to_numpy()
            y = y[x < div_cutoff]
        else:
            y = 1 - cf
        x = x[x < div_cutoff]

        if colored:
            im = ax.scatter(x, y, s=2, c=cf, label=None)
            cbar = plt.colorbar(im)
            cbar.set_label('Clonal fraction', labelpad=10, rotation=270)
        else:
            im = ax.scatter(x, y, s=2, label=None)

        ax.set_title(species_name)
        ax.set_xlabel("Expected clonal snps")
        if if_counts:
            ax.set_ylabel("Num detected transfers per 1Mbps")
        else:
            ax.set_ylabel("Recombined fraction")
        if trend_line:
            cf = df['clonal fractions']
            # only using the sufficiently close pairs to fit trend line
            x_fit = df['clonal divs'].to_numpy()
            if if_counts:
                y_fit = df['normalized transfer counts'].to_numpy()
            else:
                y_fit = 1 - df['clonal fractions'].to_numpy()

            cf_cutoff = config.clonal_fraction_cutoff
            x_fit = x_fit[cf >= cf_cutoff]
            y_fit = y_fit[cf >= cf_cutoff]
            x_plot, y_plot, sigmas = prepare_trend_line(x_fit, y_fit)

            ax.plot(x_plot, y_plot)
            if sigmas is not None:
                ax.fill_between(x_plot, y_plot - sigmas,
                                y_plot + sigmas, alpha=0.25)
                df_save = pd.DataFrame(data={'x':x_plot, 'y':y_plot, 'sigma':sigmas})
                if if_counts:
                    df_save.to_csv(os.path.join(config.analysis_directory,
                        "closely_related", "fourth_pass", "{}.csv".format(species_name)))
                else:
                    df_save.to_csv(os.path.join(config.analysis_directory,
                        "closely_related", "fourth_pass", "{}_length.csv".format(species_name)))
        ax.set_xlim([0, div_cutoff])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        fig.tight_layout()
        if if_counts:
            fig.savefig(os.path.join(config.analysis_directory,
                                     "closely_related", "wall_clock_v6",
                                     "{}.pdf".format(species_name)), dpi=300)
        else:
            fig.savefig(os.path.join(config.analysis_directory,
                                     "closely_related", "wall_clock_v6_length",
                                     "{}.pdf".format(species_name)), dpi=300)
        plt.close()
