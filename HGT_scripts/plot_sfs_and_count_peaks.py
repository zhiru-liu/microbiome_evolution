import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("..")
import config
from utils import sfs_utils, diversity_utils, HGT_utils
from parsers import parse_midas_data


def process_one_species(species_name):
    if os.path.exists(os.path.join(config.analysis_directory, 'allele_freq', species_name)):
        print("{} already processed".format(species_name))
        return

    samples, sfs_map = parse_midas_data.parse_within_sample_sfs(
        species_name, allowed_variant_types=set(['4D']))
    highcoverage_samples = list(
        diversity_utils.calculate_highcoverage_samples(species_name))

    for sample in highcoverage_samples:
        all_fs, all_pfs = sfs_utils.calculate_binned_sfs_from_sfs_map(
            sfs_map[sample], folding='major')
        df = all_fs[1] - all_fs[0]
        # For peak finding, only use the polymorphic sites
        pfs = all_pfs[all_fs < 0.95]
        fs = all_fs[all_fs < 0.95]

        # Find the max peak size
        within_sites, between_sites, total_sites = sfs_utils.calculate_polymorphism_rates_from_sfs_map(
            sfs_map[sample])
        between_line = between_sites*1.0 / \
            total_sites/((fs > 0.2)*(fs < 0.5)).sum()
        pmax = np.max([pfs[(fs > 0.1)*(fs < 0.95)].max(), between_line])

        peak_idx, cutoff = HGT_utils._find_sfs_peaks_and_cutoff(fs, pfs, pmax)

        num_peaks = len(peak_idx)

        # Now plot and save the figure
        _ = plt.figure()
        ax = plt.gca()
        ax.set_xlim([0.50, 1.00])
        ax.set_ylim([0, pmax*3])
        ax.bar((all_fs-df/2), all_pfs, width=df)
        ax.plot(fs[peak_idx]-df/2, pfs[peak_idx], 'rx', label='peaks detected')
        ax.set_xlabel('Major allele freq')

        if cutoff:
            ax.axvspan(min(fs), cutoff, alpha=0.1, color='red', label='SNP sites')
        ax.legend()

        path = os.path.join(config.analysis_directory, 'allele_freq',
                            species_name, str(num_peaks))
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/' + sample + '.png')
        plt.close()


def main():
    for species_name in os.listdir(os.path.join(config.data_directory, 'zarr_snps')):
        if species_name.startswith('.'):
            continue
        process_one_species(species_name)

if __name__ == "__main__":
    main()
