import os
import pandas as pd
import numpy as np
from utils import sfs_utils
import config
from scipy.signal import find_peaks, savgol_filter, peak_widths


def _get_sample_allele_freq_unfiltered(sample_idx, allele_map, allowed_variant_types=['4D']):
    """
    :param sample_idx: the index of the desired sample
    :param allele_map: allele counts map as returned by parse_snps function
    :param allowed_variant_types:
    :return: A dataframe that contains A, D, Gene names, and major allele freqs
    """
    sample_genes = []
    sample_As = []
    sample_Ds = []
    sample_locs = []
    for gene in allele_map:
        for variant in allowed_variant_types:
            for ind in xrange(len(allele_map[gene][variant]['alleles'])):
                snp = allele_map[gene][variant]['alleles'][ind]
                loc = allele_map[gene][variant]['locations'][ind][1]
                sample_genes.append(gene)
                A = snp[sample_idx][0]
                D_A = snp[sample_idx][1]
                sample_As.append(A)
                sample_Ds.append(D_A + A)
                sample_locs.append(loc)
    df = pd.DataFrame()
    df['A'] = sample_As
    df['D'] = sample_Ds
    df['Genes'] = sample_genes
    df['Freqs'] = df['A'] / df['D']
    # Compute the major allel frequency
    df['1-Freqs'] = 1 - df['Freqs']
    df['Major Freqs'] = df[['1-Freqs', 'Freqs']].max(axis=1)
    df['Locations'] = sample_locs
    return df


def get_sample_allele_freq(sample_idx, allele_map, allowed_variant_types=['4D']):
    # filtered low frequency alleles as well as sites not covered
    df = _get_sample_allele_freq_unfiltered(sample_idx, allele_map, allowed_variant_types)
    filtered_df = df[(df['D'] > 0) & (df['Freqs'] > 0.05)]
    return filtered_df


def _find_single_host_relative_snps_with_cutoff(sample_idx, found_samples, allele_map, cutoff):
    df = get_sample_allele_freq(sample_idx, allele_map)
    filtered_df = df[df['Major Freqs'] <= cutoff]
    gene_snp_map = {}
    for gene in filtered_df['Genes']:
        if gene in gene_snp_map:
            gene_snp_map[gene] += 1
        else:
            gene_snp_map[gene] = 1
    return gene_snp_map


def find_single_host_relative_snps(sample_idx, found_samples, allele_map, sfs_map):
    """
    The sample must have a clear single peak in the SFS; in other word, the sample only contain two clear strains.
    :param sample_idx: Index of the sample in the found_samples list
    :param found_samples: A list of sample id as returned by parse_snps function
    :param allele_map: allele counts map as returned by parse_snps function
    :param sfs_map: The map returned by parse_midas_data.parse_within_sample_sfs, so that no need to compute sfs
    :return: A map of {gene: snp counts}
    """
    _, cutoff = find_sfs_peaks_and_cutoff(found_samples[sample_idx], sfs_map)
    if not cutoff:
        print("Sample does not have single clean peak")
        return None
    else:
        return _find_single_host_relative_snps_with_cutoff(sample_idx, found_samples, allele_map, cutoff)


def get_within_host_bad_samples(species_name):
    blacklist_dir = os.path.join(config.analysis_directory, "within_blacklists")
    filename = "{}.txt".format(species_name)
    if filename in os.listdir(blacklist_dir):
        with open(os.path.join(blacklist_dir, filename), 'r') as f:
            bad_samples = f.read().splitlines()
    else:
        bad_samples = []
    return bad_samples


def _if_relative_SNP(AD1, AD2):
    # Given two A&(D-A) array from the allele counts map, check whether this site
    # is a relative SNP
    # If sample 1 and sample 2 have different polarization on this site, the site is a relative SNP
    # Assuming the samples are QP samples, which means a SNP usually has A/D > 80%
    if_SNP1 = AD1[0] > AD1[1]
    if_SNP2 = AD2[0] > AD2[1]
    return if_SNP1 != if_SNP2


def _if_missing(AD1, AD2):
    if (AD1[0] + AD1[1] == 0) or (AD2[0] + AD2[1] == 0):
        return True
    else:
        return False


def get_two_sample_SNP_genes(sample_idx, allele_counts_map, desired_genes=None, allowed_variant_types=['4D']):
    if len(sample_idx) != 2:
        print("Please provide only two sample idx")
        return None
    gene_SNP_map = {}
    gene_missing_map = {}
    if desired_genes is None:
        desired_genes = allele_counts_map.keys()
    for gene in allele_counts_map:
        if gene not in desired_genes:
            continue
        for variant in allowed_variant_types:
            snp_count = 0
            missing_count = 0
            for snp in allele_counts_map[gene][variant]['alleles']:
                if _if_missing(snp[sample_idx[0]], snp[sample_idx[1]]):
                    missing_count += 1
                    continue
                if _if_relative_SNP(snp[sample_idx[0]], snp[sample_idx[1]]):
                    snp_count += 1
            gene_SNP_map[gene] = snp_count
            gene_missing_map[gene] = missing_count
    return gene_SNP_map, gene_missing_map


def get_pairwise_SNP_matrix_per_gene(num_samples, allele_counts_map, desired_gene, allowed_variant_types=['4D']):
    """
    Generate the pairwise SNP counts at a single gene
    """
    if desired_gene not in allele_counts_map:
        raise ValueError
    snp_matrix = np.zeros((num_samples, num_samples))
    for variant in allowed_variant_types:
        for snp in allele_counts_map[desired_gene][variant]['alleles']:
            for i in range(num_samples):
                for j in range(i+1, num_samples):
                    if _if_missing(snp[i], snp[j]):
                        continue
                    if _if_relative_SNP(snp[i], snp[j]):
                        snp_matrix[i, j] += 1
                        snp_matrix[j, i] += 1
    return snp_matrix


def get_gene_snp_vector(gene_snp_map, all_core_genes):
    """
    :param gene_snp_map: returned by relative snps finding functions
    :param all_core_genes: all the core genes of this species, sorted by gene id
    :return: an array of gene snp counts
    """
    gene_snp_vector = np.zeros(len(all_core_genes))
    for i in xrange(len(all_core_genes)):
        if all_core_genes[i] in gene_snp_map:
            gene_snp_vector[i] = gene_snp_map[all_core_genes[i]]
        else:
            # redundant
            gene_snp_vector[i] = 0
    return gene_snp_vector


def find_runs(gene_snp_vec):
    """
    Find runs of zero snp genes.
    :param gene_snp_vec: A vector that holds the numbers of snps in the core genes of a species. Length of the vec is
    the number of core genes of this species. The genes are sorted by the name, hence location along the genome.
    :return: An array of the run lengths and an array of the run start and end locations.
    """
    run = False
    count = 0
    counts = []
    ends = []
    for i, has_snp in enumerate(gene_snp_vec):
        if not run and not has_snp:
            # start run
            run = True
        if not has_snp:
            count += 1
        if run and has_snp:
            # end run
            counts.append(count)
            count = 0
            ends.append(i - 1)
            run = False
    if run:
        # The last run
        counts.append(count)
        ends.append(len(gene_snp_vec) - 1)

    counts = np.array(counts)
    ends = np.array(ends)
    starts = ends - counts + 1
    return counts, starts, ends


def _cumulate_runs(runs_tuple, num_genes, run_len_threshold):
    # runs tuple = (run sizes, starts, ends)
    cumu_runs = np.zeros(num_genes)
    starts = runs_tuple[1][runs_tuple[0] > run_len_threshold]
    ends = runs_tuple[2][runs_tuple[0] > run_len_threshold]
    for start, end in zip(starts, ends):
        cumu_runs[start:end + 1] += 1
    return cumu_runs


def cumulate_runs_by_thresholds(all_runs_map, snp_counts_map, cutoffs, num_blocks, run_len_thresholds):
    """
    Compute the cumulative number of qualified runs per block.

    :param all_runs_map: {((i, j), (runs, starts, ends)), ...} as pickled by the script computing runs
    :param snp_counts_map: pickled dict of form {((i, j), num_snps), ...}
    :param cutoffs: cutoffs in number of snps that defined the within clade boundary
    :param num_blocks: length of the snps vector; could be number of core genes for example
    :param run_len_thresholds: a list of thresholds, runs longer than which would by cumulated
    :return: A list of arrays of shape (num_blocks,)
    """
    final_cumu_runs = []
    for threshold in run_len_thresholds:
        cumu_runs = np.zeros(num_blocks)
        lower = cutoffs[0]
        upper = cutoffs[1]
        for i, j in all_runs_map:
            if (snp_counts_map[i, j] >= upper) or (snp_counts_map[i, j] <= lower):
                continue
            run_tuple = all_runs_map[i, j]
            cumu_runs += _cumulate_runs(run_tuple, num_blocks, threshold)
        final_cumu_runs.append(cumu_runs)
    return final_cumu_runs


def smoothen_and_find_peaks(signal, max_peak, polynomial_degree=3, prominence_ratio=0.1):
    """
    :param signal: List of numbers interpreted as a time series signal
    :param max_peak: The size of the highest peak. The required prominence is relative to this.
    :param polynomial_degree: The degree of polynomial used in the smoothening process
    :param prominence_ratio: Required prominence to max_peak ratio
    :return: The peak indices and the width results
    """
    # The window size must be an odd integer
    window_size = int(np.sqrt(len(signal)))
    window_size = window_size + 1 if (window_size % 2 == 0) else window_size
    if window_size <= polynomial_degree:
        raise ValueError("Signal too short")
    smooth_signal = savgol_filter(signal, window_size, polynomial_degree)
    peak_idx = find_peaks(smooth_signal, prominence=prominence_ratio * max_peak)[0]
    peak_widths_results = peak_widths(smooth_signal, peak_idx, rel_height=1)
    return peak_idx, peak_widths_results


def find_sfs_peaks_and_cutoff(sample, sfs_map):
    """
    Return the major allele frequency that clearly separates the two-strain peak from the main peak
    :param sample: The name of the sample
    :param sfs_map: The map returned by parse_midas_data.parse_within_sample_sfs
    :return: The list of peak frequencies; and the frequency that can be used as cutoff,
    if there's no clear separation, return None
    """
    fs, pfs = sfs_utils.calculate_binned_sfs_from_sfs_map(sfs_map[sample], folding='major')
    df = fs[1] - fs[0]  # bin size
    # For peak finding, only use the polymorphic sites
    pfs = pfs[fs < 0.95]
    fs = fs[fs < 0.95]

    # Find the max peak size
    within_sites, between_sites, total_sites = sfs_utils.calculate_polymorphism_rates_from_sfs_map(sfs_map[sample])
    between_line = between_sites * 1.0 / total_sites / ((fs > 0.2) * (fs < 0.5)).sum()
    pmax = np.max([pfs[(fs > 0.1) * (fs < 0.95)].max(), between_line])
    peak_idx, right_freq = _find_sfs_peaks_and_cutoff(fs, pfs, pmax)
    return fs[peak_idx]-df/2, right_freq


def _find_sfs_peaks_and_cutoff(fs, pfs, pmax):
    """
    Helper function that takes an array of sfs values and the size of max peak,
    and finds the peaks. If only single peak, will also return the location of
    the cutoff that separates the peak with the major peak at 1
    """
    try:
        peak_idx, peak_width_results = smoothen_and_find_peaks(pfs, pmax)
    except ValueError:
        print("SFS bin too few: {}".format(len(pfs)))
        peak_idx = []

    if len(peak_idx) != 1:
        # Multiple peaks or no peak
        return peak_idx, None
    right_freq = fs[int(peak_width_results[3][0])]
    # Check the height of the separation point
    # the peak need to be pronounced enough
    if pfs[int(peak_width_results[3][0])] < 0.2 * pfs[peak_idx]:
        return peak_idx, right_freq
    else:
        return peak_idx, None


def shuffling(num_genes, num_no_snp, num_exps):
    # Return the CCDF of length of runs. i.e. Prob(runs > L is in the sample)
    gene_vec = np.zeros(num_genes)
    gene_vec[:(num_genes - num_no_snp)] = 1
    max_L = int(np.sqrt(num_genes))
    cum_runs = np.zeros(max_L)
    for i in xrange(num_exps):
        np.random.shuffle(gene_vec)
        run_counts, _, _ = find_runs(gene_vec)
        max_run = max(run_counts)
        if_has_run = [i + 1 < max_run for i in xrange(max_L)]
        cum_runs += if_has_run
    return cum_runs / num_exps


def find_p_in_cum_dist(cum_dist, p):
    # Find x such that CCDF(x) < p
    for i, prob in enumerate(cum_dist):
        if prob < p:
            return i
