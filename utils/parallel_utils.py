import zarr
import dask.array as da
import numpy as np
import pandas as pd
import time
import os
import bz2
import csv
import random
from utils import sample_utils, core_gene_utils, diversity_utils, HGT_utils
from parsers import parse_midas_data, parse_HMP_data
import config


def parse_annotated_snps_to_zarr(snp_filename, zarr_dir, total_num_snps, zarr_chunk_size=(10000, 10)):
    """parse_annotated_snps_to_zarr. 
    Need to calculate total number of snps to record by 
    counting the lines of the compressed annotated_snps file.

    :param snp_filename: full path to annotated_snps file
    :param zarr_dir: full path to zarr data directory
    :param total_num_snps: total number of snps to be processed; determines the size of zarr array
    :param zarr_chunk_size: tuple defining zarr chunk size
    """
    snp_file = bz2.BZ2File(snp_filename)
    # parse the sample names from header first
    items = snp_file.readline().split()[1:]
    sample_names = sample_utils.parse_merged_sample_names(items)

    alt_dir = os.path.join(zarr_dir, "full_alt.zarr")
    depth_dir = os.path.join(zarr_dir, "full_depth.zarr")
    info_dir = os.path.join(zarr_dir, "site_info.txt")
    z_alt = zarr.open(alt_dir, mode='w', shape=(total_num_snps, len(
        sample_names)), chunks=zarr_chunk_size, dtype='i4')
    z_depth = zarr.open(depth_dir, mode='w', shape=(
        total_num_snps, len(sample_names)), chunks=zarr_chunk_size, dtype='i4')
    info_file = open(info_dir, mode='w')

    snp_idx = 0
    chunk_rows = zarr_chunk_size[0]
    # use buffer array to minimize writing to zarr
    alt_chunk = np.zeros((chunk_rows, len(sample_names)))
    depth_chunk = np.zeros((chunk_rows, len(sample_names)))

    t0 = time.time()

    # begin parsing
    for line in snp_file:
        items = line.split()

        # Save site information
        info_file.write(items[0])
        info_file.write('\n')

        # parse alt,depth pairs for samples
        pairs = map(lambda x: x.split(','), items[1:])
        pairs = np.array(pairs)
        # legacy ways of parsing; slower than current method according to profiling
        # pairs = map(lambda x: re.match("(\d+),(\d+)", x).groups(), items[1:])
        # alts = map(lambda x: float(x[0]), pairs)
        # depths = map(lambda x: float(x[1]), pairs)

        alt_chunk[snp_idx % chunk_rows, :] = pairs[:, 0]
        depth_chunk[snp_idx % chunk_rows, :] = pairs[:, 1]

        # write buffer array to zarr
        if (snp_idx + 1) % chunk_rows == 0:
            begin = snp_idx/chunk_rows * chunk_rows
            end = snp_idx + 1
            z_alt[begin:end, :] = alt_chunk.astype(int)
            z_depth[begin:end, :] = depth_chunk.astype(int)

        snp_idx += 1

        if (snp_idx % 1000) == 0:
            print("%dk sites processed...\n @ %d secs" % (snp_idx/1000,
                                                          time.time() - t0))
    # saving the last chunk
    begin = (snp_idx-1)/chunk_rows * chunk_rows
    end = snp_idx
    z_alt[begin:end, :] = alt_chunk[:(snp_idx % chunk_rows)].astype(int)
    z_depth[begin:end, :] = depth_chunk[:(snp_idx % chunk_rows)].astype(int)

    snp_file.close()
    info_file.close()


def parse_snp_info(info_filename):
    with open(info_filename, mode='r') as f:
        info_items = f.read().splitlines()
    split_info_items = map(lambda x: x.split('|'), info_items)

    def _get_pvalue(items):
        if len(items) > 5:
            return float(items[5])
        else:
            return float(items[4])

    chromosomes = np.array(map(lambda x: x[0], split_info_items))
    locations = np.array(map(lambda x: int(x[1]), split_info_items))
    gene_names = np.array(map(lambda x: x[2], split_info_items))
    variants = np.array(map(lambda x: x[3], split_info_items))
    pvalues = np.array(map(_get_pvalue, split_info_items))
    return chromosomes, locations, gene_names, variants, pvalues


def get_general_site_mask(gene_names, variants, pvalues, allowed_genes, allowed_variants=['4D']):
    # allowed genes must be a list; cannot be a set!
    core_gene_mask = np.isin(gene_names, allowed_genes)
    variant_mask = np.isin(variants, allowed_variants)
    p_value_mask = pvalues < 0.05
    return core_gene_mask & variant_mask & p_value_mask


def get_QP_filtered_snps(alt_arr, depth_arr, site_mask, sample_mask):
    selected_depths = depth_arr[:, sample_mask]
    selected_depths = selected_depths[site_mask, :]

    selected_alts = alt_arr[:, sample_mask]
    selected_alts = selected_alts[site_mask, :]

    # because QP, polarize by 0.5 freq cutoff
    polarized = selected_alts >= 0.5 * selected_depths
    polarized = polarized.compute()

    covered_mask = selected_depths > 0
    covered_mask = covered_mask.compute()
    return polarized, covered_mask


def get_within_host_filtered_snps(alt_arr, depth_arr, site_mask, sample_mask, cutoffs):
    # because small arrays, can compute directory
    selected_depths = depth_arr[:, sample_mask]
    selected_depths = selected_depths[site_mask, :]

    selected_alts = alt_arr[:, sample_mask]
    selected_alts = selected_alts[site_mask, :]

    # because within, polarize by sites of intermediate frequency
    # cutoffs are major allele freq cutoffs
    alt_upper_thresholds = selected_depths * cutoffs.reshape(1, len(cutoffs))
    alt_lower_thresholds = selected_depths * (1 - cutoffs.reshape(1, len(cutoffs)))
    polarized = (selected_alts <= alt_upper_thresholds) & (selected_alts >= alt_lower_thresholds)
    polarized = polarized.compute()

    # for getting the actual haplotype of the shared regions
    polarized_hap = selected_alts >= 0.5 * selected_depths
    polarized_hap = polarized_hap.compute()

    covered_mask = selected_depths > 0
    covered_mask = covered_mask.compute()
    return polarized, polarized_hap, covered_mask


def compute_good_sample_stats():
    basepath = os.path.join(config.analysis_directory, 'typical_pairs')
    csvpath = os.path.join(basepath, 'sample_stats.csv')
    if not os.path.exists(csvpath):
        print("Did not find cached results. Calculating sample stats for all species")
        if not(os.path.exists(basepath)):
            os.makedirs(basepath)

        csv_file = open(csvpath, 'w')
        writer = csv.writer(csv_file)
        writer.writerow(['species_name', 'num_total_samples', 'num_high_coverage_samples', 'num_qp_samples',
                         'num_good_within_samples'])

        for species_name in os.listdir(os.path.join(config.data_directory, 'zarr_snps')):
            if species_name.startswith('.'):
                continue
            print("processing %s" % species_name)
            qp_mask, _ = get_QP_sample_mask(species_name)
            good_within_mask, _, _ = get_single_peak_sample_mask(species_name)
            num_desired_samples = len(diversity_utils.calculate_highcoverage_samples(species_name))

            writer.writerow([species_name, len(qp_mask), num_desired_samples, np.sum(qp_mask), np.sum(good_within_mask)])

        csv_file.close()

    return pd.read_csv(csvpath)


'''
    A class for holding relevant data of a species for analysis
'''
class DataHoarder:
    def __init__(self, species_name, mode="QP", allowed_variants=["4D"]):
        self.species_name = species_name
        self.data_dir = os.path.join(
                config.data_directory, 'zarr_snps', species_name)
        if not os.path.exists(self.data_dir):
            raise ValueError('No data found for {} at default dir:\n{}'
                             .format(species_name, self.data_dir))

        if (mode == "QP"):
            self.sample_mask, sample_names = get_QP_sample_mask(species_name)
            self.good_samples = sample_names[self.sample_mask]
            self.mode = mode
            print("Keeping only QP samples, which is %d in total" %
                  np.sum(self.sample_mask))
        elif (mode == "within"):
            self.sample_mask, sample_names, self.peak_cutoffs = get_single_peak_sample_mask(species_name)
            self.good_samples = sample_names[self.sample_mask]
            self.mode = mode
            print("Keeping only simple within host samples, which is %d in total" %
                  np.sum(self.sample_mask))
        else:
            raise ValueError("Only support QP or within modes")

        self.single_subject_samples = self.get_single_subject_idxs()

        alt_arr = da.from_zarr('{}/full_alt.zarr'.format(self.data_dir))
        depth_arr = da.from_zarr('{}/full_depth.zarr'.format(self.data_dir))
        # increase chunk size to reduce overhead
        rechunked_alt_arr = alt_arr.rechunk((1000000, 10))
        rechunked_depth_arr = depth_arr.rechunk((1000000, 10))

        print("Loading site info")
        res = parse_snp_info(os.path.join(self.data_dir, 'site_info.txt'))
        self.chromosomes = res[0]
        self.locations = res[1]
        self.gene_names = res[2]
        self.variants = res[3]
        self.pvalues = res[4]

        print("Filtering and loading sites into memory")
        t0 = time.time()
        core_genes = core_gene_utils.get_sorted_core_genes(species_name)
        # len = total number of snps
        self.general_mask = get_general_site_mask(
                self.gene_names, self.variants, self.pvalues, core_genes,
                allowed_variants=allowed_variants)
        if (mode == "QP"):
            self.snp_arr, self.covered_arr = get_QP_filtered_snps(
                    rechunked_alt_arr, rechunked_depth_arr, self.general_mask, self.sample_mask)
            print("Finish loading sites, took %d secs" % (time.time() - t0))
        else:
            self.snp_arr, self.naive_haplotype, self.covered_arr = get_within_host_filtered_snps(
                    rechunked_alt_arr, rechunked_depth_arr, self.general_mask,
                    self.sample_mask, self.peak_cutoffs)
            print("Finish loading sites, took %d secs" % (time.time() - t0))

    def get_snp_mask(self):
        # keep sites that pass the snp prevalence threshold
        # Together with self.general_mask, this filter should produce exactly the
        # same sites as parse_midas_data.parse_snps

        # Need to use the alt and depth arr again
        alt_arr = da.from_zarr('{}/full_alt.zarr'.format(self.data_dir))
        depth_arr = da.from_zarr('{}/full_depth.zarr'.format(self.data_dir))
        # increase chunk size to reduce overhead
        rechunked_alt_arr = alt_arr.rechunk((1000000, 10))
        rechunked_depth_arr = depth_arr.rechunk((1000000, 10))
        filtered_depth = rechunked_depth_arr[:, self.sample_mask]
        filtered_alt = rechunked_alt_arr[:, self.sample_mask]

        # Some snps need to be polarized according to pop_freqs
        from plos_bio_scripts import calculate_snp_prevalences
        population_freqs = calculate_snp_prevalences.parse_population_freqs(
                self.species_name, polarize_by_consensus=False)
        all_pop_freqs = np.array(map(lambda x: population_freqs.get(x, 0),
                                 zip(self.chromosomes, self.locations)))
        sites_to_flip = all_pop_freqs > 0.5

        # take care of sites need not polarize
        round_1_mask = self.general_mask & np.invert(sites_to_flip)
        alt_threshold = da.ceil(filtered_depth[round_1_mask, :] * config.parse_snps_min_freq) + 0.5
        passed_snp_mask1 = da.sum(filtered_alt[round_1_mask, :] > alt_threshold, axis=1) > 0

        # then flip alt
        round_2_mask = self.general_mask & sites_to_flip
        alt_threshold2 = da.ceil(filtered_depth[round_2_mask, :] * config.parse_snps_min_freq) + 0.5
        polarized_alts = filtered_depth[round_2_mask, :] - filtered_alt[round_2_mask, :]
        passed_snp_mask2 = da.sum(polarized_alts > alt_threshold2, axis=1) > 0

        # perform dask computation
        passed_snp_mask1 = passed_snp_mask1.compute()
        passed_snp_mask2 = passed_snp_mask2.compute()
        final_mask = self.general_mask.copy()
        final_mask[self.general_mask & np.invert(sites_to_flip)] = passed_snp_mask1
        final_mask[self.general_mask & sites_to_flip] = passed_snp_mask2
        print("%d sites left after applying snp filter" % np.sum(final_mask))
        
        return final_mask[self.general_mask]

    def get_covered_sites_mask(self, prev_thre=0.9):
        # Keep sites that have enough samples NOT MISSING
        min_samples = self.covered_arr.shape[1] * prev_thre
        return np.sum(self.covered_arr, axis=1) >= min_samples

    def get_local_haplotype(self, start, end, force_QP=False, only_snps=True, if_polarize=True):
        """
        Return a haplotype and a missing data array for all samples
        :param start: start genome location
        :param end: end genome location (non inclusive)
        :param force_QP: whether calculate haplotype from snp_arr regardless if mode='within'
        :param only_snps: whether throw away sites with no snps
        :param if_polarize: whether polarize the sites according to the dominant allele
        :return:
        """
        if self.mode == 'QP' or force_QP:
            snps = self.snp_arr[start:end, :]
        else:
            snps = self.naive_haplotype[start:end, :]
        covered = self.covered_arr[start:end, :]
        # make sure the missing sites are not counted as snps
        snps[~covered] = False
        if only_snps:
            has_snps = snps.sum(axis=1) > 0
            snps = snps[has_snps, :]
            covered = covered[has_snps, :]

        if if_polarize:
            to_flip = snps.sum(axis=1) > (0.5 * covered.sum(axis=1))
            snps[to_flip, :] = ~snps[to_flip, :]
        return snps, covered

    def get_haplotype_mask(self, prev_thre=0.9):
        prev_mask = self.get_covered_sites_mask(prev_thre)
        snp_mask = self.get_snp_mask()
        return prev_mask & snp_mask

    def get_single_subject_idxs(self):
        sub_sam_map = parse_HMP_data.parse_subject_sample_map()
        sam_sub_map = sample_utils.calculate_sample_subject_map(sub_sam_map)
        subs = list(map(sam_sub_map.get, self.good_samples))
        seen = set()
        good_idxs = []
        for i, sub in enumerate(subs):
            if sub not in seen:
                good_idxs.append(i)
                seen.add(sub)
        return np.array(good_idxs)

    def get_snp_vector(self, idx):
        if self.mode == 'QP':
            return get_two_QP_sample_snp_vector(
                    self.snp_arr, self.covered_arr, idx)
        else:
            return get_within_sample_snp_vector(
                    self.snp_arr, self.covered_arr, idx)

    def sample_local_T_pairwise(self, l):
        # l is the size of the chunk for T estimation
        pair = random.sample(self.single_subject_samples, 2)
        snp_vec, _ = self.get_snp_vector(pair)
        genome_div = np.mean(snp_vec)
        start_idx = np.random.randint(0, len(snp_vec) - l)
        local_div = np.mean(snp_vec[start_idx:start_idx + l])
        return local_div, genome_div

    def save_haplotype_fs(self, save_dir, prev_thre=0.9):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "haplotypes.phase")
        hap_mask = self.get_haplotype_mask(prev_thre)
        good_sites = self.snp_arr[hap_mask, :]
        good_locations = self.locations[self.general_mask][hap_mask]
        assert(len(good_locations) == good_sites.shape[0])
        n_haplotypes = good_sites.shape[1]
        n_sites = good_sites.shape[0]
        print("Saving %d sites to fineSTRUCTURE format" % n_sites)
        with open(save_path, 'a') as f:
            f.write(str(n_haplotypes))
            f.write('\n')
            f.write(str(n_sites))
            f.write('\n')
            f.write('P ')
            f.write(' '.join(good_locations.astype(str)))
            f.write('\n')
            np.savetxt(f, np.transpose(good_sites.astype(int)), delimiter='', fmt='%d')

        save_path = os.path.join(save_dir, "samples.ids")
        print("Saving %d names to fineSTRUCTURE format" % len(self.good_samples))
        with open(save_path, 'w') as f:
            for name in self.good_samples:
                f.write('id_' + name + '\n')


'''
    The next few functions are not limited to zarr arr, so should be moved
    to another file eventually
'''


def get_two_QP_sample_snp_vector(snp_arr, coverage_arr, sample_ids):
    id1 = sample_ids[0]
    id2 = sample_ids[1]
    mask = coverage_arr[:, id1] & coverage_arr[:, id2]
    vec = snp_arr[:, id1] != snp_arr[:, id2]
    return vec[mask], mask


def get_within_sample_snp_vector(snp_arr, coverage_arr, sample_id):
    mask = coverage_arr[:, sample_id]
    return snp_arr[mask, sample_id], mask


def _compute_runs_single_chromosome(snp_vec, locations=None):
    # get the locations of snps in the vector
    site_locations = np.nonzero(snp_vec)[0]
    if locations is not None:
        locs = locations[site_locations]
    else:
        locs = site_locations
    return locs[1:] - locs[:-1]


def compute_runs_all_chromosomes(snp_vec, chromosomes, locations=None):
    all_runs = []
    for chromo in np.unique(chromosomes):
        subvec = snp_vec[chromosomes==chromo]
        if locations is not None:
            subloc = locations[chromosomes==chromo]
        else:
            subloc = None
        all_runs.append(_compute_runs_single_chromosome(subvec, subloc))
    return np.concatenate(all_runs)


def get_sample_names(species_name):
    snp_file = bz2.BZ2File("%ssnps/%s/annotated_snps.txt.bz2" % (config.data_directory, species_name), "r")
    line = snp_file.readline()
    items = line.split()[1:]
    sample_names = sample_utils.parse_merged_sample_names(items)
    snp_file.close()
    return sample_names


def get_raw_data_idx_for_sample(species_name, sample_name):
    # sometimes need to work with zarr array directly; so helpful to get the zarr array index for a given sample
    sample_names = get_sample_names(species_name)
    res = np.where(sample_names==sample_name)[0]
    if len(res) > 0:
        return res[0]
    else:
        return None


def get_QP_sample_mask(species_name):
    sample_names = get_sample_names(species_name)

    QP_samples = set(diversity_utils.calculate_haploid_samples(species_name))
    highcoverage_samples = set(diversity_utils.calculate_highcoverage_samples(species_name))
    allowed_samples = QP_samples & highcoverage_samples
    return np.isin(sample_names, list(allowed_samples)), sample_names


def get_QP_samples(species_name):
    mask, all_samples = get_QP_sample_mask(species_name)
    QP_samples = all_samples[mask]
    return QP_samples


def get_single_peak_sample_mask(species_name):
    """
    Compute a mask that keep only samples suitable for within host analysis
    A sample need to be 1) well covered, 2) has single clean peak
    The list of sample names and the list of peak cutoffs will also be returned
    """
    sample_names = get_sample_names(species_name)

    blacklist = set(HGT_utils.get_within_host_bad_samples(species_name))

    highcoverage_samples = set(diversity_utils.calculate_highcoverage_samples(species_name))
    single_peak_dir = os.path.join(config.analysis_directory, 'allele_freq', species_name, '1')
    if not os.path.exists(single_peak_dir):
        print("No single peak samples found for {}".format(species_name))
        mask = np.zeros(len(sample_names)).astype(bool)
        return mask, sample_names, np.array([])
    else:
        single_peak_samples = set([f.split('.')[0] for f in os.listdir(single_peak_dir) if not f.startswith('.')])
        allowed_samples = single_peak_samples & highcoverage_samples - blacklist
    mask = np.isin(sample_names, list(allowed_samples))

    # filter samples with a clean single peak
    _, sfs_map = parse_midas_data.parse_within_sample_sfs(
        species_name, allowed_variant_types=set(['4D']))
    results = [HGT_utils.find_sfs_peaks_and_cutoff(
        sample, sfs_map) for sample in sample_names[mask]]
    cutoffs = np.array([res[1] for res in results])
    clean_peak_mask = np.array([cutoff is not None for cutoff in cutoffs])
    mask[mask] = clean_peak_mask
    good_cutoffs = cutoffs[clean_peak_mask]
    return mask, sample_names, good_cutoffs.astype(float)


def get_contig_lengths(good_chromo):
    contig_lengths = []
    for chromo in np.unique(good_chromo):
        # iterate over contigs; similar to run length dist calculation
        contig_len = np.sum(good_chromo==chromo)
        contig_lengths.append(contig_len)
    return contig_lengths


def get_contig_boundary(filtered_chromosomes):
    ls = []
    for i in xrange(len(filtered_chromosomes)-1):
        if filtered_chromosomes[i+1] != filtered_chromosomes[i]:
            ls.append(i)
    return np.array(ls)

