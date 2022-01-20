import numpy as np
import itertools
import random
import os
import collections
from utils import close_pair_utils, parallel_utils
from parsers import parse_HMP_data
import config


def load_single_subject_sample_idxs(species_name):
    """
    Load the idxs of single subject samples in the correct order
    Can apply directly to clonal fraction mat or div mat
    :param species_name:
    :return: a list of indices
    """
    sample_mask, sample_names = parallel_utils.get_QP_sample_mask(species_name)
    good_samples = sample_names[sample_mask]
    return parallel_utils.get_single_subject_idxs_from_list(good_samples)


def load_clonal_frac_mat(species_name, desired_samples=None, between_hosts=True):
    """
    Load precomputed clonal fraction matrix. If desired_samples is supplied, will sort
    matrix according to desired_samples
    :param species_name:
    :param desired_samples: np.array
    :param between_hosts: whether load the between host array (2d) or within host array (1d)
    :return: np.array of clonal fractions between pairs of QP samples
    """
    if between_hosts:
        clonal_frac_dir = os.path.join(config.analysis_directory, 'pairwise_clonal_fraction',
                                   'between_hosts', '%s.csv' % species_name)
    else:
        clonal_frac_dir = os.path.join(config.analysis_directory, 'pairwise_clonal_fraction',
                                   'within_hosts', '%s.csv' % species_name)

    if not os.path.exists(clonal_frac_dir):
        return None
    clonal_frac_mat = np.loadtxt(clonal_frac_dir, delimiter=',')
    if desired_samples is None:
        return clonal_frac_mat
    else:
        QP_samples = parallel_utils.get_QP_samples(species_name)
        name_to_idx = {x:i for i, x in enumerate(QP_samples)}
        desired_idxs = np.array([name_to_idx[x] for x in desired_samples])
        if between_hosts:
            return clonal_frac_mat[desired_idxs, :][:, desired_idxs]
        else:
            return clonal_frac_mat[desired_idxs]


def load_pairwise_div_mat(species_name, between_hosts=True):
    if between_hosts:
        div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                                       'between_hosts', '%s.csv' % species_name)
    else:
        div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                                       'within_hosts', '%s.csv' % species_name)

    if not os.path.exists(div_dir):
        return None
    div_mat = np.loadtxt(div_dir, delimiter=',')
    return div_mat


def compute_theta(species_name, clade_cutoff=[None, None], cf_cutoff=0.05, return_both=False):
    single_sub_idxs = load_single_subject_sample_idxs(species_name)
    return _compute_theta(species_name, single_sub_idxs,
                          clade_cutoff=clade_cutoff, cf_cutoff=cf_cutoff, return_both=return_both)


def _compute_theta(species_name, single_sub_idxs, clade_cutoff=[None, None], cf_cutoff=0.05, return_both=False):
    pd_mat = load_pairwise_div_mat(species_name)
    pd_mat = pd_mat[single_sub_idxs, :][:, single_sub_idxs]
    cf_mat = load_clonal_frac_mat(species_name)
    cf_mat = cf_mat[single_sub_idxs, :][:, single_sub_idxs]
    uptri = np.triu_indices(pd_mat.shape[0], 1)
    pds = pd_mat[uptri]
    cfs = cf_mat[uptri]

    clade_cutoff[0] = 0 if clade_cutoff[0] is None else clade_cutoff[0]
    clade_cutoff[1] = pds.max() if clade_cutoff[1] is None else clade_cutoff[1]

    within_mask = (cfs < cf_cutoff) & (pds < clade_cutoff[1]) & (pds > clade_cutoff[0])
    between_mask = (cfs < cf_cutoff) & (pds > clade_cutoff[1])
    within_theta = np.mean(pds[within_mask])
    if return_both:
        between_theta = np.mean(pds[between_mask])
        return within_theta, between_theta
    else:
        return within_theta


def compute_runs(dh, good_idxs):
    """
    Computing all homozygous runs for a given set of pairs/samples
    :param dh: DataHoarder instance for a given species
    :param good_idxs: list of sample idxs to process
    :return:
    """
    good_chromo = dh.chromosomes[dh.general_mask]

    run_data = {}
    num_qualified_data = 0
    for pair in good_idxs:
        # get the snp data
        snp_vec, coverage_arr = dh.get_snp_vector(pair)
        if snp_vec is None:
            continue
        runs = parallel_utils.compute_runs_all_chromosomes(snp_vec, good_chromo[coverage_arr])
        run_data[pair] = runs
        num_qualified_data += 1
    print("%s %s has %d qualified pairs out of %d" % (dh.species_name, dh.mode, num_qualified_data, len(good_idxs)))
    return run_data


def generate_within_sample_idxs(dh, clonal_frac_cutoff=config.typical_clonal_fraction_cutoff,
                                clade_cutoff=None):
    """
    :param dh: DataHoarder instance
    :param clonal_frac_cutoff: clonal fraction cutoff for keeping typically diverged pairs
    :param clade_cutoff: if provided, will be used to separate within-clade and between-clade cases
    :return: one or two lists of idxs of samples
    """
    species_name = dh.species_name
    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'within_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')

    clonal_frac_mat = load_clonal_frac_mat(species_name, between_hosts=False)

    single_subject_samples = dh.get_single_subject_idxs()
    if clade_cutoff is None:
        mask = clonal_frac_mat[single_subject_samples] < clonal_frac_cutoff
        return single_subject_samples[mask]
    else:
        mask1 = (clonal_frac_mat[single_subject_samples] < clonal_frac_cutoff) & \
                (div_mat[single_subject_samples] < clade_cutoff)
        mask2 = (clonal_frac_mat[single_subject_samples] < clonal_frac_cutoff) & \
                (div_mat[single_subject_samples] >= clade_cutoff)
        return single_subject_samples[mask1], single_subject_samples[mask2]


def generate_between_sample_idxs(dh, clonal_frac_cutoff=config.typical_clonal_fraction_cutoff,
                                 num_pairs=100, clade_cutoff=None):
    species_name = dh.species_name
    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'between_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')
    clonal_frac_mat = load_clonal_frac_mat(species_name)

    if clade_cutoff is None:
        idx1, idx2 = np.where(clonal_frac_mat < clonal_frac_cutoff)
        idx1, idx2 = idx1[idx1 > idx2], idx2[idx1 > idx2]  # dedup
        all_pairs = zip(idx1, idx2)
        if num_pairs >= len(all_pairs):
            return all_pairs
        else:
            return random.sample(all_pairs, num_pairs)
    else:
        # first perform hierarchical clustering into two clades
        cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(div_mat, clade_cutoff)
        clade1, clade2 = cluster_dict[1], cluster_dict[2]
        # next form all within-clade pairs that are typically diverged
        clade1_pairs = [x for x in itertools.combinations(clade1, 2) if clonal_frac_mat[x] < clonal_frac_cutoff]
        clade2_pairs = [x for x in itertools.combinations(clade2, 2) if clonal_frac_mat[x] < clonal_frac_cutoff]
        within_clade_pairs = clade1_pairs + clade2_pairs
        # next form all between-clade pairs that are typically diverged
        between_clade_pairs = [x for x in itertools.product(clade1, clade2) if clonal_frac_mat[x] < clonal_frac_cutoff]
        # finally, sample desired number of pairs from two lists
        if num_pairs >= len(within_clade_pairs):
            within_clade_pairs = within_clade_pairs
        else:
            within_clade_pairs = random.sample(within_clade_pairs, num_pairs)
        if num_pairs >= len(between_clade_pairs):
            between_clade_pairs = between_clade_pairs
        else:
            between_clade_pairs = random.sample(between_clade_pairs, num_pairs)
        return within_clade_pairs, between_clade_pairs


def generate_between_sample_idxs_control_country(dh, country_counts_dict,
                                                 cf_cutoff=config.typical_clonal_fraction_cutoff, num_pairs=100):
    """
    Takes in the distribution of country sample counts (from within host samples, for example), and generate
    correct sampling of between host pairs after controlling this country structure
    :param dh:
    :param country_counts_dict: A dict of country: number of samples
    :param cf_cutoff: clonal fraction cutoff to remove close pairs
    :param num_pairs: suggested number of pairs to be sampled. The actual sample number is adjusted to match the
    smallest country set (see code below for detailed handling)
    :return: dict of country: pairs
    """
    sample_country_map = parse_HMP_data.parse_sample_country_map()
    between_countries = [sample_country_map[x] for x in dh.good_samples[dh.single_subject_samples]]
    between_countries = np.array(between_countries)
    idxs_by_country = {}
    for country in country_counts_dict:
        mask = between_countries == country
        idxs_by_country[country] = dh.single_subject_samples[mask]

    # figure out the fraction of sample pairs for each country
    total_counts = np.sum(country_counts_dict.values())
    country_frac_dict = {x: country_counts_dict[x] / float(total_counts) for x in country_counts_dict}

    cf_mat = load_clonal_frac_mat(dh.species_name)
    passed_pairs_map = {}
    for country in idxs_by_country:
        country_idxs = idxs_by_country[country]
        passed_pairs = [x for x in itertools.combinations(country_idxs, 2) if cf_mat[x] < cf_cutoff]
        passed_pairs_map[country] = passed_pairs
    # the max total sampling this dataset can support, according to the required sampling fraction
    max_samples_list = [len(passed_pairs_map[country]) / country_frac_dict[country] for country in country_frac_dict]
    # if num pairs is too large, reduce to match the smallest country
    actual_samples = min(min(max_samples_list), num_pairs)

    final_selected_pairs = {}
    for country, pairs in passed_pairs_map.items():
        final_selected_pairs[country] = random.sample(pairs, int(actual_samples * country_frac_dict[country]))
    return final_selected_pairs


def compute_max_runs(runs_data):
    return np.array(map(max, runs_data.values()))


def _filter_and_sum(vals, threshold):
        return np.sum(vals[vals > threshold])


def compute_cumu_runs(runs_data, threshold):
    return np.array(list(map(lambda x: _filter_and_sum(x, threshold), runs_data.values())))


def get_joint_plot_x_y(species_name, clade_cutoff=None):
    """
    Prepare the x and y in the joint distribution plot of pairwise divergence and identical fraction
    :param species_name:
    :param clade_cutoff: If provided, will be used to cluster into major clades, and provide x,y for only the biggest clade
    :return:
    """
    single_sub_idxs = load_single_subject_sample_idxs(species_name)
    clonal_frac_dir = os.path.join(config.analysis_directory, 'pairwise_clonal_fraction',
                                   'between_hosts', '%s.csv' % species_name)
    clonal_frac_mat = np.loadtxt(clonal_frac_dir, delimiter=',')
    clonal_frac_mat = clonal_frac_mat[single_sub_idxs, :][:, single_sub_idxs]

    div_dir = os.path.join(config.analysis_directory, 'pairwise_divergence',
                           'between_hosts', '%s.csv' % species_name)
    div_mat = np.loadtxt(div_dir, delimiter=',')
    div_mat = div_mat[single_sub_idxs, :][:, single_sub_idxs]

    if clade_cutoff is not None:
        cluster_dict = close_pair_utils.get_clusters_from_pairwise_matrix(div_mat, threshold=clade_cutoff)
        bigger_clade_id = np.argmax(map(len, cluster_dict.values()))
        bigger_clade_idxs = cluster_dict.values()[bigger_clade_id]
        clonal_frac_mat = clonal_frac_mat[bigger_clade_idxs, :][:, bigger_clade_idxs]
        div_mat = div_mat[bigger_clade_idxs, :][:, bigger_clade_idxs]

    x = clonal_frac_mat[np.triu_indices(clonal_frac_mat.shape[0], 1)]
    y = div_mat[np.triu_indices(div_mat.shape[0], 1)]
    return x, y


def fit_quadratic_curve(x, y, min_x=0.1):
    """
    Helper function to prepare a "quadratic model" to explain the variance in y distribution using variable x
    Of course, for our purpose x is the identical fraction and y is the pairwise divergence
    :param min_x: sets the range of x values to be used in fitting
    :return: the resulting quadratic function
    """
    xfit = x[x >= min_x]
    yfit = y[x >= min_x]
    # adding end point at x=0
    xfit = np.hstack([xfit, [1]])
    yfit = np.hstack([yfit, [0]])
    params = np.polyfit(xfit, yfit, 2)

    def F(xs):
        res = params[0]*xs**2 + params[1]*xs + params[2]
        res[xs < min_x] = np.mean(y[x < min_x])  # fit does not extend below min x
        return res
    return F


def asexual_curve(x, block_size=config.first_pass_block_size, default=0):
    """
    The expected behavior for the joint distribution under a random mutation model (or asexual)
    :param x: input array
    :param default: the value to return when x=0
    :return: predicted y values
    """
    default_out = np.ones_like(x) * block_size * default
    y = -np.log(x, out=default_out, where=x > 0) / block_size
    return y


def get_E_rectale_within_host_countries(within_dh):
    # generating within pairs country composition
    sample_country_map = parse_HMP_data.parse_sample_country_map()
    def if_pass_cf(snp_vec):
        clonal_frac = close_pair_utils.compute_clonal_fraction(snp_vec, config.first_pass_block_size)
        return clonal_frac <= config.typical_clonal_fraction_cutoff
    good_sample_mask = np.array(within_dh.find_good_within_samples(if_pass_cf))
    samples_for_pileup = within_dh.single_subject_samples[good_sample_mask[within_dh.single_subject_samples]]
    samples_for_pileup = within_dh.good_samples[samples_for_pileup]
    countries = [sample_country_map[x] for x in samples_for_pileup]
    country_counts_dict = collections.Counter(countries)
    return country_counts_dict
