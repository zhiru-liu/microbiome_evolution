import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import snp_data_utils, typical_pair_utils
from parsers import parse_HMP_data
import config


# parsing HMP metadata
sample_metadata_map = {}
# First load HMP metadata
file = open(config.metadata_directory+"HMP1-2_ids_order.txt","r")
file.readline() # header
for line in file:
    items = line.split("\t")
    subject_id = items[0].strip()
    sample_id = items[1].strip()
    accession_id = items[2].strip()
    country = items[3].strip()
    continent = items[4].strip()
    order = long(items[5].strip())
    sample_metadata_map[sample_id] = (subject_id, sample_id, accession_id, country, continent, order)

subject_sample_map = parse_HMP_data.parse_subject_sample_map(sample_metadata_map)
sample_order_map = parse_HMP_data.parse_sample_order_map(sample_metadata_map)

def find_good_subjects(good_samples):
    good_subjects = {}
    for subject in subject_sample_map:
        samples = subject_sample_map[subject].keys()
        qualified = np.sum(np.isin(samples, dh.good_samples))
        if qualified >= 2:
            good_subjects[subject] = [sample for sample in samples if sample in good_samples]
    return good_subjects

def reorder_samples(samples):
    tps = [sample_order_map[sample][1] for sample in samples]
    order = np.argsort(tps)
    return np.array(samples)[order]

def process_sample(dh, sample):
    good_chromo = dh.chromosomes[dh.general_mask]

    idx = np.nonzero(dh.good_samples == sample)[0][0]
    snp_vec, coverage_arr = dh.get_snp_vector(idx)
    locations = np.where(coverage_arr)[0]
    runs, starts, ends = snp_data_utils.compute_runs_all_chromosomes(snp_vec, good_chromo[coverage_arr],
                                                                     locations=locations, return_locs=True)
    order = np.argsort(runs)
    sorted_runs = runs[order]
    sorted_starts = starts[order]
    sorted_ends = ends[order]
    return sorted_runs, sorted_starts, sorted_ends


def process_sample_pair(dh, samples):
    save_path = os.path.join(config.analysis_directory, 'de_novo_screen', dh.species_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # res is (sorted_runs, sorted_starts, sorted_ends)
    res1 = process_sample(dh, samples[0])
    res2 = process_sample(dh, samples[1])

    # plot horizontal comparison
    top_k = 35
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.set_title('Subject %s' % subject)
    ax.set_xlabel('Core genome location (syn sites)')
    ax.set_ylabel('Run length (4D syn sites)')
    ax.plot(res1[1][-top_k:], res1[0][-top_k:], 'x', label='Before: %s' % samples[0], color='tab:blue')
    ax.plot(res1[2][-top_k:], res1[0][-top_k:], 'x', color='tab:blue')
    ax.plot(res2[1][-top_k:], res2[0][-top_k:], '.', label='Before: %s' % samples[1], color='tab:orange')
    ax.plot(res2[2][-top_k:], res2[0][-top_k:], '.', color='tab:orange')

    ax.hlines(res1[0][-top_k:], res1[1][-top_k:], res1[2][-top_k:])
    ax.hlines(res2[0][-top_k:], res2[1][-top_k:], res2[2][-top_k:])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.savefig(os.path.join(save_path, '%s_%s.pdf'%samples), bbox_inches='tight')
    plt.close()


sample_df = snp_data_utils.compute_good_sample_stats()
sample_df = sample_df[sample_df['num_good_within_samples'] > 5]
for species in sample_df['species_name']:
    dh = snp_data_utils.DataHoarder(species, mode="within")
    good_subs = find_good_subjects(dh.good_samples)
    if len(good_subs)==0:
        print("%s has no two time point subjects" % species)
        continue
    print("%s has %d multi time point subjects" % (species, len(good_subs)))

    for subject in good_subs:
        samples = reorder_samples(good_subs[subject])
        process_sample_pair(dh, (samples[0], samples[1]))
        if len(samples) == 3:
            process_sample_pair(dh, (samples[1], samples[2]))
