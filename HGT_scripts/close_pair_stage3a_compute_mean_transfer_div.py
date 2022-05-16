import numpy as np
import os
import pandas as pd
from utils import parallel_utils, close_pair_utils
import config


def ref_location_to_contig_location(ref_location, contig, contig_list, contig_cum_lens):
    return ref_location - contig_cum_lens[np.where(contig_list==contig)[0][0]]


def compute_div_in_transfers(dh, transfer_df):
    # lots of indices acrobatics in this function...
    syn_core_mask = parallel_utils.get_general_site_mask(species_name, allowed_variants=['4D'])  # ref -> 4D core
    core_mask = dh.general_mask  # ref -> core
    core_to_ref_coords = np.where(core_mask)[0]  # shape=length of core genome
    full_to_syn_mask = syn_core_mask[core_mask]  # core -> 4D core
    contig_list = pd.unique(dh.chromosomes)
    core_contig_lengths = parallel_utils.get_contig_lengths(dh.chromosomes)
    contig_cum_lens = np.insert(np.cumsum(core_contig_lengths), 0, 0)

    divs = []
    full_starts = []
    full_ends = []
    run_lengths = []
    ref_starts = []
    ref_ends = []
    ref_contigs = []
    for pair in pd.unique(transfer_df['pairs']):
        full_snp_vec, full_coverage_vec = dh.get_snp_vector(pair)
        # snp_vec and coverage_vec in 4D core genome space
        snp_vec = full_snp_vec[full_to_syn_mask[full_coverage_vec]]  # 4D syn snp vector used by the HMM
        coverage_vec = full_coverage_vec[full_to_syn_mask]

        full_to_snp_vec = full_to_syn_mask & full_coverage_vec
        full_coords = np.where(full_to_snp_vec)[0]

        good_chromo = dh.chromosomes[syn_core_mask][coverage_vec]
        contig_lengths = parallel_utils.get_contig_lengths(good_chromo)
        block_size = config.second_pass_block_size
        for _, row in transfer_df[transfer_df['pairs'] == pair].iterrows():
            # translating HMM coordinates (because of contig+block) to snp_vec coordinates
            start = close_pair_utils.block_loc_to_genome_loc(row['starts'], contig_lengths, block_size, left=True)
            end = close_pair_utils.block_loc_to_genome_loc(row['ends'], contig_lengths, block_size, left=False)
            # compute divergence of this transfer
            div = np.sum(snp_vec[int(start):int(end)]) / float(end - start)
            divs.append(div)

            # translating snp_vec coordinates to full sites coordinates
            full_start, full_end = full_coords[int(start)], full_coords[int(end)-1]  # right end is now inclusive
            full_starts.append(full_start)
            full_ends.append(full_end)

            # calculating the total number of sites in transfer (ignoring non-core genes)
            dist_between = np.sum(full_coverage_vec[full_start:full_end])
            run_lengths.append(dist_between)

            # translating snp_vec coordinates to reference genome coordinates (loc on contig)
            ref_start, ref_end = core_to_ref_coords[full_start], core_to_ref_coords[full_end]
            contig = dh.chromosomes[ref_start]
            if contig != dh.chromosomes[ref_end]:
                raise RuntimeWarning("Potential bug: Run event spanning two contigs! Species: {}; Pair: {}".format(species_name, pair))
            contig_start = ref_location_to_contig_location(ref_start, contig, contig_list, contig_cum_lens)
            contig_end = ref_location_to_contig_location(ref_end, contig, contig_list, contig_cum_lens)
            ref_contigs.append(contig)
            ref_starts.append(contig_start)
            ref_ends.append(contig_end)
    transfer_df['divergences'] = divs
    transfer_df['core genome starts'] = full_starts
    transfer_df['core genome ends'] = full_ends
    transfer_df['transfer lengths (core genome)'] = run_lengths
    transfer_df['contigs'] = ref_contigs
    transfer_df['reference genome starts'] = ref_starts
    transfer_df['reference genome ends'] = ref_ends
    return transfer_df


if __name__ == "__main__":
    second_pass_dir = os.path.join(config.analysis_directory, "closely_related", "second_pass")
    data_dir = os.path.join(config.analysis_directory, "closely_related", "third_pass")

    for filename in os.listdir(second_pass_dir):
        if filename.startswith('.'):
            continue
        species_name = filename.split('.')[0]
        print("Processing {}".format(species_name))
        filepath = os.path.join(data_dir, "%s_all_transfers.pickle" % species_name)
        if not os.path.exists(filepath):
            print("Intermediate file not found for {}, skipping".format(species_name))
            continue
        transfer_df = pd.read_pickle(filepath)
        dh = parallel_utils.DataHoarder(species_name=species_name, mode='QP', allowed_variants=['1D', '2D', '3D','4D'])
        transfer_df = compute_div_in_transfers(dh, transfer_df)
        transfer_df.to_pickle(filepath)
