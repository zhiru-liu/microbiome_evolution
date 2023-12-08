import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from Bio import Phylo
from cStringIO import StringIO
import sys
sys.path.append("..")
import config


def plot_one_species(species_name, haplotype_dir):
    # load chromosome painting result
    fs_path = os.path.join(config.analysis_directory, "fs_results",
                           "%s_unlinked_hap.chunkcounts.out" % species_name)
    fs_df = pd.read_csv(fs_path, skiprows=[0], sep=' ', index_col=0)

    # load pairwise divergence matrix
    pd_path = os.path.join(config.analysis_directory, "pairwise_divergence",
                           "between_hosts", "%s.csv" % species_name)
    pd_mat = np.loadtxt(pd_path, delimiter=',')

    with open('{}/{}/samples.ids'.format(haplotype_dir, species_name), 'r') as f:
        oldorder = np.array(f.read().splitlines())
    fullorder_idx_dir = '{}/{}/fullorder_idx.txt'.format(haplotype_dir, species_name)

    if os.path.exists(fullorder_idx_dir):
        # load cached full order
        fullorder_idx = np.loadtxt(fullorder_idx_dir).astype(int)
        fullorder = oldorder[fullorder_idx]
    else:
        # Load tree computed by fineSTRUCTURE
        tree_path = os.path.join(config.analysis_directory, "fs_results",
                                 "%s_unlinked_hap_tree.xml" % species_name)
        tree = ET.parse(tree_path)
        root = tree.getroot()
        handle = StringIO(root[-1].text)
        phylo_tree = Phylo.read(handle, "newick")
        fullorder = [x.name for x in phylo_tree.get_terminals()]
        # compute the index mapping between old order and fullorder
        # oldorder[fullorder_idx] will be the same as fullorder
        fullorder_idx = []
        for item in fullorder:
            fullorder_idx.append(np.nonzero(item == oldorder)[0][0])
        np.savetxt(fullorder_idx_dir, fullorder_idx)

    # plot both matrices
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(fs_df.loc[fullorder, fullorder])
    ax.set_title('Ancestry matrix')
    ax.set_xlabel('Sample idx, sorted by chromosome painting')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[1]
    im = ax.imshow(pd_mat[fullorder_idx, :][:, fullorder_idx])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.set_title("Pairwise divergence matrix")
    ax.set_xlabel('Sample idx, sorted by chromosome painting')

    plt.tight_layout()
    fig_path = os.path.join(config.analysis_directory,
                            "population_structure", "%s.pdf" % species_name)
    fig.savefig(fig_path, dpi=600)
    plt.close()

if __name__ == "__main__":
    # supply the path to haplotypes; actually only using the order of sample names
    hap_path = sys.argv[1]
    base_dir = 'zarr_snps'
    for species_name in os.listdir(os.path.join(config.data_directory, base_dir)):
        if species_name.startswith('.'):
            continue
        plot_one_species(species_name, hap_path)
