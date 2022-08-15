import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import config
from utils import parallel_utils

fontsize = 6
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

pairs = [(113, 140), (183, 331)]
dh = None


def get_snp_vec(pair):
    global dh
    cache_file = os.path.join(config.plotting_intermediate_directory, "cached_close_pair_{}.csv".format(pair))
    if os.path.exists(cache_file):
        return np.loadtxt(cache_file)
    elif dh is None:
        dh = parallel_utils.DataHoarder("Bacteroides_vulgatus_57955")
    snp_vec, _ = dh.get_snp_vector(pair)
    np.savetxt(cache_file, snp_vec)
    return snp_vec


fig, ax = plt.subplots(2, 1, figsize=(4, 3))
plt.subplots_adjust(hspace=0.2)
for i in range(2):
    snp_vec = get_snp_vec(pairs[i])
    window_size = 1000
    local_pi = np.convolve(snp_vec, np.ones(window_size) / float(window_size), mode='same')

    ax[i].plot(local_pi, color='tab:grey')
    # ax[i].set_ylim(-0.001, 0.045)
    # ax[i].set_xlim([0, 260000])
ax[1].set_xlabel("Core genome synonymous location")
ax[0].set_ylabel("SNV density")
ax[1].set_ylabel("SNV density")

fig.savefig(os.path.join(config.figure_directory, 'supp', "supp_long_transfers.pdf"), bbox_inches='tight')
