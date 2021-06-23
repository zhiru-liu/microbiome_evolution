import sys
import os
import pickle
import numpy as np
sys.path.append("..")
import config
from utils import parallel_utils, close_pair_utils


ckpt_path = os.path.join(config.analysis_directory,
                         "closely_related", "second_pass")
for species_name in os.listdir(ckpt_path):
    if species_name.startswith('.'):
        continue
    data = pickle.load(open(os.path.join(ckpt_path, ), 'rb'))
