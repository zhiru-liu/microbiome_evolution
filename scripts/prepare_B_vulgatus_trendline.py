#!/usr/bin/env python3
# awkwardly need to use python3 for this single file because of the package statsmodels!

import os
import pandas as pd
import sys
sys.path.append("..")
import config
from close_pair_stage4_plot_trendline import prepare_trend_line

B_vulgatus_data = pd.read_csv(os.path.join(config.plotting_intermediate_directory, 'B_vulgatus_close_pair_data.csv'))

# might need to filter to only the plotting range
within_data = prepare_trend_line(B_vulgatus_data['clonal divs'].to_numpy(),
                                 B_vulgatus_data['within counts'].to_numpy())
between_data = prepare_trend_line(B_vulgatus_data['clonal divs'].to_numpy(),
                                  B_vulgatus_data['between counts'].to_numpy())

# save data into csv
df = pd.DataFrame({'within_x': within_data[0], 'within_y': within_data[1], 'within_sigma':within_data[2],
                   'between_x': between_data[0], 'between_y': between_data[1], 'between_sigma': between_data[2]})
# df.to_csv(os.path.join(config.plotting_intermediate_directory, "B_vulgatus_fraction_trend_line.csv"))
df.to_csv(os.path.join(config.plotting_intermediate_directory, "B_vulgatus_trend_line.csv"))
