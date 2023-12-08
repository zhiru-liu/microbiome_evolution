import matplotlib.pyplot as plt
import os
import numpy as np
def get_pretty_species_name(species_name, include_number=False, manual=False):
    
    items = species_name.split("_")
    
    pretty_name = "%s %s" % (items[0], items[1])
    
    if include_number:
        pretty_name += (" (%s)" % (items[2]))

    # manually matching GarudGood et al convention
    if manual:
        if species_name=='Faecalibacterium_prausnitzii_57453':
            return pretty_name + ' 3'
        elif species_name == 'Faecalibacterium_prausnitzii_62201':
            return pretty_name + ' 2'
    return pretty_name
    
def get_abbreviated_species_name(species_name):
    
    items = species_name.split("_")
    
    pretty_name = "%s. %s" % (items[0][0], items[1])
        
    return pretty_name


def plot_ecdf(ax, x, complementary=True, return_xy=False):
    # adapted from matplotlib 3.8
    x = np.array(x)
    argsort = np.argsort(x)
    x = x[argsort]
    cum_weights = (1. + np.arange(len(x))) / len(x)
    if not complementary:
        X = np.hstack([x[0], x])
        Y = np.hstack([0, cum_weights])
        line, = ax.plot(X, Y,
                          drawstyle="steps-post", rasterized=True)
    else:
        X = np.hstack([0, x, x[-1]])
        Y = np.hstack([1, 1, 1-cum_weights])
        line, = ax.plot(X,Y,
                          drawstyle="steps-pre", rasterized=True)
    if return_xy:
        return X, Y
    else:
        return line

def save_figure_data(data, data_names, path, filename):
    for i in range(len(data)):
        np.savetxt(os.path.join(path, filename + '_' + data_names[i] + '.csv'), data[i], delimiter=',')