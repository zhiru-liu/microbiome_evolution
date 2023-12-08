import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
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

def plot_jitters(ax, X, ys, width, if_box=True, colorVal='tab:blue', alpha=0.1, marker='.'):
    kernel = gaussian_kde(ys)

    theory_ys = np.linspace(ys.min(),ys.max(),100)
    theory_pdf = kernel(theory_ys)

    scale = width/theory_pdf.max()

    xs = np.random.uniform(-1,1,size=len(ys))*kernel(ys)*scale

    q25 = np.quantile(ys,0.25)
    q50 = np.quantile(ys,0.5)
    q75 = np.quantile(ys,0.75)

    # ax.fill_betweenx(theory_ys, X-theory_pdf*scale,X+theory_pdf*scale,linewidth=0.25,facecolor=light_colorVal,edgecolor=colorVal)
    other_width = width+0.1
    if if_box:
        ax.plot([X-other_width,X+other_width],[q25,q25],'-',color=colorVal,linewidth=1)
        ax.plot([X-other_width,X+other_width],[q50,q50],'-',color=colorVal,linewidth=1)
        ax.plot([X-other_width,X+other_width],[q75,q75],'-',color=colorVal,linewidth=1)
        ax.plot([X-other_width,X-other_width],[q25,q75],'-',color=colorVal,linewidth=1)
        ax.plot([X+other_width,X+other_width],[q25,q75],'-',color=colorVal,linewidth=1)

    if len(ys)<900:
        ax.scatter(X+xs,ys,marker=marker,color=colorVal,alpha=alpha,s=2)
    else:
        ax.scatter(X+xs,ys,marker=marker,color=colorVal,alpha=alpha,s=2, rasterized=True)
