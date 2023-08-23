import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec

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


def plot_jitters(ax, X, ys, width, colorVal='tab:blue', alpha=0.5):
    kernel = gaussian_kde(ys)

    theory_ys = np.linspace(ys.min(),ys.max(),100)
    theory_pdf = kernel(theory_ys)

    scale = width/theory_pdf.max()

    xs = np.random.uniform(-1,1,size=len(ys))*kernel(ys)*scale

    q25 = np.quantile(ys,0.25)
    q50 = np.quantile(ys,0.5)
    q75 = np.quantile(ys,0.75)

    if len(ys)<900:
        ax.plot(X+xs,ys,'.',color=colorVal,alpha=alpha,markersize=5,markeredgewidth=0.0)
    else:
        ax.plot(X+xs,ys,'.',color=colorVal,alpha=alpha,markersize=5,markeredgewidth=0.0, rasterized=True)

    # ax.fill_betweenx(theory_ys, X-theory_pdf*scale,X+theory_pdf*scale,linewidth=0.25,facecolor=light_colorVal,edgecolor=colorVal)
    other_width = width+0.1
    ax.plot([X-other_width,X+other_width],[q25,q25],'-',color=colorVal,linewidth=1)
    ax.plot([X-other_width,X+other_width],[q50,q50],'-',color='tab:orange',linewidth=1)
    ax.plot([X-other_width,X+other_width],[q75,q75],'-',color=colorVal,linewidth=1)
    ax.plot([X-other_width,X-other_width],[q25,q75],'-',color=colorVal,linewidth=1)
    ax.plot([X+other_width,X+other_width],[q25,q75],'-',color=colorVal,linewidth=1)


def plot_cf_pd_joint(axes, x, y, block_size):
    scatter_ax, marg_ax = axes
    xs = np.linspace(0.01, 1, 100)
    ys = -np.log(xs) / block_size
    scatter_ax.plot(xs, ys, '--r', zorder=1, label='indep. SNPs')

    scatter_ax.scatter(x, y, s=1, linewidth=0, zorder=2, rasterized=True)
    marg_ax.hist(y, orientation='horizontal', bins=100, alpha=0.6)