import numpy as np


def prepare_LD(ld_map):
    if len(ld_map) == 0:
        return None

    all_distances, all_rsquared_numerators, all_rsquared_denominators, all_ns, all_intergene_distances, all_intergene_rsquared_numerators, all_intergene_rsquared_denominators, all_intergene_ns, all_control_rsquared_numerator, all_control_rsquared_denominator, all_control_n, all_pi = ld_map[('all','4D')]
    all_control_rsquared = all_control_rsquared_numerator/all_control_rsquared_denominator

    distances, rsquared_numerators, rsquared_denominators, ns, intergene_distances, intergene_rsquared_numerators, intergene_rsquared_denominators, intergene_ns, control_rsquared_numerator, control_rsquared_denominator, control_n, pi = ld_map[('largest_clade','4D')]
    control_rsquared = control_rsquared_numerator/control_rsquared_denominator

    # smooth this stuff:
    smoothed_distances = distances
    window_width = 10**(0.1)

    dmins = smoothed_distances/(window_width**0.5)
    dmaxs = smoothed_distances*(window_width**0.5)

    smoothed_rsquared_numerators = []
    smoothed_rsquared_denominators = []
    smoothed_counts = []

    all_smoothed_rsquared_numerators = []
    all_smoothed_rsquared_denominators = []
    all_smoothed_counts = []

    for dmin,dmax in zip(dmins,dmaxs):
        binned_numerators = rsquared_numerators[(distances>=dmin)*(distances<=dmax)]
        binned_denominators = rsquared_denominators[(distances>=dmin)*(distances<=dmax)]
        binned_counts = ns[(distances>=dmin)*(distances<=dmax)]
        smoothed_rsquared_numerators.append( (binned_numerators*binned_counts).sum()/binned_counts.sum() )
        smoothed_rsquared_denominators.append( (binned_denominators*binned_counts).sum()/binned_counts.sum() )
        smoothed_counts.append( binned_counts.sum() )

        binned_numerators = all_rsquared_numerators[(distances>=dmin)*(distances<=dmax)]
        binned_denominators = all_rsquared_denominators[(distances>=dmin)*(distances<=dmax)]
        binned_counts = all_ns[(distances>=dmin)*(distances<=dmax)]
        all_smoothed_rsquared_numerators.append( (binned_numerators*binned_counts).sum()/binned_counts.sum() )
        all_smoothed_rsquared_denominators.append( (binned_denominators*binned_counts).sum()/binned_counts.sum() )
        all_smoothed_counts.append( binned_counts.sum() )


    smoothed_rsquared_numerators = np.array( smoothed_rsquared_numerators )
    smoothed_rsquared_denominators = np.array( smoothed_rsquared_denominators )
    smoothed_counts = np.array( smoothed_counts )

    all_smoothed_rsquared_numerators = np.array( all_smoothed_rsquared_numerators )
    all_smoothed_rsquared_denominators = np.array( all_smoothed_rsquared_denominators )
    all_smoothed_counts = np.array( all_smoothed_counts )

    early_distances = distances[distances<101]
    early_rsquareds = rsquared_numerators[distances<101]*1.0/rsquared_denominators[distances<101]
    early_ns = ns[distances<101]

    early_distances = early_distances[early_ns>0.5]
    early_rsquareds = early_rsquareds[early_ns>0.5]
    early_ns = early_ns[early_ns>0.5]

    distances = smoothed_distances
    rsquareds = smoothed_rsquared_numerators/(smoothed_rsquared_denominators)
    ns = smoothed_counts

    distances = distances[ns>0]
    rsquareds = rsquareds[ns>0]
    ns = ns[ns>0]

    all_distances = smoothed_distances
    #all_distances = dmins
    all_rsquareds = all_smoothed_rsquared_numerators/(all_smoothed_rsquared_denominators)
    all_ns = all_smoothed_counts

    all_distances = all_distances[all_ns>0]
    all_rsquareds = all_rsquareds[all_ns>0]
    all_ns = all_ns[all_ns>0]
    return distances, rsquareds, control_rsquared, all_distances, all_rsquareds, all_control_rsquared, early_distances, early_rsquareds
