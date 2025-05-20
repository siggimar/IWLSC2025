import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d

from studies import STUDY4 as dataset
from figure_formatting import colors, label_dict, log_tick_ax
from _Figure_03 import present_kde

fontsize = 17
axis_tick_label_size = 14
xlims = [-1,3]
ylims = [0,2]
decision_boundary = np.log10(9)


def find_intersections( x_n, y_n, x_b, y_b, threshold, eps=1e-2 ):
    # define interpolants
    interp_n = interp1d( x_n, y_n, kind='linear', fill_value='extrapolate' )
    interp_b = interp1d( x_b, y_b, kind='linear', fill_value='extrapolate' )

    # establish the domain
    x_min = max( min(x_n), min(x_b) )
    x_max = min( max(x_n), max(x_b) )
    x_values = np.linspace( x_min, x_max, 1000 )

    # calculate elementwise difference
    diff = np.abs( interp_n(x_values)-interp_b(x_values) )

    # extract indices with smallest difference  (includes regions with ~0)
    intersection_indices = np.where( diff<eps )[0]

    # extract intersection coords where interpolant is above the threshold
    intersections = [ (x_values[i], interp_n(x_values[i])) for i in intersection_indices if interp_n(x_values[i]) > threshold and interp_b(x_values[i]) > threshold ]

    return intersections


def fill_between(ax, x, y, x_1, x_2, color):
    ax.fill_between(
        x=x, 
        y1= y, 
        where=( x_1<x )&( x<x_2 ),
        color=color,
        alpha= 0.3
    )


def annotate( ax ):
    dx = 0.1
    dy = 0.12
    
    # decision boundary
    y_db = ylims[1]*.96
    x_db = decision_boundary

    # right side
    t = ax.text( x_db+dx, y_db, 'predict Non-brittle (negative)', rotation=0, verticalalignment='center', horizontalalignment='left', size=fontsize, zorder=4 )
    ax.add_patch( patches.FancyArrowPatch((x_db+dx,y_db-dy),(x_db+dx+.7,y_db-dy) , arrowstyle='->', mutation_scale=30, lw=1.2, zorder=2) )
    
    # left side
    t = ax.text( x_db-dx, y_db, 'predict Brittle (positive)', rotation=0, verticalalignment='center', horizontalalignment='right', size=fontsize, zorder=4 )
    ax.add_patch( patches.FancyArrowPatch((x_db-dx,y_db-dy),(x_db-dx-.7,y_db-dy) , arrowstyle='->', mutation_scale=30, lw=1.2, zorder=2) )
    
    pts = [(0.60, 1.4),(1.03, 0.2),(0.7, 0.5),(1.15, 0.5)]

    # true positives
    ann = ax.annotate('True positives (TP)\n\n',
                    xy=pts[0], xycoords='data',
                    xytext=(-0.5, 1.5), textcoords='data',
                    size=fontsize, va="center", ha="center",
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3,rad=-0.2",
                                    fc="w"), zorder=99)

    # false negatives
    ann = ax.annotate('False negatives (FN)',
                    xy=pts[1], xycoords='data',
                    xytext=(2, 0.5), textcoords='data',
                    size=fontsize, va="center", ha="center",
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3,rad=-0.2",
                                    fc="w"), zorder=99)

    # false positives
    ann = ax.annotate('False positives (FP)',
                    xy=pts[2], xycoords='data',
                    xytext=(-0.5, .9), textcoords='data',
                    size=fontsize, va="center", ha="center",
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3,rad=0.2",
                                    fc="w"), zorder=99)

    # true negatives
    ann = ax.annotate('True negatives (TN)',
                    xy=pts[3], xycoords='data',
                    xytext=(2, .9), textcoords='data',
                    size=fontsize, va="center", ha="center",
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3,rad=0.2",
                                    fc="w"), zorder=99)

    for pt in pts:
        ax.plot( *pt, ls='none', marker='o', c=(0,0,0), ms=10 )


def figure_4():
    Q_t_non_sens = np.array( dataset[0]['QQT'] )
    Q_t_brittle = np.array( dataset[1]['QQT'] )

    Q_t_non_sens_dens = np.log10( Q_t_non_sens )
    Q_t_brittle_dens = np.log10( Q_t_brittle )

    fig, ax = plt.subplots( figsize=(12,5), tight_layout=True )

    # calculate KDEs
    ns_x, ns_y = present_kde( ax, Q_t_non_sens_dens, xlims, color=colors[4], label='Non-brittle KDE' )
    br_x, br_y = present_kde( ax, Q_t_brittle_dens, xlims, color=colors[3], label='Brittle KDE' )

    # extract equal density points above minimum density threshold
    coords = find_intersections( ns_x, ns_y, br_x, br_y, threshold=0.5 )
    x_avg = (coords[0][0]+coords[1][0])/2 # function returns 2 closest to intersection
    y_avg = (coords[0][1]+coords[1][1])/2
    ax.plot( x_avg, y_avg, ls='none', marker='o', ms=10, mec=(100/255,0,200/255), mfc=(100/255,0,200/255), label='Equal density', zorder=20)

    # annotate Equal density boundary    
    ax.add_patch( patches.FancyArrowPatch( (x_avg,coords[0][1]),(x_avg,0), ls='--', arrowstyle='->', color=(100/255,0,200/255), mutation_scale=30, lw=1.2, zorder=2) )
    ax.text( x_avg, 0-0.075, "{:.2f}".format(np.power(10,x_avg)), verticalalignment='center', horizontalalignment='center', size=fontsize, color=(100/255,0,200/255), zorder=4 )

    # draw decision boundary
    ax.plot([decision_boundary]*2, ylims, lw=3, ls='--', c=(0,0,0), label='Decision boundary', zorder=20)

    # shade kde regions by decision boundary
    fill_between( ax, ns_x, ns_y, xlims[0], decision_boundary, colors[4] ) # false positives
    fill_between( ax, ns_x, ns_y, decision_boundary, xlims[1], colors[1] ) # true negatives

    fill_between( ax, br_x, br_y, xlims[0], decision_boundary, colors[3] ) # true positives
    fill_between( ax, br_x, br_y, decision_boundary, xlims[1], colors[0] ) # false negatives

    annotate( ax )

    log_tick_ax( ax.xaxis )
    ax.set_xlim( xlims )
    ax.set_ylim( ylims )

    ax.set_ylabel( 'Kernel density estimate, KDE (-)', fontsize=fontsize )
    ax.set_xlabel( label_dict['Qt'], fontsize=fontsize )
    ax.xaxis.set_tick_params( labelsize=axis_tick_label_size )    
    ax.yaxis.set_tick_params( labelsize=axis_tick_label_size )
    
    ax.legend( fontsize=fontsize, bbox_to_anchor=(.72, .93), loc='upper left')
    plt.savefig('Figure_4.png', dpi=120)
    plt.show()


if __name__=='__main__':
    figure_4()