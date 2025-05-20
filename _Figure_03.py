import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from studies import STUDY4 as dataset
from figure_formatting import colors, label_dict

fontsize = 17
axis_tick_label_size = 14

# manually implemented to play with the logs
def present_histogram( ax, data, n_bins, ec='black', fc=(0,0,1), label=None ):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)

    for i in range(len(hist)):
        ax.fill_between( [bin_edges[i], bin_edges[i+1]], [0, 0], [hist[i], hist[i]], ec=ec, fc=fc, label=label)
        label=None


def present_kde( ax, x, x_lims, color=colors[3], lw=3, label='Gaussian KDE' ):
    n_pts = 1000
    kde = gaussian_kde( x )
    x_vals = np.linspace( x_lims[0], x_lims[1], n_pts )
    kde_vals = kde( x_vals )
    ax.plot( x_vals, kde_vals, lw=lw, color=color, label=label, zorder=10)

    # for verification.  Use term bandwidth in paper - this is close to accurate 
    print( kde.covariance_factor() )

    return x_vals, kde_vals


def figure_3():
    n_bins = 30

    fig, axs = plt.subplots( 1,2, figsize=(12,3.5), tight_layout=True )

    # grab data
    Q_t = np.array( dataset[1]['QQT'] )    
    log_Q_t = [ False, True ]

    x_lims = [ [0.1, 100], [-1, 2] ]

    for i, (is_log, some_xlim) in enumerate( zip(log_Q_t,x_lims) ):
        # log-transform Qts for density estimates in fig b
        Q_t_dens = np.log10( Q_t ) if is_log else Q_t

        # histogram with column boundaries
        hist_col = [ colors[3][i] for i in range(3)] + [0.2]
        present_histogram( axs[i], Q_t_dens, fc=hist_col, n_bins=n_bins, label='Histogram')

        # draw points on x-axis
        axs[i].scatter( Q_t_dens, np.zeros(len(Q_t)), fc=colors[0], alpha=1, lw=0.5, clip_on=False, marker='|',s=22, label='Data Points' ) # colors[1], markers[0]

        # draw the kde
        present_kde( axs[i], Q_t_dens, some_xlim)

        if i==0: 
            axs[i].set_xscale('log')
            xlabel = label_dict['Qt']
        else:
            xlabel = r'$\text{log}_{10} \left( Q_t \right) \quad (-)$'
            axs[i].set_xticks([-1, 0, 1, 2])

        axs[i].set_xlim(some_xlim)
        axs[i].set_ylim([0,None])
        axs[i].set_xlabel( xlabel, fontsize=fontsize )
        axs[i].set_ylabel( 'Point density (-)', fontsize=fontsize )

        # tick size
        axs[i].xaxis.set_tick_params( labelsize=axis_tick_label_size )
        axs[i].yaxis.set_tick_params( labelsize=axis_tick_label_size )

        # image_index
        axs[i].text( # add subfigure index
                0.93, 0.91,  # 0.94, 0.92 good when not using legend
                '(' + chr( ord('a') + i ) + ')', 
                fontsize=22, 
                ha='center', va='center', 
                transform=axs[i].transAxes, 
                bbox=dict( edgecolor='none', facecolor=(1,1,1,.8) )
            )

    axs[i].legend( fontsize=fontsize*.95, markerscale=2 )
    plt.savefig('Figure_3.png', dpi=120)
    plt.show()


if __name__=='__main__':
    figure_3()