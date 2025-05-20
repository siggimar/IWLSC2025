import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

# libraries specific to study
from cptu_classification_models_2d import add_sbt_chart_3d, add_sbt_chart
from studies import STUDY4 as dataset, get_chart_data, add_data
from figure_formatting import colors
from kde_models import get_kde_models, combined_log_mask, interpolate_griddata


'''
    First attempt to visualize intersection of two KDE surfaces.  Uses matplotlib
    The 3D module has a known limitation, and does not draw true-to-view 3D renderings
    Approach abandoned in favor of pyVista
'''

def apply_surface_mask( kde, threshold ):
    # mask away areas of with point densities
    mask = kde[2]>threshold    
    X_ = np.where(mask, kde[0], np.nan)
    Y_ = np.where(mask, kde[1], np.nan)
    Z_ = np.where(mask, kde[2], np.nan)
    return X_, Y_, Z_


def add_wireframe( ax, kde, matr, threshold ):
    X_, Y_, Z_ = apply_surface_mask( kde, threshold )
    label = matr[2].split('(')[0] + ' KDE surface'
    ax.plot_wireframe( X_, Y_, Z_, color=matr[4], alpha=.20, zorder=10, label=label)
    
    p = Rectangle((1e-5, 1e-5), 1e-5, 1e-5, color=matr[4], alpha= 0.2, fill=False, hatch='++', label=label)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=1e-5, zdir="z")


# draw (transparent) & extract surface contour coordinates
def add_level( ax, res, level, X, Y, Z):
    contours = ax.contour( X, Y, Z, offset=0, levels=[level], colors=[(0,0,0,0)]) # transparent

    # extract contour line coordinates
    contour_coords = []
    for some_contour in contours.allsegs:
        for segment in some_contour:
            contour_coords.append( segment )

    contour_coords = sorted(contour_coords, key=lambda x: len(x), reverse=True)
    res[level] = contour_coords


# find dict keys of close contours
def contour_keys( res, level):
    eps = 1e-5

    # all available keys within eps from level
    keys = np.array(list(res.keys()))
    diff = np.abs(keys - level)
    lvls = np.where(diff <= eps)[0]

    # return the keys (not indexes)
    return [keys[lvl] for lvl in lvls] # keys of closes levels


def draw_level( ax,  contours, level, ls=None, offset=0, lw=1, c=(0,0,0), zorder=11, label=None):
    keys = contour_keys(  contours, level ) # should be one
    for key in keys:
        for contour in  contours[key]:
            x = contour[:, 0]
            y = contour[:, 1]
            #if not label: label = str(int(key*100)) + '% sensitive'
            ax.plot( x, y, ls=ls, c=c, lw=lw, label=label ) # outside lims: for legend
            label=None # only label one contour


def add_3d_density_contours( ax, kde, contours, level):
    keys = contour_keys(  contours, level ) # should be one
    for key in keys:
        for contour in  contours[key]:
            x,y,z = interpolate_griddata( kde[0], kde[1], kde[2], contour ) #3D
            ax.plot( x, y, z, ls='-', color=(0,0,1), lw=3, label='KDE surface intersection' )


def add_equal_density( ax, kdes, threshold, levels=[], ls='--', lw=3, c=None ):
    # draws contours at Z_rel=0.5 & returns xy coords of longest
    if c is None: c=colors[5]

    # calculate equal density boundary
    X_a, Y_a, Z_a = apply_surface_mask( kdes[0], threshold )
    X_b, Y_b, Z_b = apply_surface_mask( kdes[1], threshold )

    # check if figure is defined with log-axis
    xlog = ax.get_xaxis().get_scale().lower()=='log'
    ylog = ax.get_yaxis().get_scale().lower()=='log'

    # transform XY for log_scales
    if xlog: X_a = np.power(10,X_a)
    if ylog: Y_a = np.power(10,Y_a)

    # calculate relative density
    Z_rel = Z_b / (Z_a+Z_b)

    # desired levels: [ 0.0, 0.05, 0.1,... 1.0]
    levels = [round(i * 0.05, 2) for i in range(21)]
    res = {}
    for level in levels: # use matplotlib contours to extract contour coordinates
        add_level( ax, res, level, X_a, Y_a, Z_rel)

    draw_level (ax, res, 0.5, ls=ls, lw=lw, c=c, label='Projected equal density line') # equal density

    return res


def Figure_5a( k ):
    markersize = 6
    markeralpha = .35
    markerz = -1

    fig, ax = plt.subplots( subplot_kw={"projection": "3d"}, figsize=(12,12) )
    materials = get_chart_data( k, dataset )

    # draw reference chart
    add_sbt_chart_3d( ax, k )

    # draw CPTu data
    for matr in materials:
        x, y, label, m, c, logx, logy, xlim, ylim = matr

        # removes <=0 points to be plotted on log scale
        x_, y_ = combined_log_mask( x, y, logx, logy)

        if logx: x_=np.log10(x_)
        if logy: y_=np.log10(y_)
        z_ = [0]*len(x_)

        labels = label.split('(')
        label = labels[0] + 'points (' + labels[1]

        ax.scatter3D(
            x_,
            y_,
            z_,
            label=label,
            s=markersize,
            marker=m,
            fc=c,
            alpha=markeralpha,
            zorder=markerz
        )

    # add KDE
    KDE_threshold = 0.05
    kdes = get_kde_models( materials )
    for kde, matr in zip(kdes, materials):
        add_wireframe( ax, kde, matr, threshold=KDE_threshold )

    # add equal density lines
    contours = add_equal_density( ax, kdes, threshold=KDE_threshold ) #2D
    add_3d_density_contours( ax, kde, contours, 0.5)
    a=1

    # add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict( zip(labels, handles) )
    lgnd = plt.legend( by_label.values(), by_label.keys(), markerscale=2, bbox_to_anchor=(.9, .9), fontsize=15 )

    # save/show
    plt.savefig( 'Figure_05a.png', dpi=100 )
    plt.show()


def Figure_5b( k ):
    markersize = 6
    markeralpha = .35
    markerz = -1

    fig, ax = plt.subplots( figsize=(8,8), tight_layout=True )
    materials = get_chart_data( k, dataset ) # grab data 

    add_sbt_chart( ax, k ) # classification chart
    add_data( ax, materials, markersize, markeralpha, markerz, clip_on=False ) # data from study            

    # add KDE
    KDE_threshold = 0.05
    kdes = get_kde_models( materials )
    add_equal_density( ax, kdes, threshold=KDE_threshold )
    plt.show()


if __name__=='__main__':
    k=3
    #Figure_5a( k=k )
    Figure_5b( k=k )