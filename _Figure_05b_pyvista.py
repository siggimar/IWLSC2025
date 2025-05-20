import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

import mpl_toolkits.mplot3d.art3d as art3d

# alternative plotter library
import pyvista as pv
import vtk

# libraries specific to study
from cptu_classification_models_2d import add_sbt_chart_3d, add_sbt_chart_3d_pyvista, get_indexed_chart_lims, get_indexed_chart_logs
from studies import STUDY4 as dataset, get_chart_data, add_data
from figure_formatting import colors
from kde_models import get_kde_models, combined_log_mask, interpolate_griddata


'''
    Second attempt to visualize KDE surface intersections. Uses vtk/pyvista.
    Discovered a bug in vtk/pyvista: slight alpha addition caps surface opacity at ~0.8.
    Transparency fixed by plotting two surfaces per model (see lines #359 and #360).
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
            if not label: label = str(int(key*100)) + '% sensitive'
            ax.plot( x, y, ls=ls, c=c, lw=lw, label=label ) # outside lims: for legend


def add_3d_density_contours( ax, kde, contours, level):
    keys = contour_keys(  contours, level ) # should be one
    for key in keys:
        for contour in  contours[key]:
            x,y,z = interpolate_griddata( kde[0], kde[1], kde[2], contour ) #3D
            ax.plot( x, y, z, ls='-', color=(0,0,1), lw=3, label='KDE surface intersection' )


def add_equal_density( ax, kdes, threshold, levels=[], linestyles=[] ):
    # draws contours at Z_rel=0.5 & returns xy coords of longest
    lw = 3
    c = colors[5]

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

    draw_level (ax, res, 0.5, ls='--', lw=lw, c=c, label='Projected equal density line') # equal density

    return res


# pyvista functions

def add_coordinate_system( plotter, k ):
    ax_lims = get_indexed_chart_lims( k )    
    ax_logs = get_indexed_chart_logs( k )
    
    ax_lims.append( [0,3] )
    ax_logs.append( False )

    for i, (some_lim, some_log ) in enumerate(zip(ax_lims, ax_logs) ):        
        if some_log: ax_lims[i]=np.log10(some_lim).tolist()



    label_dict = {
        'Bq': '',#'\n' + r'$B_q$',
        'Fr':  '',#'\n' + r'$log_{10} \left( F_r \right)$',
        'Qt':  '',#'\n' + r'$log_{10} \left( Q_t \right)$',
    }

    ax_labels = [ label_dict['Bq'], label_dict['Fr'], label_dict['Qt'] ]
    ax_lims_comb = list(ax_lims[0] + ax_lims[1] + ax_lims[2])

    grid_args = dict( 
        axes_ranges = ax_lims_comb,
        bounds = ax_lims_comb,
        font_size=12,
        xtitle=ax_labels[0], 
        ytitle=ax_labels[1], 
        ztitle=ax_labels[2],
        n_ylabels=3,
        n_zlabels=4,
        )

    plotter.show_grid(**grid_args)


def add_face( plotter, vertices, f_color, opacity):
    face = np.array( [4,0,1,2,3] ) # simple faces

    # create mesh
    mesh = pv.PolyData( vertices, face )
    plotter.add_mesh(
        mesh,
        color=f_color,
        opacity=opacity,
        show_edges=False,
        ambient=.6
    )


def get_ax_lims( k ):
    ax_lims = get_indexed_chart_lims( k )
    ax_logs = get_indexed_chart_logs( k )

    ax_lims.append([0,3])
    ax_logs.append(False)

    for i, (some_lim, some_log) in enumerate(zip(ax_lims, ax_logs)):
        if some_log: ax_lims[i] = np.log10(some_lim)

    return ax_lims


# define markers to match Figure 2
def get_marker( m, scale = 0.02 ):
    marker = pv.Circle( radius=0.8 * scale )

    if m=='^':
        xmin, xmax, ymin, ymax, zmin = [-1*scale, 1*scale, -0.6*scale, 1.2*scale, 0*scale]
        pt_a = [xmin, ymin,zmin]
        pt_b = [xmax, ymin,zmin]
        pt_c = [0, ymax, 0]
        marker = pv.Triangle( [pt_a,pt_b,pt_c] )

    return marker


def add_data_pyvista( plotter, materials, k, alpha=.35 ):
    for matr in materials:
        x, y, label, m, c, logx, logy, xlim, ylim = matr
        z = np.array([0.0]*len(x))

        if logx: x=np.log10(x)
        if logy: y=np.log10(y)
        points = np.array([x,y,z]).T

        points = points[~np.isnan(points).any(axis=1)]
        points = points[~np.isinf(points).any(axis=1)]

        point_cloud = pv.PolyData( points )

        marker = get_marker( m )

        plotter.add_mesh(
            point_cloud.glyph(scale=False, geom=marker), 
            color=c,
            opacity=alpha,

            show_edges=True,
            edge_opacity=1,
            line_width=2,
            edge_color=c,
        )


def add_ax_faces( plotter, k ):
    ax_lims = get_ax_lims( k )

    f_color = np.array( [.79,.79,.79] )
    opacity = .5
    add_face( plotter, # Z = 0
             [ ( ax_lims[0][0],ax_lims[1][0], ax_lims[2][0] ),
                ( ax_lims[0][1],ax_lims[1][0], ax_lims[2][0] ),
                ( ax_lims[0][1],ax_lims[1][1], ax_lims[2][0] ),
                ( ax_lims[0][0],ax_lims[1][1], ax_lims[2][0] )
            ], 
            f_color*0.8, 
            opacity)

    add_face( plotter, # x = x_min
             [ ( ax_lims[0][0],ax_lims[1][0], ax_lims[2][0] ),
                ( ax_lims[0][0],ax_lims[1][1], ax_lims[2][0] ),
                ( ax_lims[0][0],ax_lims[1][1], ax_lims[2][1] ),
                ( ax_lims[0][0],ax_lims[1][0], ax_lims[2][1] )
            ], 
            f_color, 
            opacity)

    add_face( plotter, # y = y_max
             [ ( ax_lims[0][0],ax_lims[1][1], ax_lims[2][0] ),
                ( ax_lims[0][1],ax_lims[1][1], ax_lims[2][0] ),
                ( ax_lims[0][1],ax_lims[1][1], ax_lims[2][1] ),
                ( ax_lims[0][0],ax_lims[1][1], ax_lims[2][1] )
            ], 
            f_color, 
            opacity)


def add_surface( plotter, kde, matr, threshold, alpha=.9 ):
    x, y, label, m, c, logx, logy, xlim, ylim = matr    
    X, Y, Z = kde[0], kde[1], kde[2]
    if threshold: X, Y, Z = apply_surface_mask(kde, threshold)

    label = matr[2].split('(')[0] + ' KDE surface'
    
    grid = pv.StructuredGrid(X, Y, Z)

    # Calculate opacity based on Z values    
    c = list(c)
    c_transparent = c + [0.0]
    c_opaque =  c + [1.0]

    color_list = [c_transparent]  + [c_opaque]*2
    if threshold: color_list = [c_opaque]  + [c_opaque]

    cmap = mcolors.LinearSegmentedColormap.from_list("", color_list)

    plotter.add_mesh(
        grid, 
        scalars=Z.T,
        cmap=cmap,
        show_scalar_bar=False,
        )

def set_view( plotter ):
    ax_lims = get_ax_lims( k ) 

    camera_pos = ( 7.0, -4, 2.5 )    
    camera_focus = list(( np.average(some_lim) for some_lim in ax_lims ))
    camera_focus[-1] *=.8
    camera_up = ( 0, 0, 1 ) 

    cpos = [
        camera_pos,
        camera_focus,
        camera_up,
        ]
    plotter.camera_position = cpos

    if True: # find desired viewpoint coordinate
        def my_cpos_callback():
            plotter.add_text(str(plotter.camera.position), name="cpos")
            return
        plotter.add_key_event("p", my_cpos_callback)


def polyline( points ): #from pyvista documentation
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange( 0, len(points), dtype=np.int_ )
    the_cell = np.insert( the_cell, 0, len(points) )
    poly.lines = the_cell
    return poly


def draw_polyline( plotter, x, y, z=None, lw=1, c=(0,0,0), label=None ):
    if z is None: z = [0]*len(x)

    pts = np.column_stack([x,y,z])    
    plotter.add_mesh( polyline( pts ), color=c, line_width=lw, label=label )



def draw_contours_2d( plotter, contours, level, c=(0,0,0.5), lw=3 ):
    keys = contour_keys(  contours, level ) # should be one
    for key in keys:
        for contour in  contours[key]:
            x, y = contour.T
            draw_polyline( plotter, x, y, lw=lw, c=c )



def draw_contours_3d( plotter, contours, kde, level, c=(0,0,0.5), lw=3 ):
    keys = contour_keys(  contours, level ) # should be one
    for key in keys:
        for contour in  contours[key]:
            x, y, z = interpolate_griddata( kde[0], kde[1], kde[2], contour ) #3D
            z = z + 0.03 # pull out of surface
            draw_polyline( plotter, x, y, z, lw=lw, c=c )

def Figure_5b_pyvista( k, show=True ):
    materials = get_chart_data( k, dataset )

    plotter = pv.Plotter( notebook=False, off_screen=not show )

     # grey coordinate system walls
    add_ax_faces( plotter, k )

    # draw reference chart
    add_sbt_chart_3d_pyvista( plotter, k )

    # draw data
    add_data_pyvista( plotter, materials, k )

    # add KDE
    KDE_threshold = 0.05
    kdes = get_kde_models( materials )
    for kde, matr in zip(kdes, materials):        
        add_surface( plotter, kde, matr, threshold=None )
        add_surface( plotter, kde, matr, threshold=0.4 ) # ugly fix: opaque above threshold

    fig, ax = plt.subplots( subplot_kw={"projection": "3d"}, figsize=(1,1) ) # used to extract contour coords
    contours = add_equal_density( ax, kdes, threshold=KDE_threshold ) #2D

    # lineweights (LW) display & save differently, set for the saved file (very thick if shown)
    draw_contours_2d( plotter, contours, level=0.5, c=(100/255,0,200/255), lw=10 ) # draw equal density lines 2D
    draw_contours_3d( plotter, contours, kde, level=0.5, c=(0,0,1.0), lw=12 ) # draw equal density lines 3D

    # draw coordinate box
    add_coordinate_system( plotter, k )
    set_view( plotter )

    if show: plotter.show( window_size=[1024, 768] )
    else: plotter.screenshot('Figure_5b.png',scale=4)
    #plotter.save_graphic('Figure_4a.svg', title='PyVista Export')#, raster=True, painter=True)


if __name__=='__main__':
    k=3
    Figure_5b_pyvista( k=k, show=False )