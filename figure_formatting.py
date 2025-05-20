import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors


# match fonts to template if available
font_path = 'C:\\Windows\\Fonts\\Cambria.ttc' # update to match your system
if os.path.isfile(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = font_prop.get_name()
    plt.rcParams['mathtext.it'] = font_prop.get_name()
    plt.rcParams['mathtext.bf'] = font_prop.get_name()


colors={ #(112/255,48/255,160/255) # '#333'
    0 : (0/255,142/255,194/255), # NPRA blue
    1 : (93/255,184/255,46/255), # NPRA green
    2 : (255/255,150/255,0/255), # NPRA orange
    3 : (237/255,28/255,46/255), # NPRA red
    4 : (68/255,79/255,85/255),  # NPRA dark grey
    5 : (0/255,0/255,255/255), # MS Office light blue
    'mesh':(0,0,0,0.4)
}


# returns color from segmented map, standard uses NPRA green/orange/red (sensitivity model)
def c_interp(percentage, color_list = [(93/255,184/255,46/255), (255/255,150/255,0/255), (237/255,28/255,46/255)]):
    cmap = mcolors.LinearSegmentedColormap.from_list("", color_list)
    # interp color at given percentage
    return cmap(percentage)

markers={
    0 : "o",
    1 : "^",
    2 : "s"
}


label_dict = {
    'Bq': r'$B_q \: (-)$',
    'f_sn': r'$f_{sn} \: (-)$',
    'q_tn': r'$q_{tn} \: (-)$',
    'Qt': r'$Q_t \: (-)$',
    'qt': r'$q_t \: (kPa)$',
    'Fr': r'$F_r \: (\%)$',
    'Rf': r'$R_f \: (\%)$',
    'fs': r'$f_s \: (kPa)$',
    'qe': r'$q_e \: (kPa)$',
    'du_n': r"$\Delta u / \sigma' _{v0}\: (kPa)$",
}


# returns tick labels to simulate log-scale in linear space for 3D plots
# see in comments here: https://stackoverflow.com/questions/3909794/plotting-mplot3d-axes3d-xyz-surface-plot-with-log-scale
def log_tick_formatter( val, pos=None ):
    return f"$10^{{{val:g}}}$"


def log_tick_ax( ax ): # make shift log scale presentation
    ax.set_major_formatter( mticker.FuncFormatter(log_tick_formatter) )
    ax.set_major_locator( mticker.MaxNLocator(integer=True) )


def format_axes_3D( ax, labels ):
    # axis (x,y,z) are defined as (lin,log,log) - drawn in linear scale    
    axis_label_size = 25
    axis_tick_label_size = axis_label_size * .75

    ax_lims = get_ax_lims( labels )
    axis_labels = get_axis_labels( labels )

    all_axis = [ax.xaxis,ax.yaxis,ax.zaxis]

    # make shift log scale presentation
    for some_axis in all_axis[1:]:
        log_tick_ax( some_axis )

    ax.set_box_aspect([1,1,1])

    # set limits
    ax.set_xlim( ax_lims[0] )
    ax.set_ylim( ax_lims[1] )
    ax.set_zlim( ax_lims[2] )

    # axis labels
    ax.set_xlabel( '\n' + axis_labels[0], fontsize=axis_label_size )
    ax.set_ylabel( '\n' + axis_labels[1], fontsize=axis_label_size )
    ax.set_zlabel( '\n' + axis_labels[2], fontsize=axis_label_size )

    # tick size
    for some_axis in all_axis:
        some_axis.set_tick_params( labelsize=axis_tick_label_size )

    # remove tick at 2 for Bq-axis
    ax.set_xticks(np.arange(-0.5,1.6,0.5))
    ax.view_init(30, 45, 0)


def format_axes_2D( ax, chart ):
    axis_label_size = 18
    axis_tick_label_size = 14

    chart_description = chart[ 'desc' ]

    # detect and set log formatting
    logx = chart_description[ 'x_axis' ][1].lower() == 'log'
    logy = chart_description[ 'y_axis' ][1].lower() == 'log'
    formatter = mticker.FuncFormatter( lambda y, _: '{:.16g}'.format(y) )
    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter( formatter )
    if logy:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter( formatter )

    # tick fontsize
    ax.xaxis.set_tick_params( labelsize=axis_tick_label_size )
    ax.yaxis.set_tick_params( labelsize=axis_tick_label_size )

    # axis labels
    ax.set_xlabel( label_dict[chart_description['x_axis'][0]], fontsize=axis_label_size )
    ax.set_ylabel( label_dict[chart_description['y_axis'][0]], fontsize=axis_label_size )
    ax.yaxis.set_label_coords(-.14, .45)

    xlim, ylim = get_chart_limits( chart )

    ax.set_xlim( xlim )
    ax.set_ylim( ylim )


def get_chart_limits( chart ):
    xlim = chart[ 'desc' ][ 'x_bounds' ]
    ylim = chart[ 'desc' ][ 'y_bounds' ]

    if min( len(xlim), len(ylim)) == 0: # not used with finalized chart definition
        xs = []
        ys = []
        for region in chart['regions']: # long list of all values
            xs += chart[ 'regions' ][ region ][ 'xy' ][0]
            ys += chart[ 'regions' ][ region ][ 'xy' ][1]

        xlim = [ min(xs), max(xs) ]
        ylim = [ min(ys), max(ys) ]

        print( chart['desc']['name'], 'xlim:', xlim, '.\tylim:', ylim )

    return [xlim, ylim]


def get_axis_labels( labels ):
    remove = [ 'x(', 'y(', 'z(', ')' ]
    for r in remove:
        labels = labels.replace(r,'')
    x, y, z = labels.split(' ')

    return [ label_dict[x], label_dict[y], label_dict[z] ]


def get_ax_lims( labels ):
    lims = {
        'v13': [ (-0.6, 2), (-1, 1), (0,3) ], # yz in log scale
        'v15/17': [ (-0.6, 2), (-3, 0), (0,3) ],
    }

    ax_lims = lims['v13'] if 'Qt' in labels else lims['v15/17']
    return ax_lims