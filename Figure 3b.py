import matplotlib.pyplot as plt

from figure_formatting import colors
from cptu_classification_models_2d import add_sbt_chart, model_defs
from studies import STUDY4 as dataset, get_chart_data, add_data
from kde_models import get_kde_models
from _Figure_05 import add_equal_density
from figure_formatting import label_dict


rows, cols = 3, 2


def figure_3a( k ):
    fig, ax = plt.subplots( figsize=(6,8), tight_layout=True )
    
    materials = get_chart_data( k, dataset ) # grab data 

    add_sbt_chart( ax, k ) # classification chart
    add_data( ax, materials, markersize=16 ) # data from study            

    # add KDE
    KDE_threshold = 0.05
    kdes = get_kde_models( materials )
    add_equal_density( ax, kdes, threshold=KDE_threshold, ls='-',lw=5, c=(100/255, 0, 200/255) )

    if k==0: fig.legend( markerscale=2.5, bbox_to_anchor=(.5, 1.0), fontsize=14, framealpha=1)

    ax.grid()

    
    # reformat this figure for larger publication
    chart_name = list(model_defs.keys())[k]
    x_var = model_defs[chart_name]['desc']['x_axis'][0]
    y_var = model_defs[chart_name]['desc']['y_axis'][0]
    ax.set_xlabel( 'Normalized sleeve friction,  ' + label_dict[x_var], fontsize=23 )
    ax.set_ylabel( 'Normalized cone resistance,  ' + label_dict[y_var], fontsize=23 )
    ax.xaxis.set_tick_params( labelsize=18 )
    ax.yaxis.set_tick_params( labelsize=18 )

    #plt.savefig( 'Figure_05a.png', dpi=110 )
    plt.show()


if __name__=='__main__':
    k=3
    figure_3a( k=3 )