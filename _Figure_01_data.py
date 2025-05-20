import numpy as np
import matplotlib.pyplot as plt


# libraries specific to study
from cptu_classification_models_2d import add_reference_models, model_defs
from studies import STUDY4 as dataset
from figure_formatting import format_axes_3D, get_ax_lims, markers, colors


def draw_data( ax, study, is_2025_study=False,s=10,alpha=0.2 ):
    for matr in study:
        c = colors[3] if matr['name']=='Sensitive' or 'Quick' in matr['name'] else colors[4]
        m = markers[0] if matr['name']=='Sensitive' or 'Quick' in matr['name'] else markers[1]
        
        x = np.array( matr['BQ'] )
        y = np.log10( matr['FSN'] )
        z = np.log10( matr['QTN'] )

        label = matr[ 'name' ].replace( 'Quick', 'Quick clay' ).replace( '_','-' ) + ' (' + str( len(x) ) + ')'

        if is_2025_study:
            y = np.log10( matr['FR'] )
            z = np.log10( matr['QQT'] )
            label = label.replace( 'Quick clay', 'Sensitive' )
            label = label.replace( 'Sensitive', 'Brittle' )

        ax.scatter3D(
            x, 
            y, 
            z,
            label=label,
            s=s,
            marker=m,
            fc=c,
            alpha=alpha,
            zorder=-1
        )


def present_model():
    fig, ax = plt.subplots( subplot_kw={"projection": "3d"}, figsize=(12,12) )

    label_dict = { # Robertson '90/91 models for reference
        'Bq': '\n' + r'$B_q$',
        'Fr': '\n' + r'$log_{10} \left( F_r \right)$',
        'Qt': '\n' + r'$log_{10} \left( Q_t \right)$',
    }

    ax_labels = [ label_dict['Bq'], label_dict['Fr'], label_dict['Qt'] ]
    ax_labels_ = 'x(Bq) y(Fr) z(Qt)'
    
    ax_lims = get_ax_lims( ax_labels )
    ax_lims[1] = ( -1, 1 )
    add_reference_models( ax, ax_lims )
    draw_data( ax, dataset, is_2025_study=True, s=14, alpha=.35 )

    # pretty print & save
    format_axes_3D( ax, ax_labels_ )   
    handles, labels = plt.gca().get_legend_handles_labels()

    by_label = dict( zip(labels, handles) )
    lgnd = plt.legend( by_label.values(), by_label.keys(), markerscale=2, bbox_to_anchor=(.65, .85), fontsize=15 )

    plt.savefig( 'Figure_01_data.png', dpi=100 )
    plt.show()


if __name__=='__main__':
    present_model()