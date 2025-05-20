import matplotlib.pyplot as plt

from figure_formatting import colors
from cptu_classification_models_2d import add_sbt_chart
from studies import STUDY4 as dataset, get_chart_data, add_data
from kde_models import get_kde_models
from _Figure_05 import add_equal_density

rows, cols = 3, 2


def figure_2():
    fig, axs = plt.subplots( rows, cols, figsize=(10,12), tight_layout=True )

    k = 0
    for row_of_axs in axs:
        for ax in row_of_axs:
            materials = get_chart_data( k, dataset ) # grab data 

            add_sbt_chart( ax, k ) # classification chart
            add_data( ax, materials ) # data from study            

            # add KDE
            KDE_threshold = 0.05
            kdes = get_kde_models( materials )
            add_equal_density( ax, kdes, threshold=KDE_threshold, ls='-', c=(100/255, 0, 200/255) )

            ax.text( # add subfigure index
                0.08, 0.93,  # 0.94, 0.92 good when not using legend
                '(' + chr( ord('a') + k ) + ')', 
                fontsize=22, 
                ha='center', va='center', 
                transform=ax.transAxes, 
                bbox=dict( edgecolor='none', facecolor=(1,1,1,.8) )
            )

            if k==0: fig.legend( markerscale=2.5, bbox_to_anchor=(.5, 1.0), fontsize=14, framealpha=1)

            k += 1

            ax.grid()

    plt.savefig( 'Figure_02.png', dpi=100 )
    plt.show()


if __name__=='__main__':
    figure_2()