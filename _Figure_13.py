import numpy as np
import matplotlib.pyplot as plt
from cptu_polyhedron_classifier import polyhedron_clf
from studies import STUDY4 as dataset, get_data
from sklearn.metrics import confusion_matrix
from scipy.integrate import simpson

from figure_formatting import colors, label_dict, log_tick_ax




def get_study_data( variables ):
    x_, y_, z_, labels_ = np.array([]), np.array([]), np.array([]), np.array([], dtype=np.int32)

    for matr in dataset:
        # fetch data
        x = get_data( matr, variables['x'] )
        y = get_data( matr, variables['y'] )
        z = get_data( matr, variables['z'] )

        # construct labels
        labels = np.zeros( len(x), dtype=np.int32)
        if 'non_' not in matr['name'].lower():
            labels += 1

        x_ = np.append( x_, x )
        y_ = np.append( y_, y )
        z_ = np.append( z_, z )
        labels_ = np.append( labels_, labels )
    
    return x_, y_, z_, labels_


def threshold_clf( threshold, labels ):
    labels = np.array( labels ) # outside model is -1 -> therefore always Non_sensitive (same as 2D)

    res = np.zeros( len(labels) )
    res[ labels>=threshold ] = 1

    return res


def calc_scores( labels_true, labels_predicted, t ):
    acc = np.zeros( len(t) )
    f_1 = np.zeros( len(t) )
    prec = np.zeros( len(t) )
    rec = np.zeros( len(t) )
    fpr = np.zeros( len(t) )


    for i, some_t in enumerate( t ):
        labels_pred = threshold_clf( some_t, labels_predicted )

        cm=confusion_matrix( labels_true, labels_pred )

        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]

        acc[i] = (tp+tn)/len(labels_true) * 100 # in (%)
        if (tp+fp) == 0:
            prec[i] = 100
        else:
            prec[i] = tp/(tp+fp) * 100 # in (%)
        rec[i] = tp/(tp+fn) * 100 # in (%)

        if prec[i]==0 or rec[i]==0:
            f_1[i] = 0
        else:
            f_1[i] = 2/(1/rec[i]+1/prec[i]) # in (%)
        fpr[i] = fp/(tn+fp) * 100 # in (%)

    return acc, f_1, prec, rec, fpr


def format_ax( ax, x_label, y_label, xlim=[-20,120], ylim=[0,100]):
    axis_label_fontsize = 18
    axis_tick_label_size = 16

    ax.set_xlabel( x_label, fontsize=axis_label_fontsize )
    ax.set_ylabel( y_label, fontsize=axis_label_fontsize )

    # tick fontsize
    ax.xaxis.set_tick_params( labelsize=axis_tick_label_size )
    ax.yaxis.set_tick_params( labelsize=axis_tick_label_size )

    # set axis limits
    ax.set_xlim( xlim )
    ax.set_ylim( ylim )

    ax.set_xticks( np.arange(xlim[0], xlim[1]+0.1, 20) )

    ax.grid()



def accuracy_curve( ax, acc, t ):
    fs = 16
    dy = 5
    c = (0,0,0)
    format_ax( ax, x_label='Threshold, t (%)', y_label='Accuracy (%)' )

    ax.plot( t, acc, c=c, lw=2, zorder=2 )
    
    indices = np.where(acc == np.max(acc))[0]
    y = np.average( acc[ indices ] )
    x = np.average( t[ indices ] )

    ax.plot( x, y, marker='o', mec=(1,1,1), mew=2.5, mfc=c, ms=8, clip_on=False, zorder=9 )    
    txt = ax.text( x, y+dy*0.7, str(round(y,1))+'%', ha='center', c=c, fontsize= fs )
    txt.set_bbox( dict(facecolor=(1,1,1), edgecolor=(1,1,1), pad=0.2) )


def f1_curve( ax, f1, t ):
    fs = 16
    dy = 5
    c = (237/255,28/255,46/255)
    format_ax( ax, x_label='Threshold, t (%)', y_label=r'F$_1$ score (%)' )

    ax.plot( t, f1, c=c, lw=2, zorder=2 )
    
    indices = np.where(f1 == np.max(f1))[0]
    y = np.average( f1[ indices ] )
    x = np.average( t[ indices ] )

    ax.plot( x, y, marker='o', mec=(1,1,1), mew=2.5, mfc=c, ms=8, clip_on=False, zorder=9 )    
    txt = ax.text( x, y+dy*0.7, str(round(y,1))+'%', ha='center', c=c, fontsize= fs )
    txt.set_bbox( dict(facecolor=(1,1,1), edgecolor=(1,1,1), pad=0.2) )


def prec_rec_curve( ax, prec, rec, ts ):
    fs = 16
    dy = -2
    dx = -2
    c=(0,142/255,194/255)
    format_ax( ax, x_label='Recall (%)', y_label='Precision (%)', xlim=[0,100] )

    ax.plot( rec, prec, c=c, lw=2, zorder=2, clip_on=False )

    t_list = np.array( [20,40,50,60,70,80] )
    for i, t in enumerate( t_list ):
        idx = (np.abs(ts-t)).argmin()
        y = np.average( prec[ idx ] )
        x = np.average( rec[ idx ] )
        ax.plot( x, y, marker='o', mec=(1,1,1), mew=2.5, mfc=c, ms=8, clip_on=False, zorder=9 )
        
        label = str(int(ts[idx]))
        if i==0:
            label = 't = ' + label + '%'
        
        txt = ax.text( x+dx, y+dy, label, va='top', ha='right', c=c, fontsize= fs )
        txt.set_bbox( dict(facecolor=(1,1,1), edgecolor=(1,1,1), pad=0.2) )
    

def roc_curve( ax, tpr, fpr, ts ):
    fs = 16
    dx = 2
    c = (93/255,184/255,46/255)
    hc=(.95,.95,.95)
    format_ax( ax, x_label='False positive rate, FPR (%)', y_label='True positive rate, TPR (%)', xlim=[0,100] )

    ax.plot( fpr, tpr, c=c, lw=2, zorder=2 )

    t_list = [20,40,50,60,70,80]
    for i, t in enumerate( t_list ):
        idx = (np.abs(ts-t)).argmin()
        y = np.average( tpr[ idx ] )
        x = np.average( fpr[ idx ] )
        ax.plot( x, y, marker='o', mec=(1,1,1), mew=2.5, mfc=c, ms=8, clip_on=False, zorder=9 )

        label = str(int(ts[idx]))
        if i==0:
            label = 't = ' + label + '%'

        # varies for better AUC text fit
        dy = 4/30*t - 28/3
        dy = max( -4, min(0, dy) ) - 2

        # adjustment for CM
        thc = hc
        if t in [50,60]:
            x -= 9
            y += 7
            thc=(1,1,1)


        txt = ax.text( x+dx, y+dy, label, va='center', ha='left', c=c, fontsize= fs )
        txt.set_bbox( dict(facecolor=thc, edgecolor=thc, pad=0.2) )

    # calculate & draw area under curve
    u_fpr, indices = np.unique(fpr, return_index=True)
    u_tpr = tpr[indices]

    auc = simpson( u_tpr, u_fpr )/10000
    auc_label = 'Area under curve (AUC) = ' + str( round(auc,2) )
    txt = ax.text( 99, 2, auc_label, ha='right', fontsize= 18 )
    txt.set_bbox( dict(facecolor=hc, edgecolor=hc, pad=0.1))

    ax.fill_between( x=fpr, y1=tpr, where= (0 <= fpr)&(fpr <= 100), color=hc, zorder=-2 )

    # tpr=fpr
    ax.plot( [0,100], [0,100], c=[0.5,0.5,0.5], ls='--', lw=1, zorder=1 )


    # max_cm
    cm = tpr-fpr
    idx = np.argmax( cm )

    x = fpr[idx]
    y = tpr[idx]
    max_val = r'$CM_{max}$' + ' = ' + str(round(cm[idx],1))
    ax.plot( [x,x], [x,y], c=(0,0,0), ls='--', lw=1.5, zorder=1 )
    tx = ax.text(x-dy*0.1, x + 1, max_val, fontsize=fs*.85, c=(0,0,0), rotation=90, ha='left', va='bottom', rotation_mode='anchor')


def figure_13():
    clf = polyhedron_clf('brittle_screening_2025', label_outside=-10)

    # get model variables from dataset
    model_vars = clf.variables
    x, y, z, labels_true = get_study_data( model_vars )
    
    #labels_predicted = np.random.choice( [-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], len(x) ) # for development

    import os
    import pickle
    save_path = 'labels_predicted.pkl'
    save = True
    if os.path.isfile( save_path ) and True:
        with open( save_path, 'rb') as f:
            labels_predicted = pickle.load( f )
        save=False
    else:
        labels_predicted = clf.predict( x, y, z ) # keep only this in final version

    if save:
        with open( save_path, 'wb') as f:
            pickle.dump( labels_predicted, f )

    # summarize classification results
    labels_predicted_count = labels_predicted.copy()
    labels_predicted_count[labels_predicted_count==-10]=999
    counts = np.bincount(labels_predicted_count)
    count_dict = {i: count for i, count in enumerate(counts)}
    count_dict = {k:v for k,v in count_dict.items() if v>0}

    t = np.arange( -20, 120.001, 0.01 )

    acc, f_1, prec, rec, fpr = calc_scores( labels_true, labels_predicted, t )

    fig, axs = plt.subplots( 2, 2, figsize=(12,8), tight_layout=True )

    accuracy_curve( axs[0][0], acc, t )
    f1_curve( axs[0][1], f_1, t )
    prec_rec_curve( axs[1][0], prec, rec, t )
    roc_curve( axs[1][1], rec, fpr, t  )

    k = 0
    for r in range(2):
        for c in range(2):            
            ax = axs[r][c]
            ax.text( # add subfigure index
                0.05, 0.90,  # 0.94, 0.92 good when not using legend
                '(' + chr( ord('a') + k ) + ')', 
                fontsize=22, 
                ha='center', va='center', 
                transform=ax.transAxes, 
                bbox=dict( edgecolor='none', facecolor=(1,1,1,.0) )
            )
            k += 1


    plt.savefig( 'Figure 13.png', dpi=100)
    plt.show()
    a=1


if __name__=='__main__':
    figure_13()