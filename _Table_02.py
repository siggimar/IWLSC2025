import numpy as np
from cptu_classification_charts import general_model
from cptu_classification_models_2d import model_defs as sbts
from studies import STUDY4 as dataset, get_chart_data, add_data
from figure_formatting import colors, markers


# counts tp,fp,tn,fn && calc derived metrics
def calc_metrics( label, hits, some_res ):
    k = len(hits)

    # count hits
    if 'brittle' in label.lower():
        some_res['p'] = k
        some_res['tp'] = sum(hits)
        some_res['fn'] = k - some_res['tp']
    else:
        some_res['n'] = k
        some_res['fp'] = sum(hits)
        some_res['tn'] = k - some_res['fp']

    # second pass only
    if 'tp' in some_res.keys() and 'fp' in some_res.keys():
        # derived parameters
        some_res['Accuracy'] = ( some_res['tp']+some_res['tn'] ) / ( some_res['p']+some_res['n'] ) * 100
        if ( some_res['tp']+some_res['fp'] )==0:
            some_res['Precision']=999
        else:
            some_res['Precision'] = some_res['tp'] / ( some_res['tp']+some_res['fp'] ) * 100
        some_res['Recall'] = some_res['tp'] / some_res['p'] * 100

        if some_res['Precision']==0 or some_res['Recall']==0:
            some_res['F1-score'] = 0
        else:    
            some_res['F1-score'] = 2 / ( 1/some_res['Precision']+1/some_res['Recall'] )

        some_res['FPR'] = some_res['fp'] / some_res['n'] * 100

        some_res['CM'] = some_res['Recall'] - some_res['FPR']


# print results readable form
def pretty_print_table( results ):
    eps = 1 # margin

    # max length for SBTs and cols with margin
    sbts = max([len(k) for k in results.keys()]) + eps
    headers = max([len(k) for k in results[next(iter(results))].keys()]) + eps

    top = 'SBT chart'.rjust(sbts)
    cols = [ header.rjust(headers) for header in results[next(iter(results))].keys() ]
    top += ''.join(cols)
    print(top)
    
    for sbt in results:
        line = sbt.rjust(sbts)
        for col in results[sbt]:
            line += str(round(results[sbt][col],1)).rjust(headers)
        print(line)


# perform calculations
def classification_2D():
    plot_sens_prediction = True
    res = {}
    chart_names = list(sbts.keys())

    for k in range(6):
        chart_name = chart_names[k]
        materials = get_chart_data( k, dataset ) # grab data

        tmp_res = {}

        for matr in materials:
            x, y, label, m, c, logx, logy, xlim, ylim = matr

            chart_classifier = general_model( sbts[chart_name], fill_regions=False )
            some_result = np.array(chart_classifier.predict( x, y ))
            hits = ( some_result==sbts[chart_name]['desc']['sensitive'] )

            matr.append(hits)

            calc_metrics( label, hits, tmp_res )

        res[chart_name] = tmp_res

        if plot_sens_prediction: # visual check of results
            chart_classifier.prep_figure()
            markersize = 16
            markerz = 0

            for matr in materials:
                x, y, label, m, c, logx, logy, xlim, ylim, hits = matr

                c = colors[3] if 'Brittle' in label else colors[4]
                m = markers[0] if 'Brittle' in label else markers[1]

                fc = [c if hit else (0,0,0,0) for hit in hits]
                ec = [c if hit else (0,0,0,.2) for hit in hits]

                chart_classifier.ax.scatter(
                    x, 
                    y, 
                    label=label,
                    s=markersize,
                    marker=m,
                    fc=fc,
                    ec=ec,
                    zorder=markerz,
                )
            chart_classifier.plot()

    pretty_print_table( res ) # extract Table 2 data


if __name__=='__main__':
    classification_2D()