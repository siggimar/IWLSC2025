import numpy as np
from scipy import stats
from figure_formatting import markers, colors
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator


def z_vals_from_interp( some_interpolator_class, X, Y, Z, xy):
    # combine flattened X and Y into single array of points
    points = np.column_stack( (X.flatten(), Y.flatten()) )

    # fit interpolator to meshgrid
    some_interpolator = some_interpolator_class( points, Z.flatten() )

    # use interpolator to estimate z values points
    z = some_interpolator( xy )

    # extract x and y from XY list
    x = np.array( [p[0] for p in xy] )
    y = np.array( [p[1] for p in xy] )

    x_, y_ = np.array(xy).T
    return x, y, z


def interpolate_RBF(X, Y, Z, xy):
    return z_vals_from_interp( RBFInterpolator, X, Y, Z, xy )


def interpolate_griddata( X, Y, Z, xy ):
    return z_vals_from_interp( LinearNDInterpolator, X, Y, Z, xy )


def combined_log_mask( x, y, logx, logy):
    x_, y_ = np.array(x), np.array(y)
    mask = (x_ > 0 if logx else np.ones_like(x_, dtype=bool)) & (y_ > 0 if logy else np.ones_like(y_, dtype=bool))
    return x_[mask], y_[mask]


def data_to_kde( x, y, logx=True, logy=True, N=100, xlims=(1e-1,1e4), ylims=(1e-3,1e2) ):
    ''' function to calculate data meshgrid '''
    n_pts = N # increments in meshgrid

    x_, y_ = combined_log_mask( x, y, logx, logy)

    if logx: x_ = np.log10( x_ )
    if logy: y_ = np.log10( y_ )

    xlims = xlims if not logx else (np.log10(xlims[0]),np.log10(xlims[1]))
    ylims = ylims if not logy else (np.log10(ylims[0]),np.log10(ylims[1]))

    xmin, xmax, ymin, ymax = x_.min(), x_.max(), y_.min(), y_.max()
    xmin, xmax, ymin, ymax = xlims[0], xlims[1], ylims[0], ylims[1]

    X, Y = np.mgrid[xmin:xmax:(n_pts*1j), ymin:ymax:(n_pts*1j)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack( [x_, y_] )

    # bandwidth selection method by Scott, 1992: "Multivariate Density Estimation: ..."
    kernel = stats.gaussian_kde( values ) 
    Z = np.reshape(kernel(positions).T, X.shape)

    return X, Y, Z, x_, y_ # also returns points from estimation (in lin/log scale)


def get_kde_models( materials, SBT=None, N=200 ):
    kdes = []

    if SBT:
        xlim = 1

    for matr in materials:
        x, y, label, m, c, logx, logy, xlim, ylim = matr
        kdes.append( data_to_kde(
            x=x,
            y=y,
            logx=logx,
            logy=logy,
            N=N,
            xlims=xlim,
            ylims=ylim
            )
        )
    return kdes


if __name__=='__main__':
    pass