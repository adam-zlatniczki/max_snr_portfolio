from scipy.spatial import cKDTree
import numpy as np
from numpy import log, polyfit, sqrt, std, subtract, log10
from sklearn.covariance import MinCovDet
from scipy.special import comb
from hurst import compute_Hc


def fractal_dim(series, agg=np.median):
    x = np.arange(0.0, series.shape[0], 1.0) / series.shape[0]
    y = (series - min(series)) / (max(series) - min(series))

    manifold = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    tree = cKDTree(manifold)

    k = int(np.ceil(np.sqrt(manifold.shape[0])))
    distances, _ = tree.query(manifold, k+1, workers=-1)

    local_dims = np.log(2) / np.log(distances[:, k] / distances[:, k//2])

    return agg(local_dims)

def hurst_from_dim(series, agg=None):
    if agg is None:
        agg = np.median

    return 2 - fractal_dim(series, agg)

def hurst_var(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 10 * int(log10(ts.shape[0])))

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

def hurst(ts, lags=None, robust_cov=True, plot=False):
    if lags is None:
        lags = 10 * int(np.log10(ts.shape[0]))

    minCovDet = MinCovDet(assume_centered=True, random_state=0)
    n = ts.shape[0]

    # calculate lagged variances
    var_lags = np.zeros(lags-1)

    for lag in range(1,lags):
        lagged_series = ts[lag:]-ts[:-lag]

        if robust_cov:
            minCovDet.fit(lagged_series.reshape(-1,1))
            var_lags[lag-1] = minCovDet.covariance_.item()
        else:
            var_lags[lag-1] = np.dot(lagged_series, lagged_series) / (n - lag - 1)

    # calculate log-log slopes
    slopes = np.zeros(int(comb(lags-2,2)))
    cntr = 0
    for i in range(1,lags-1):
        for j in range(i+1,lags-1):
            slopes[cntr] = np.log(var_lags[j] / var_lags[i]) / (2 * np.log(float(j) / i))
            cntr += 1

    return np.median(slopes)

def evaluate_weights(weights, cumul_returns_df):
    portfolio_cumul_returns = np.dot(cumul_returns_df, weights)

    H_estimates = [
        compute_Hc(portfolio_cumul_returns, kind="random_walk", simplified=False)[0],
        hurst(portfolio_cumul_returns, robust_cov=True),
        hurst_var(portfolio_cumul_returns),
        hurst_from_dim(portfolio_cumul_returns)
    ]

    return H_estimates