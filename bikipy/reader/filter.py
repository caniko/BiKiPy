"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import signal
from scipy.interpolate import CubicSpline


def convertparms2start(pn):
    """Creating a start value for sarimax in case of an value error
    See: https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk"""
    if "ar." in pn:
        return 0
    elif "ma." in pn:
        return 0
    elif "sigma" in pn:
        return 1
    else:
        return 0


def FitSARIMAXModel(x, p, pcutoff, alpha, ARdegree, MAdegree, nforecast=0, disp=False):
    # Seasonal Autoregressive Integrated Moving-Average with eXogenous regressors (SARIMAX)
    # see http://www.statsmodels.org/stable/statespace.html#seasonal-autoregressive-integrated-moving-average-with-exogenous-regressors-sarimax
    Y = x.copy()
    Y[p < pcutoff] = np.nan  # Set uncertain estimates to nan (modeled as missing data)
    if np.sum(np.isfinite(Y)) > 10:

        # SARIMAX implementation has better prediction models than simple ARIMAX (however we do not use the seasonal etc. parameters!)
        mod = sm.tsa.statespace.SARIMAX(
            Y.flatten(),
            order=(ARdegree, 0, MAdegree),
            seasonal_order=(0, 0, 0, 0),
            simple_differencing=True,
        )
        # Autoregressive Moving Average ARMA(p,q) Model
        # mod = sm.tsa.ARIMA(Y, order=(ARdegree,0,MAdegree)) #order=(ARdegree,0,MAdegree)
        try:
            res = mod.fit(disp=disp)
        except ValueError:  # https://groups.google.com/forum/#!topic/pystatsmodels/S_Fo53F25Rk (let's update to statsmodels 0.10.0 soon...)
            startvalues = np.array([convertparms2start(pn) for pn in mod.param_names])
            res = mod.fit(start_params=startvalues, disp=disp)
        except np.linalg.LinAlgError:
            # The process is not stationary, but the default SARIMAX model tries to solve for such a distribution...
            # Relaxing those constraints should do the job.
            mod = sm.tsa.statespace.SARIMAX(
                Y.flatten(),
                order=(ARdegree, 0, MAdegree),
                seasonal_order=(0, 0, 0, 0),
                simple_differencing=True,
                enforce_stationarity=False,
                enforce_invertibility=False,
                use_exact_diffuse=False,
            )
            res = mod.fit(disp=disp)

        predict = res.get_prediction(end=mod.nobs + nforecast - 1)
        return predict.predicted_mean, predict.conf_int(alpha=alpha)
    else:
        return np.nan * np.zeros(len(Y)), np.nan * np.zeros((len(Y), 2))


def columnwise_spline_interp(data, max_gap=0):
    """
    Perform cubic spline interpolation over the columns of *data*.
    All gaps of size lower than or equal to *max_gap* are filled,
    and data slightly smoothed.
    Parameters
    ----------
    data : array_like
        2D matrix of data.
    max_gap : int, optional
        Maximum gap size to fill. By default, all gaps are interpolated.
    Returns
    -------
    interpolated data with same shape as *data*
    """
    if np.ndim(data) < 2:
        data = np.expand_dims(data, axis=1)
    nrows, ncols = data.shape
    temp = data.copy()
    valid = ~np.isnan(temp)
    x = np.arange(nrows)
    for i in range(ncols):
        mask = valid[:, i]
        if np.sum(mask) > 3:  # Make sure there are enough points to fit the cubic spline
            spl = CubicSpline(x[mask], temp[mask, i])
            y = spl(x)
            if max_gap > 0:
                inds = np.flatnonzero(np.r_[True, np.diff(mask), True])
                count = np.diff(inds)
                inds = inds[:-1]
                to_fill = np.ones_like(mask)
                for ind, n, is_nan in zip(inds, count, ~mask[inds]):
                    if is_nan and n > max_gap:
                        to_fill[ind : ind + n] = False
                y[~to_fill] = np.nan
            # Get rid of the interpolation beyond the spline knots
            y[y == 0] = np.nan
            temp[:, i] = y
    return temp


def filter_data(
    df: pd.DataFrame,
    method: str,
    windowlength: int = 5,
    p_bound: float = 0.001,
    ARdegree: int = 3,
    MAdegree: int = 1,
    alpha: float = 0.01,
):
    nrows = df.shape[0]
    if method == "arima":
        temp = df.values.reshape((nrows, -1, 3))
        placeholder = np.empty_like(temp)
        for i in range(temp.shape[1]):
            x, y, p = temp[:, i].T
            meanx, _ = FitSARIMAXModel(x, p, p_bound, alpha, ARdegree, MAdegree, False)
            meany, _ = FitSARIMAXModel(y, p, p_bound, alpha, ARdegree, MAdegree, False)
            meanx[0] = x[0]
            meany[0] = y[0]
            placeholder[:, i] = np.c_[meanx, meany, p]
        data = pd.DataFrame(
            placeholder.reshape((nrows, -1)),
            columns=df.columns,
            index=df.index,
        )
    elif method == "median":
        data = df.copy()
        mask = data.columns.get_level_values("coords") != "likelihood"
        data.loc[:, mask] = df.loc[:, mask].apply(signal.medfilt, args=(windowlength,), axis=0)
    elif method == "spline":
        data = df.copy()
        mask_data = data.columns.get_level_values("coords").isin(("x", "y"))
        xy = data.loc[:, mask_data].values
        prob = data.loc[:, ~mask_data].values
        missing = np.isnan(xy)
        xy_filled = columnwise_spline_interp(xy, windowlength)
        filled = ~np.isnan(xy_filled)
        xy[filled] = xy_filled[filled]
        inds = np.argwhere(missing & filled)
        if inds.size:
            # Retrieve original individual label indices
            inds[:, 1] //= 2
            inds = np.unique(inds, axis=0)
            prob[inds[:, 0], inds[:, 1]] = 0.01
            data.loc[:, ~mask_data] = prob
        data.loc[:, mask_data] = xy
    else:
        raise ValueError(f"Unknown filter type {method}")

    return data
