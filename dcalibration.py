import numpy as np
from scipy.interpolate import interp1d

def dcalibration(proba_t, e, nbins = 20):
    """
    Compute the D calibration of the model
    (The lower the better)

    Formula 49 in https://arxiv.org/pdf/1811.11347.pdf

    Args:
        proba_t (np.array n): Predicted survival for each patient at event time (ie S(t_i |x_i))
        e (np.array n): Indicator of observed event 
        nbins (int, optional): Number of bins to use. Defaults to 20.

    Returns:
        float: Mean Squared Error between expected percentage of patient in the bin and observed percentage
    """
    # Censored computation
    count, bins = np.histogram(proba_t[e != 0], bins = nbins, range = (0, 1))

    # Weighting of each cell takes into account censoring
    assigned = np.digitize(proba_t[e == 0], bins = bins) - 1
    weights = [(1 - bins[i] / proba_t[e == 0][assigned == i]).sum() if (assigned == i).sum() > 0 else 0 for i in np.arange(nbins)]

    # Impact on all previous cell
    blur = [(1 / (nbins * proba_t[e == 0][assigned == i])).sum() if (assigned == i).sum() > 0 else 0 for i in np.arange(nbins)]
    blur = np.cumsum(np.insert(blur[::-1], 0, 0))[::-1][:-1]

    bins = count + weights + blur
    return np.sum(np.power(bins / len(proba_t) - 1. / nbins, 2))

def predict_t(survival, times):
    """
    Interpolate the survival function to estimate the survival at time t

    Args:
        survival (Dataframe T * n): Survival prediction at all time in T (index)
        times (np.array n): Time at which to estimate survival for each point
    """
    res = []
    for i, t in zip(survival.columns, times):
        res.append(interp1d(survival.index, survival[i].values, fill_value = (1, survival[i].values[-1]), bounds_error = False)(t))
    return np.array(res)
