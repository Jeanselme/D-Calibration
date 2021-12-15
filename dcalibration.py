import numpy as np
from scipy.interpolate import interp1d

def dcalibration(proba_t, e, nbins = 20):
    """
    Compute the D calibration of the model
    (The lower the better)

    Args:
        proba_t (np.array n): Predicted survival for each patient at event time (ie S(t_i |x_i))
        e (np.array n): Indicator of observed event 
        nbins (int, optional): Number of bins to use. Defaults to 20.

    Returns:
        float: Mean Squared Error between expected percentage of patient in the bin and observed percentage
    """
    uncensored = proba_t[e != 0] # Only observed event
    observed, _ = np.histogram(uncensored, bins = nbins, range = (0, 1))
    return np.sum(np.power(observed / len(proba_t) - 1. / nbins, 2))

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
