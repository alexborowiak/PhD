''' Signal to noise

This package contains all the functions needed for calculating the signal to noise of timeseries.

This also included the calculatin of anomalies, which should either be moved to antoher module, or the name of 
this package module should perhaps be renamed.

'''

import numpy as np
import pandas as pd
import xarray as xr
import os

    
def climatology(hist: xr.Dataset, start = 1850, end = 1901):

    ### CLIMATOLOGY

    # Getting just the years for climatology. This should be for each pixel, the mean temperature
    # from 1850 to 1900. 
    climatology = hist.where(hist.time.dt.year.isin(np.arange(start,end)), drop = True)\
                        .mean(dim = 'time')

    return climatology


def anomalies(data, hist):

    climatology = climatology(hist)

    data_resampled = data.resample(time = 'Y').mean()
    data_anom = (data_resampled - climatology).chunk({'time':8})


    return data_anom

def space_mean(data):
    
        # Lat weights
        weights = np.cos(np.deg2rad(data.lat))
        weights.name= 'weights'
        
        # Calculating the weighted mean.
        data_wmean = data.weighted(weights).mean(dim = ['lat','lon'])

        return data_wmean    


def grid_trend(x, use = [0][0]):
    
    
    '''
    Parameters
    ----------
    x: the y values of our trend
    use: 
    [0][0] will just return the gradient
    [0,1] will return the gradient and y-intercept.
    '''
    if all(~np.isfinite(x)):
        return np.nan
    
    # If every point is just a nan values. We don't want to do the polyfit calculation. Just return nan
    t = np.arange(len(x))

    # Getting the gradient of a linear interpolation
    idx = np.isfinite(x) #checking where the nans.
    x = x[idx]
    t = t[idx]
    
    if len(x) < 3:
        return np.nan
    
    poly = np.polyfit(t,x,1)
    
    return poly[use]




def grid_noise_detrend(y):
    x = np.arange(len(y))

    # Getting the gradient of a linear interpolation
    idy = np.isfinite(y) #checking where the nans.
    y = y[idy]
    x = x[idy]
    
    if len(y) < 10:
        return np.nan
    
    m,c = np.polyfit(x,y,1)

    trend_line = m * x + c

    y_detrend = y - trend_line

    std_detrend = np.std(y_detrend)
    
    return std_detrend


def apply_along_helper(arr, axis, func1d):
    '''
    Parameters
    -------
    arr : an array
    axis: the axix to apply the grid_noise function along
    
    
    Example
    --------
    >>> ipsl_anom_smean.rolling(time = ROLL_PERIOD, min_periods = MIN_PERIODS, center = True)\
    >>>    .reduce(apply_along_helper, grid_noise_detrend)
    '''
        
    # func1ds, axis, arr 
    return np.apply_along_axis(func1d, axis[0], arr)





def loess_filter(data: np.array, step_size = 10):
    
    '''
    Applies the loess filter to a 1D numpy array.
    
    Parameters
    -----------
    data: the 1D array of values to apply the loess filter to
    step_size: the number of steps in each of the loess filter. The default is 50 points 
    in each window.
    
    Returns
    -------
    yhat: the data but, the loess version.
    
    Example
    -------
    >>> mean_temp = data.temp.values
    >>> mean_temp_loess = loess_filter(mean_temp)
    >>> 
    >>> # The mean temperature that has been detrended using the loess method.
    >>> mean_temp_loess_detrend = mean_temp - mean_temp_loess
    
    '''
    
    # Removign the nans (this is important as if two dataarrays where together in dataset
    # one might have been longer than the other, leaving a trail of NaNs at the end.)
    idy = np.isfinite(y)
    y = y[idy]
    
    # The equally spaced x-values.
    x =  np.arange(len(y))
    
    
    # The fraction to consider the linear trend of each time.
    frac = step_size/len(y)
    
    #yhat is the loess version of y - this is the final product.
    yhat = lowess(y, x, frac  = frac)
    
    return yhat

