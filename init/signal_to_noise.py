import numpy as np
import pandas as pd
import xarray as xr
import os

# This package is for the vaious methods of calculating singal to noise.

# grid trend:
    # Calculates the trend at each grid cell

# trend_help
    # Applies the grid trend along an axis.
    
# grid_noise
    # the lineaerly detrend noise (standard deviation)
    
# grid_noise_helper
    # applies the grid_noise at each grid cell in an array.

    
    
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




