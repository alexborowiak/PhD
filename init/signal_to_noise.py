''' Signal to noise

This package contains all the functions needed for calculating the signal to noise of timeseries.

This also included the calculatin of anomalies, which should either be moved to antoher module, or the name of 
this package module should perhaps be renamed.

'''

import numpy as np
import pandas as pd
import itertools
import xarray as xr
import os

import statsmodels.api as sm 
lowess = sm.nonparametric.lowess
    
    
def climatology(hist: xr.Dataset, start = 1850, end = 1901):
    '''
    CLIMATOLOGY
    Getting just the years for climatology. This should be for each pixel, the mean temperature
    from 1850 to 1900.
    
    Parameters
    ----------
    hist: xarray dataset with dimension time
    start: float/int of the start year.
    end: float/ind of the end year
    
    Returns:
    climatologyL xarray dataset with the mean of the time dimension for just the years from 
    start to end. Still contains all other dimensions (e.g. lat and lon) if passed in.
    
    '''
    climatology = hist.where(hist.time.dt.year.isin(np.arange(start,end)), drop = True)\
                        .mean(dim = 'time')

    return climatology

# TODO: Need kwargs for this.
def anomalies(data, hist):

    climatology = climatology(hist)

    data_resampled = data.resample(time = 'Y').mean()
    data_anom = (data_resampled - climatology).chunk({'time':8})


    return data_anom

def space_mean(data: xr.Dataset):
    '''
    When calculating the space mean, the mean needs to be weighted by latitude.

    Parameters
    ----------
    data: xr.Dataset with both lat and lon dimension

    Returns
    -------
    xr.Dataset that has has the weighted space mean applied.
    '''
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





def loess_filter(y: np.array, step_size = 10):
    
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
    
    return yhat[:,1]


def sn_grad_loess(data,
                  roll_period = 60, 
                  step_size = 60, 
                  min_periods = 0,
                  verbose = 0, 
                  return_all = 0, 
                  unit = 'y') -> xr.DataArray:
    
    '''
    This function applies rolling calculatin and several of the other functions found in signal
    to nosie: loess filer and apply_along_help with grid_trend
    Parameters
    ----------
    data: xr.Dataset or xr.DataArray with one variables. Either is fine, however Dataset will
          be converted to Dataarray.
    roll_period: The winodw of the rolling.
    step_size: the number of points that will go into each loess filter.
    min_periods: this is the minimum number of points the xarray can take. If set to zero
                 then the min_periods will be the roll_period.
    verbose: TODO
    return_all: returns all data calculated here. Otherwise will just return sn.
    unit: this is the unit when shifting the time backwards for the sn. 
    
    '''
    
    # If Datatset then convert to DataArray.
    if isinstance(data, xr.Dataset):
        data = data.to_array()
    
    # If no min_periods, then min_periods is just roll_period.
    if ~min_periods:
        min_periods = roll_period
    
    # Getting the graident at each point with the rolling function. Then multipolying 
    # by the number of points to get the signal.
    signal = data.rolling(time = roll_period, min_periods = min_periods, center = True)\
        .reduce(apply_along_helper, func1d = grid_trend) * roll_period
    
    # Loess filter
    loess = loess_filter(data.values, step_size = step_size)
    
    # Detredning with the loess filer.
    loess_detrend = data - loess
    
    # The noise is the rolling standard deviation of the data that has been detrended with loess.
    noise = \
           loess_detrend.rolling(time = roll_period, min_periods = min_periods, center = True).std()
    
    
    # Signal/Noise.
    sn = signal/noise    
    sn.name = 'S/N'
    
    # This will get rid of all the NaN points on either side that arrises due to min_periods.
    sn = sn.isel(time = slice(
                               int((roll_period - 1)/2),
                                -int((roll_period - 1)/2)
                              )
                )
    
    # We want the time to match what the data is (will be shifter otherwise).
    sn['time'] = data.time.values[:len(sn.time.values)]

    
    # Sometimes a new coord can be created, so all data is returned with squeeze.
    if return_all:
        return sn.squeeze(), signal.squeeze(), noise.squeeze(), loess.squeeze(), loess_detrend
    
    return sn.squeeze()



def sn_grad_loess_grid(data,
                  roll_period = 60, 
                  step_size = 60, 
                  min_periods = 0,
                  verbose = 0, 
                  return_all = 0, 
                  unit = 'y') -> xr.DataArray:
    
    '''
    This function applies rolling calculatin and several of the other functions found in signal
    to nosie: loess filer and apply_along_help with grid_trend
    Parameters
    ----------
    data: xr.Dataset or xr.DataArray with one variables. Either is fine, however Dataset will
          be converted to Dataarray.
    roll_period: The winodw of the rolling.
    step_size: the number of points that will go into each loess filter.
    min_periods: this is the minimum number of points the xarray can take. If set to zero
                 then the min_periods will be the roll_period.
    verbose: TODO
    return_all: returns all data calculated here. Otherwise will just return sn.
    unit: this is the unit when shifting the time backwards for the sn. 
    
    '''
    
    # If Datatset then convert to DataArray.
    if isinstance(data, xr.Dataset):
        data = data.to_array()
    
    # If no min_periods, then min_periods is just roll_period.
    if ~min_periods:
        min_periods = roll_period
    
    # Getting the graident at each point with the rolling function. Then multipolying 
    # by the number of points to get the signal.
    signal = data.rolling(time = roll_period, min_periods = min_periods, center = True)\
        .reduce(apply_along_helper, func1d = grid_trend) * roll_period
    
    # Loess filter
    loess = np.apply_along_axis(loess_filter, data.get_axis_num('time'), data.values, step_size = step_size)
#     loess = loess_filter(data.values, step_size = step_size)
    
    # Detredning with the loess filer.
    loess_detrend = data - loess
    
    # The noise is the rolling standard deviation of the data that has been detrended with loess.
    noise = \
           loess_detrend.rolling(time = roll_period, min_periods = min_periods, center = True).std()
    
    
    # Signal/Noise.
    sn = signal/noise    
    sn.name = 'S/N'
    
    # This will get rid of all the NaN points on either side that arrises due to min_periods.
    sn = sn.isel(time = slice(
                               int((roll_period - 1)/2),
                                -int((roll_period - 1)/2)
                              )
                )
    
    # We want the time to match what the data is (will be shifter otherwise).
    sn['time'] = data.time.values[:len(sn.time.values)]

    
    # Sometimes a new coord can be created, so all data is returned with squeeze.
    if return_all:
        return sn.squeeze(), signal.squeeze(), noise.squeeze(), loess.squeeze(), loess_detrend
    
    return sn.squeeze()



def consecutive_counter(data: np.array) -> np.array:
    '''
    Calculates two array. The first is the start of all the instances of 
    exceeding a threshold. The other is the consecutive length that the 
    threshold.
    TODO: Need to adds in the rolling timeframe. The data is not just unstable
    starting at a specific point, but for the entire time. 
    
    Parameters
    ----------
    data: np.ndarray
          Groups of booleans.
    
    Returns
    -------
    consec_start: An array of all start times of consecuitve sequences.
    consec_len: The length of all the exceedneces.
    
    TODO: Could this be accelerated with numba.njit???? The arrays will 
    always be of unkonw length.
    '''
    condition = data
    #condition = data >= stable_bound

    consec_start_arg = []
    consec_len = []
    
    # Arg will keep track of looping through the list.
    arg = 0

    # This loop will grup the array of Boleans together.  Key is the first value in the
    # group and group will be the list of similar values.
    for key, group in itertools.groupby(condition):

        # Consec needs to be defined here for the arg
        consec = len(list(group))

        if key:
            consec_start_arg.append(arg)
            consec_len.append(consec)

        arg += consec

    return np.array(consec_start_arg), np.array(consec_len)