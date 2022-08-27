''' Signal to noise

This package contains all the functions needed for calculating the signal to noise of timeseries.

This also included the calculatin of anomalies
TODOL which should either be moved to antoher module, or the name of 
this package module should perhaps be renamed.

'''

import numpy as np
import pandas as pd
import itertools
import xarray as xr
from typing import Optional
import os, sys
from dask.diagnostics import ProgressBar
# Custom xarray classes that addes different method.
import xarray_class_accessors as xca

import statsmodels.api as sm 
lowess = sm.nonparametric.lowess

# +
import logging

LOG_FORMAT = "%(message)s"
logging.basicConfig(format=LOG_FORMAT, filemode='w')
logger = logging.getLogger()

# Making the log message appear as a print statements rather than in the jupyter cells
logger.handlers[0].stream = sys.stdout


# -

# # Drafts

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
    
    print('Calculating signal...', end = '')
    
    # Getting the graident at each point with the rolling function. Then multipolying 
    # by the number of points to get the signal.
    signal = data.rolling(time = roll_period, min_periods = min_periods, center = True)\
        .reduce(apply_along_helper, func1d = grid_trend) * roll_period
    
    print('Done')
    print('Calculating loess filter...', end = '')
    
    # Loess filter
    loess = np.apply_along_axis(loess_filter, data.get_axis_num('time'), data.values, step_size = step_size)
    # loess = loess_filter(data.values, step_size = step_size)
    
    print('Done')
          
    # Detredning with the loess filer.
    loess_detrend = data - loess
    
    print('Calculating Noise...', end='')
    # The noise is the rolling standard deviation of the data that has been detrended with loess.
    noise = \
           loess_detrend.rolling(time = roll_period, min_periods = min_periods, center = True).std()
    
    print('Done')

    print('Calculating Signal to Noise with adjusttment...', end='')
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
    
    print('Done. \n Function complete - returning output')
    
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


def global_mean_sn(da: xr.DataArray, control: xr.DataArray, window = 61, return_all = False,
                  logginglevel='ERROR') -> xr.DataArray:
    '''
    Calculates the signal to noise for an array da, based upon the control.
    
    A full guide on all the functions used here can be found at in 02_gmst_analysis.ipynb
    
    Parameters
    ----------
    da: input array the the signal to noise is in question for 
    control: the control to compare with
    window = 61: the window length
    return_all = False - see below (return either 4 datasets or 9)
    logginglevel = 'ERROR'
    
    
    Note: 
    Returns 4 datasets: da_sn, da_sn_stable, da_sn_increasing, da_sn_decreasing
    
    
    But can be changed to return 9 datasets with return_all = True: 
                da_stable, da_increasing, da_decreasing, 
                da_sn, da_sn_stable, da_sn_increasing, da_sn_decreasing, 
                control_smean_lbound, control_smean_ubound
    da_sn_stable, da_sn_increasing, da_sn_decreasing, 
                control_smean_lbound, control_smean_ubound
    '''
    
    # Chaninging the logging level so that the info can be displayed if required.
    eval(f'logging.getLogger().setLevel(logging.{logginglevel})')
    #### Control
    
    logger.debug(f'Input files\nda\n{da}\ncontrol\n{control}')
    # Singal
    logger.debug('Calculating control signal...')
    control_signal = control.sn.signal_grad(roll_period = window)
    logger.debug(f'\n{control_signal}')

    # Noise
    logger.debug(f'Calculating control noise')
    control_loess = control.sn.loess_grid()
    logger.debug(f'\n{control_loess}')

    control_noise = control_loess.sn.noise_grad(roll_period = window)

    # Signal to Noise
    logger.debug(f'Calculating control signal to noise')
    control_sn = control_signal/control_noise
    logger.debug(f'\n{control_sn}')

    # The upper and lower bounds of what is stable.
    # TODO: Don't want to use max
    logger.debug(f'Upper and lower control bounds')
    control_smean_ubound = control_sn.reduce(np.nanpercentile,dim='time', q=99) # (xca.dask_percentile,dim='time', q=99) # .max(dim='time')
    control_smean_lbound = control_sn.reduce(np.nanpercentile,dim='time', q=1) #.min(dim='time')
    logger.debug(f'{control_smean_lbound.values}  - {control_smean_ubound.values}')


    ### Da
    logger.debug(f'Experiment (da) file\n{da}')
    logger.debug(f'da signal')
    da_signal = da.sn.signal_grad(roll_period = window)
    logger.debug(f'{da_signal}')
    logger.debug('da loess')
    da_loess = da.sn.loess_grid()
    logger.debug(f'{da_loess}')
    logger.debug(f'da noise')
    da_noise = da_loess.sn.noise_grad(roll_period = window)
    logger.debug(f'{da_noise}')
    logger.debug('da signal to noise')
    da_sn = da_signal/da_noise
    logger.debug(f'{da_sn}')


    # TEMP
    # The global mean temperature anomalies that are stable
    da_stable = da.where(np.logical_and(da_sn <= control_smean_ubound,da_sn >= control_smean_lbound))
    # Increasing temperature
    da_increasing = da.where(da_sn >= control_smean_ubound )
    # Decreasing temperature.
    da_decreasing = da.where(da_sn <= control_smean_lbound )

    # SN
    # The global mean signal-to-noise points that are stable
    da_sn_stable = da_sn.where(
        np.logical_and(da_sn <= control_smean_ubound,da_sn >= control_smean_lbound ))
    # Increasing temperature S/N
    da_sn_increasing = da_sn.where(da_sn >= control_smean_ubound )
    # Decreasing temperature S/N
    da_sn_decreasing = da_sn.where(da_sn <= control_smean_lbound )
    
    
    if return_all:
        return (da_stable, da_increasing, da_decreasing, 
                da_sn, da_sn_stable, da_sn_increasing, da_sn_decreasing, 
                control_smean_lbound, control_smean_ubound)
    return da_sn, da_sn_stable, da_sn_increasing, da_sn_decreasing


# +

def sn_multi_window(da, control_da, start_window = 21, end_window = 221, step_window = 8,
                  logginglevel='ERROR'):
    '''
    Calls the global_mean_sn function repeatedly for windows ranging betweent start_window
    and end_window with a step size of step_window.
    
    Parameters
    ----------
    
    da, control_da, start_window = 21, end_window = 221, step_window = 8
    
    
    Returns
    -------
    unstable_sn_multi_window_da , stable_sn_multi_window_da  Both these data sets contian dimension of time and window.
    '''
    
    decreasing_sn_array = []
    increasing_sn_array = []
    stable_array = []

    windows = range(start_window, end_window,step_window)
    
    print(f'Starting window loop from {start_window} to {end_window} with step size of {step_window}')
    # Looping through
    for window in windows:

        print(f'{window}, ', end='')
        da_sn, da_sn_stable, da_sn_increasing, da_sn_decreasing = global_mean_sn(da, control_da, window,
                                                                                logginglevel=logginglevel)
        
        increasing_sn_array.append(da_sn_increasing)
        decreasing_sn_array.append(da_sn_decreasing)
        stable_array.append(da_sn_stable)
    
    # Mergine the das together to form a an array witht he S/N values and a dim called window
    increasing_sn_multi_window_ds = xr.concat(increasing_sn_array, pd.Index(windows, name = 'window'))
    decreasing_sn_multi_window_ds = xr.concat(decreasing_sn_array, pd.Index(windows, name = 'window'))
    
    
    # Loading into memoery. 
    with ProgressBar():
        increasing_sn_multi_window_ds = increasing_sn_multi_window_ds.compute()
        
    with ProgressBar():
        decreasing_sn_multi_window_ds = decreasing_sn_multi_window_ds.compute()
    

    unstable_sn_multi_window_da  = increasing_sn_multi_window_ds.fillna(0) + decreasing_sn_multi_window_ds.fillna(0)

    unstable_sn_multi_window_da  = xr.where(unstable_sn_multi_window_da  != 0, unstable_sn_multi_window_da , np.nan)
    
    # Converting the time stamp to year.
    # TODO: Is this needed, it makes calculating with other things tricky as the timestamp has now
    # changed. 
    unstable_sn_multi_window_da['time'] = unstable_sn_multi_window_da.time.dt.year.values
    unstable_sn_multi_window_da.name = 'SN'
    
    
    stable_sn_multi_window_da  = xr.where(np.isfinite(unstable_sn_multi_window_da ), 1, 0)
    
    return unstable_sn_multi_window_da , stable_sn_multi_window_da 


# -

def number_finite(da: xr.DataArray, dim:str='model') -> xr.DataArray:
    '''
    Gets the number of points that are finite .
    The function gets all points that are finite across the dim 'dim'.
    
    Paramaters
    ----------
    da: xr.Dataset or xr.DataArray (ds will be converted to da). This is the dataset 
        that the number of finite points across dim.
    number_present: xr.DataArray - the max number of available observations at each timestep
    dim: str - the dimension to sum finite points across
    
    Returns
    ------
    da: xr.DataArray - the fraction of finite points.
    
    '''
    
    # If da is a dataset, we are going to convert to a data array.
    if isinstance(da, xr.Dataset):
        da = da.to_array(dim=dim)
    
    # The points that are finite become1 , else 0
    finite_da = xr.where(np.isfinite(da), 1, 0)
    # Summing the number of finite points.
    number_da = finite_da.sum(dim)
    
    return number_da


def percent_finite(da, number_present: xr.DataArray, dim:str='model') -> xr.DataArray:
    '''
    Gets the percent of points that are finite based upon the number of available models.
    The function gets all points that are finite across the dim 'dim'.
    
    Paramaters
    ----------
    da: xr.Dataset or xr.DataArray (ds will be converted to da). This is the dataset 
        that the number of finite points across dim.
    number_present: xr.DataArray - the max number of available observations at each timestep
    dim: str - the dimension to sum finite points across
    
    Returns
    ------
    da: xr.DataArray - the fraction of finite points.
    
    '''
    
    number_da = number_finite(da, dim)
    
    # Converting to a percent of the max number of finite points possible.
    da = number_da * 100/number_present
    
    # Renaming the da with percennt and dim.
    da.name = f'percent_of_{dim}'
    
    return da


def count_over_data_vars(ds: xr.Dataset, data_vars: list = None, dim='model') -> xr.DataArray:
    '''
    Counts the number of data vars that are present. 
    
    Parameters
    ----------
    ds (xr.Dataset): the dataset to count over
    data_vars (list): the data vars that need to be coutned.
    dim (str): the dimenesion to be counted over
    
    Returns
    -------
    number_da (xr.Dataarray): the number of occurences accross the data vars
    
    '''
    
    # If data_vars is none then we want all the data vars from out dataset
    if data_vars is None:
        data_vars = ds.data_vars
    
    # Subsetting the desired data vars and then counting along a dimenstion. 
    da = ds[data_vars].to_array(dim=dim) 
    # This is the nubmer of models peresent at each timestep.
    number_da = da.count(dim=dim)
    # In the multi-window function, time has been changed to time.year, so must be done here as well
    # Note: This may be removed in future.
    number_da['time'] = ds.time.dt.year
    number_da.name = f'number_of_{dim}'
    return number_da
