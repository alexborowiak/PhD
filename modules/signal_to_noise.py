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
from typing import Optional, Union
import xarray_extender as xe
import os, sys
from dask.diagnostics import ProgressBar
# Custom xarray classes that addes different method.
import xarray_class_accessors as xca
import utils
from typing import List

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

def dask_percentile(array: np.ndarray, axis: str, q: float):
    '''
    Applies np.percetnile in dask across an axis
    Parameters:
    -----------
    array: the data to apply along
    axis: the dimension to be applied along
    q: the percentile
    
    Returns:
    --------
    qth percentile of array along axis
    
    Example
    -------
    xr.Dataset.data.reduce(xca.dask_percentile,dim='time', q=90)
    '''
#     array = array.rechunk({axis: -1})
    return array.map_blocks(
        np.percentile,
        axis=axis,
        q=q,
        dtype=array.dtype,
        drop_axis=axis)


def add_lower_upper_to_dataset(da: xr.DataArray, lower: xr.DataArray, upper:xr.DataArray) -> xr.Dataset:
    '''Convert the dataarray to dataset, then ddd the lower and upper bounds
    as a variable.'''
    ds = da.to_dataset(name='signal_to_noise')
    
    ds['lower_bound'] = lower
    ds['upper_bound'] = upper
    
    return ds


def global_sn(
    da: xr.DataArray, 
    control: xr.DataArray,
    da_loess: Optional[xr.DataArray] = None,
    control_loess: Optional[xr.DataArray] = None,
    window = 61, return_all = False,logginglevel='ERROR')-> xr.Dataset:
    '''
    Calculates the signal to noise for an array da, based upon the control.
    
    A full guide on all the functions used here can be found at in 02_gmst_analysis.ipynb
    
    Parameters
    ----------
    da: xr.DataArray
        input array the the signal to noise is in question for 
    control: xr.DataArray
        the control to compare with
    da_loess: Optional[xr.DataArray]
        loess filtered da
    control_loess: Optional[xr.DataArray]
        loess filtered control
    window = 61: 
        the window length
    return_all = False
        see below (return either 4 datasets or 9)
    logginglevel = 'ERROR'
    
    
    Note: 
    Returns 4 datasets: ds_sn, ds_sn_stable, ds_sn_increasing, ds_sn_decreasing
    
    
    But can be changed to return 9 datasets with return_all = True: 
                da_stable, da_increasing, da_decreasing, 
                ds_sn, d_sn_stable, ds_sn_increasing, ds_sn_decreasing, 
                control__lbound, control__ubound

    '''
    
    # Chaninging the logging level so that the info can be displayed if required.
    eval(f'logging.getLogger().setLevel(logging.{logginglevel})')

    
    logger.debug(f'- Input files\nda\n{da}\ncontrol\n{control}')

    #### Control
    
    
    # Singal
    logger.info('- Calculating control signal...')
    control_signal = control.sn.signal_grad(roll_period = window)
    logger.debug(f'\n{control_signal}')

    # Noise
    if control_loess is None:
        logger.info(f'- control_loess not provided. Calculating control noise')
        control_loess = control.sn.loess_grid()
        logger.debug(f'\n{control_loess}')

    control_noise = control_loess.sn.noise_grad(roll_period = window)

    # Signal to Noise
    logger.info(f'Calculating control signal to noise')
    control_sn = control_signal/control_noise
    logger.debug(f'\n{control_sn}')

    # The upper and lower bounds of what is stable.
    # TODO: Don't want to use max
    logger.info(f'- Upper and lower control bounds')
    
    
    # PHD-10
    # The bounds of what can be considered an increasing (90th percentile)
    # and a decreasing trend (10th percentile).
    try:
        control__ubound = control_sn.reduce(xe.dask_percentile,dim='time', q=99) #xe.dask_percentile
        control__lbound = control_sn.reduce(xe.dask_percentile,dim='time', q=1) 
        logger.info('Map blocks')
    
    # TODO: Sometimes will end up with numpy array instead of dask array, not sure
    # why. 
    except AttributeError as e:
        control__ubound = control_sn.reduce(np.nanpercentile,dim='time', q=99)
        control__lbound = control_sn.reduce(np.nanpercentile,dim='time', q=1)
        logger.info('np.nanpercentile')
    logger.debug(f'{control__lbound.values}  - {control__ubound.values}')


    ### Da
    logger.debug(f'- Experiment (da) file\n{da}')
    logger.info(f'- da signal')
    da_signal = da.sn.signal_grad(roll_period = window)
    logger.debug(f'{da_signal}')
    logger.info('- da loess')
    
    if da_loess is None:
        logger.debug(f'- da_loess not provided. Calculating control noise')
        da_loess = da.sn.loess_grid()
        
    logger.info(f'{da_loess}')
    logger.debug(f'- da noise')
    da_noise = da_loess.sn.noise_grad(roll_period = window)
    logger.debug(f'{da_noise}')
    logger.info('- da signal to noise')
    da_sn = da_signal/da_noise
    logger.debug(f'{da_sn}')


    # TEMP
    # The global temperature anomalies that are stable
    da_stable = da.where(np.logical_and(da_sn <= control__ubound,da_sn >= control__lbound))
    # Increasing temperature
    da_increasing = da.where(da_sn >= control__ubound )
    # Decreasing temperature.
    da_decreasing = da.where(da_sn <= control__lbound )

    # SN
    # The global signal-to-noise points that are stable
    da_sn_stable = da_sn.where(
        np.logical_and(da_sn <= control__ubound,da_sn >= control__lbound ))
    # Increasing temperature S/N
    da_sn_increasing = da_sn.where(da_sn >= control__ubound )
    # Decreasing temperature S/N
    da_sn_decreasing = da_sn.where(da_sn <= control__lbound )
    
    
    # Convering to dataset and adding lower and upper bounds as variable
    ds_sn = add_lower_upper_to_dataset(da_sn, control__lbound, control__ubound)
    ds_sn_stable = add_lower_upper_to_dataset(da_sn_stable, control__lbound, control__ubound)
    ds_sn_increasing = add_lower_upper_to_dataset(da_sn_increasing, control__lbound, control__ubound)
    ds_sn_decreasing = add_lower_upper_to_dataset(da_sn_decreasing, control__lbound, control__ubound)

    
    if return_all:
        return (da_stable, da_increasing, da_decreasing, 
                ds_sn, ds_sn_stable, ds_sn_increasing, ds_sn_decreasing, 
                control__lbound, control__ubound)

    return ds_sn, ds_sn_stable, ds_sn_increasing, ds_sn_decreasing

# TECH DEBT: THe global mean version works for versions with lat and lon coords. Can remove this version
global_mean_sn = global_sn


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
        da_sn, da_sn_stable, da_sn_increasing, da_sn_decreasing = global_mean_sn(da, control_da,window = window,
                                                                                logginglevel=logginglevel)
        
        increasing_sn_array.append(da_sn_increasing)
        decreasing_sn_array.append(da_sn_decreasing)
        stable_array.append(da_sn_stable)
    
    # Mergine the das together to form a an array witht he S/N values and a dim called window
    increasing_sn_multi_window_ds = xr.concat(increasing_sn_array, pd.Index(windows, name = 'window'))
    decreasing_sn_multi_window_ds = xr.concat(decreasing_sn_array, pd.Index(windows, name = 'window'))
    
    
    # Loading into memoery. 
    increasing_sn_multi_window_ds = increasing_sn_multi_window_ds.compute()
    decreasing_sn_multi_window_ds = decreasing_sn_multi_window_ds.compute()
    

    unstable_sn_multi_window_da  = increasing_sn_multi_window_ds.fillna(0) + decreasing_sn_multi_window_ds.fillna(0)

    unstable_sn_multi_window_da  = xr.where(unstable_sn_multi_window_da  != 0, unstable_sn_multi_window_da , np.nan)
    
    # Converting the time stamp to year.
    # TODO: Is this needed, it makes calculating with other things tricky as the timestamp has now
    # changed. 
    unstable_sn_multi_window_da['time'] = unstable_sn_multi_window_da.time.dt.year.values
#     unstable_sn_multi_window_da.name = 'SN'
    
    
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




def percent_of_non_nan_points_in_period(ds: xr.Dataset, period_list: List[tuple]) ->  xr.Dataset:
    '''
    Gets the percent of points that are non-non in different integer time periods.
    
    Parameters
    ----------
    ds: xr.Dataset
    periods: List[Tuple(int)]
        [(0,19), (20,39)]
        
    In the above example this will get the percent of points that are stable between
    years 0 and 19, 20 and 39.
    
    First used in zec_05
    
    '''
    xr_dict = {}
    for period in period_list:
        length = period[1] - period[0] + 1
        print(f'{period} - {length} years')
        name = str(period).replace('(','').replace(')','').replace(', ', '_')
        percent_stable = ds['signal_to_noise'].isel(time=slice(*period)).count(dim='time')
        percent_stable = percent_stable* 100/length
        percent_stable.name = 'percent'
        xr_dict[name] = percent_stable

    percent_ds = xr.concat(list(xr_dict.values()), pd.Index(list(xr_dict.keys()), name='period')).to_dataset()
    
    # The mean percent stable
    percent_mean_da = percent_ds['percent'].mean(dim='model')
    percent_mean_da.name = 'mean'

    # The uncertainty in perent stable
    percent_ucnertainty_da = (
        percent_ds['percent'].max(dim='model') - percent_ds['percent'].min(dim='model'))
    percent_ucnertainty_da.name = 'uncertainty'

    # Merge
    merged_ds = xr.merge([percent_mean_da, percent_ucnertainty_da, percent_ds])
    return merged_ds


