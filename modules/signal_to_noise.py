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

logger = utils.get_notebook_logger()



def calculate_ice_earth_fraction(ds: xr.Dataset) -> xr.Dataset:
    '''Calculates the fraction of the earth that is covered in ice for each time step'''
    ocean_as_1_ds = xr.where(np.isfinite(ds), 1, 0).isel(time=0)
    global_frac_ds = ds.sum(dim=['lat', 'lon'])/ocean_as_1_ds.sum(dim=['lat', 'lon'])
    return global_frac_ds

def calculate_global_value(ds: xr.Dataset, control_ds: xr.Dataset, variable:str):
    '''Calculates anomalies and mean.'''
    if variable == 'sic':
        ds_mean = calculate_ice_earth_fraction(ds)
        control_mean = calculate_ice_earth_fraction(control_ds)

    else:


        # Space mean and anomalmies
        ds_anom = ds.clima_ds.anomalies(historical_ds=control_ds)
        
        control_mean = control_ds.clima_ds.space_mean()
        ds_mean = ds_anom.clima_ds.space_mean() 
    
    return ds_mean.compute(), control_mean.compute()


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



def calculate_upper_and_lower_bounds(ds: xr.Dataset, lower_bound:float = 1, upper_bound: float = 99, 
                                    logginglevel='ERROR'):
    
    utils.change_logging_level(logginglevel)
    logger.info(f'Calculating Upper and lower control bounds')
   
    try:
        control__ubound = ds.reduce(xe.dask_percentile,dim='time', q=upper_bound)
        control__lbound = ds.reduce(xe.dask_percentile,dim='time', q=lower_bound) 
        logger.info('Map blocks used')
    
    except AttributeError as e:
        control__ubound = ds.reduce(np.nanpercentile,dim='time', q=upper_bound)
        control__lbound = ds.reduce(np.nanpercentile,dim='time', q=lower_bound)
        logger.info('np.nanpercentile used')
        
    logger.debug(f'{control__lbound.values}  - {control__ubound.values}')
    return (control__lbound, control__ubound)




# def get_ds_above_below_and_between_bounds(da: xr.DataArray, da_sn: xr.DataArray, 
#                                           control__lbound: xr.DataArray, control__ubound: xr.DataArray) -> xr.DataArray:
    
#     da_stable = da.where(np.logical_and(da_sn <= control__ubound,da_sn >= control__lbound))
#     da_increasing = da.where(da_sn >= control__ubound )
#     da_decreasing = da.where(da_sn <= control__lbound )
    
#     return (da_stable, da_increasing, da_decreasing)



def calculate_rolling_signal_to_noise(window: int, da: xr.DataArray, lowess_filter:bool=True, 
                                     logginglevel='ERROR') -> xr.DataArray:
    '''
    Window first for multiprocessing reasons.
    Calculates the rolling signal to nosie with an optional lowess filter.
    '''
    utils.change_logging_level(logginglevel)
    print(f'{window}, ', end='')
    signal_da = da.sn.calculate_rolling_signal(window=window, logginglevel=logginglevel)
    
    if lowess_filter:
        logger.info('Appyling lowess filter')
        da = da.sn.apply_loess_filter()
    noise_da = da.sn.calculate_rolling_noise(window=window, logginglevel=logginglevel)
    
    logger.info('Calculating signal to noise')
    sn_da = signal_da/noise_da
    
    sn_da.name = 'signal_to_noise'
    
    return sn_da



def synchronous_calculate_multi_window_signal_to_noise(da: xr.DataArray,
                    windows: tuple, lowess_filter:bool=True,  logginglevel='ERROR'):
    
    '''Synchronously calculates signal to noise'''
    to_concat = []
    
    for window in windows:            
        sn_da = calculate_rolling_signal_to_noise(da = da, window=window,
                                                  lowess_filter=lowess_filter,
                                                  logginglevel=logginglevel)
        to_concat.append(sn_da)
        
    return to_concat


def parallel_calculate_multi_window_signal_to_noise(da: xr.DataArray,
                    windows: tuple, lowess_filter:bool=True, logginglevel='ERROR'):
    
    from functools import partial
    from multiprocessing import Pool
    
    partial_calculate_rolling_signal_to_noise = partial(calculate_rolling_signal_to_noise,
                                                        da=da, lowess_filter=lowess_filter)
    with Pool() as pool:
        to_concat = pool.map(partial_calculate_rolling_signal_to_noise, windows)
        
    return to_concat



def calculate_multi_window_signal_to_noise(da: xr.DataArray, lowess_filter:bool=True,
                    start_window = 21, end_window = 221, step_window = 8, parallel=False,
                    logginglevel='ERROR'):
    
    '''Calcualtes the signal to noise for a range of different window lengths, either in parallel
    or synchronously.'''
    
    # Make sure da is computed before starting
    da = da.compute()
    windows = range(start_window, end_window, step_window)
    
    if not parallel:
        to_concat = synchronous_calculate_multi_window_signal_to_noise(da=da, lowess_filter=lowess_filter,
                                                            windows=windows, logginglevel=logginglevel)
    else:
        to_concat = parallel_calculate_multi_window_signal_to_noise(da=da, lowess_filter=lowess_filter,
                                                            windows=windows, logginglevel=logginglevel)

        
    da_multi_window = xr.concat(to_concat, dim='window')
    
    da_multi_window = da_multi_window.sortby('window')
    
    return da_multi_window   



def calculate_rolling_signal_to_nosie_and_bounds(
                    experiment_da: xr.DataArray, control_da: xr.DataArray,
                    start_window = 21, end_window = 221, step_window = 8, parallel=False,
                    logginglevel='ERROR') -> xr.Dataset:
    
    '''
    Calculates the siganl to nosie for experiment_da and control_da. The signal to noise for
    control_da is then usef ot calculate lbound, and uboud. These bounds are then added to sn_ds, 
    to make a dataset with vars: signal_to_noise, lower_bound, upper_bound.
    '''
    
    # This is to be slotted into sn_multi_window
    print('\nExperiment\n--------\n')
    experiment_da_sn = calculate_multi_window_signal_to_noise(da=experiment_da, lowess_filter=True,
                            start_window=start_window, end_window=end_window, step_window = step_window,
                                                              parallel=parallel, logginglevel=logginglevel)
    print('\nControl\n------\n')
    control_sn = calculate_multi_window_signal_to_noise(da=experiment_da, lowess_filter=False,
                            start_window=start_window, end_window=end_window, step_window = step_window,
                                                              parallel=parallel, logginglevel=logginglevel)
    
    lower_bound, upper_bound = calculate_upper_and_lower_bounds(control_sn)    
    
    sn_multiwindow_ds = xr.merge([experiment_da_sn.to_dataset(), 
                  lower_bound.to_dataset(name='lower_bound'), 
                  upper_bound.to_dataset(name='upper_bound')], 
                compat='override')
    
    return sn_multiwindow_ds

def sn_multi_window(
                    experiment_da: xr.DataArray, control_da: xr.DataArray,
                    start_window = 21, end_window = 61, step_window = 2, parallel=False,
                    logginglevel='ERROR') -> xr.Dataset:
    
    sn_multiwindow_ds = calculate_rolling_signal_to_nosie_and_bounds(
                            experiment_da=experiment_da, control_da=control_da,
                            start_window=start_window, end_window=end_window, step_window = step_window,
                            parallel=parallel, logginglevel=logginglevel)
    
    
    unstable_sn_multi_window_da = sn_multiwindow_ds.utils.above_or_below(
        'signal_to_noise', greater_than_var = 'upper_bound', less_than_var = 'lower_bound')
    
    stable_sn_multi_window_da = sn_multiwindow_ds.utils.between(
        'signal_to_noise', less_than_var = 'upper_bound', greater_than_var = 'lower_bound')
    
    return unstable_sn_multi_window_da , stable_sn_multi_window_da
    


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




def percent_of_non_nan_points_in_period(ds: xr.Dataset, period_list: List[tuple], logginglevel='ERROR') ->  xr.Dataset:
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
    utils.change_logging_level(logginglevel)
    xr_dict = {}
    for period in period_list:
        length = period[1] - period[0] + 1
        logger.info(f'{period} - {length} years')
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


def pattern_correlation_of_models(da: xr.DataArray, logginglevel='ERROR') -> pd.DataFrame:
    '''
    Calculatees the pattern correlation between a model and all the
    mean of all the other models. 
    
    Parameters
    -----------
    da: xr.DataArray
        Must contain the coordinates 'model' and 'period'
    Returns
    -------
    df: pd.DataFrame
        Pandas dataframe with columns as the period and the rows as the model
    
    
    '''
    from scipy.stats import spearmanr
    
    utils.change_logging_level(logginglevel)
    
    models = da.model.values
    periods = da.period.values
    
    # Loop through all models and period
    pattern_corr_dict = {}
    for period in periods:
        logger.info(f'{period=}')
        period_dict = {}
        for model in models:
            logger.info(f'{model=}')

            # Mean of all model but the model in question
            da_drop_model = da.where(da.model.isin(models[models != model]), drop=True)
            da_drop_model_period = da_drop_model.sel(period = period)
            da_mean = da_drop_model_period.mean(dim='model').values.flatten()
            
            # The model in question
            da_single_model = da.sel(model=model, period=period)
            percent_np = da_single_model.values.flatten()

            pattern_corr = spearmanr(percent_np, da_mean)
                
            # Just the correlation
            period_dict[model] = pattern_corr[0]
        pattern_corr_dict[period] = period_dict
    df = pd.DataFrame(pattern_corr_dict)
    return df



# def global_sn(
#     da: xr.DataArray,  control: xr.DataArray,
#     da_loess: Optional[xr.DataArray] = None,
# #     control_loess: Optional[xr.DataArray] = None, # !!!! Control data should not be loess filtered
#     window = 61, return_all = False, logginglevel='ERROR')-> xr.Dataset:
#     '''
#     Calculates the signal to noise for an array da, based upon the control.
    
#     A full guide on all the functions used here can be found at in 02_gmst_analysis.ipynb
    
#     Parameters
#     ----------
#     da: xr.DataArray
#         input array the the signal to noise is in question for 
#     control: xr.DataArray
#         the control to compare with
#     da_loess: Optional[xr.DataArray]
#         loess filtered da
#     control_loess: Optional[xr.DataArray]
#         loess filtered control
#     window = 61: 
#         the window length
#     return_all = False
#         see below (return either 4 datasets or 9)
#     logginglevel = 'ERROR'
    
    
#     Note: 
#     Returns 4 datasets: ds_sn, ds_sn_stable, ds_sn_increasing, ds_sn_decreasing
    
    
#     But can be changed to return 9 datasets with return_all = True: 
#                 da_stable, da_increasing, da_decreasing, 
#                 ds_sn, d_sn_stable, ds_sn_increasing, ds_sn_decreasing, 
#                 control__lbound, control__ubound

#     '''
    
#     utils.change_logging_level(logginglevel)
#     logger.info('Calculating signal to noise')
    
#     # !!!!!! You should not be loess filter the control data!!!!!
#     #if control_loess is None:
#     #   logger.info(f'- control_loess not provided.')
#     #    control_loess = control.sn.apply_loess_filter()
        
#     control_signal = control.sn.calculate_rolling_signal(window = window, logginglevel=logginglevel)
#     control_noise = control.sn.calculate_rolling_noise(window = window, logginglevel=logginglevel) #control_loess
#     control_sn = control_signal/control_noise
#     control__lbound, control__ubound = calculate_upper_and_lower_bounds(control_sn)


#     if da_loess is None:
#         logger.debug(f'- da_loess not provided')
#         da_loess = da.sn.apply_loess_filter()
        
        
#     da_signal = da.sn.calculate_rolling_signal(window = window, logginglevel=logginglevel)
#     da_noise = da_loess.sn.calculate_rolling_noise(window = window, logginglevel=logginglevel)
#     da_sn = da_signal/da_noise


#     da_sn_stable, da_sn_increasing, da_sn_decreasing = \
#                         get_ds_above_below_and_between_bounds(da_sn, da_sn, control__lbound, control__ubound)
    
    
#     ds_sn = add_lower_upper_to_dataset(da_sn, control__lbound, control__ubound)
#     ds_sn_stable = add_lower_upper_to_dataset(da_sn_stable, control__lbound, control__ubound)
#     ds_sn_increasing = add_lower_upper_to_dataset(da_sn_increasing, control__lbound, control__ubound)
#     ds_sn_decreasing = add_lower_upper_to_dataset(da_sn_decreasing, control__lbound, control__ubound)

    
#     if return_all:
        
#         da_stable, da_increasing, da_decreasing = \
#                         get_ds_above_below_and_between_bounds(da, da_sn, control__lbound, control__ubound)
#         return (da_stable, da_increasing, da_decreasing, 
#                 ds_sn, ds_sn_stable, ds_sn_increasing, ds_sn_decreasing, 
#                 control__lbound, control__ubound)

#     return ds_sn, ds_sn_stable, ds_sn_increasing, ds_sn_decreasing
    
    
    
# def sn_multi_window(da, control_da, start_window = 21, end_window = 221, step_window = 8,
#                   logginglevel='ERROR'):
#     '''
#     Calls the global_mean_sn function repeatedly for windows ranging betweent start_window
#     and end_window with a step size of step_window.
    
#     Parameters
#     ----------
    
#     da, control_da, start_window = 21, end_window = 221, step_window = 8
    
    
#     Returns
#     -------
#     unstable_sn_multi_window_da , stable_sn_multi_window_da  Both these data sets contian dimension of time and window.
#     '''
#     utils.change_logging_level(logginglevel)
    
#     decreasing_sn_array = []
#     increasing_sn_array = []
#     stable_array = []
    
#     windows = range(start_window, end_window,step_window)
    
#     print(f'{start_window=}, {end_window=}, {step_window=}')
#     # Looping through
#     for window in windows:

#         print(f'{window}, ', end='')
#         da_sn, da_sn_stable, da_sn_increasing, da_sn_decreasing = global_sn(da, control_da, window = window,
#                                                                                 logginglevel=logginglevel)
        
#         increasing_sn_array.append(da_sn_increasing)
#         decreasing_sn_array.append(da_sn_decreasing)
#         stable_array.append(da_sn_stable)
    
#     # Mergine the das together to form a an array witht he S/N values and a dim called window
#     increasing_sn_multi_window_ds = xr.concat(increasing_sn_array, pd.Index(windows, name = 'window'))
#     decreasing_sn_multi_window_ds = xr.concat(decreasing_sn_array, pd.Index(windows, name = 'window'))
    
    
#     # Loading into memoery. 
#     increasing_sn_multi_window_ds = increasing_sn_multi_window_ds.compute()
#     decreasing_sn_multi_window_ds = decreasing_sn_multi_window_ds.compute()
    

#     unstable_sn_multi_window_da  = increasing_sn_multi_window_ds.fillna(0) + decreasing_sn_multi_window_ds.fillna(0)
#     unstable_sn_multi_window_da  = xr.where(unstable_sn_multi_window_da  != 0, unstable_sn_multi_window_da , np.nan)
    
#     unstable_sn_multi_window_da['time'] = unstable_sn_multi_window_da.time.dt.year.values
    
    
#     stable_sn_multi_window_da  = xr.where(np.isfinite(unstable_sn_multi_window_da ), 1, 0)
    
#     return unstable_sn_multi_window_da , stable_sn_multi_window_da 


# -