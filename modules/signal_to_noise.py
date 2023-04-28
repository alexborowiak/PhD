''' Signal to noise

This package contains all the functions needed for calculating the signal to noise of timeseries.

This also included the calculatin of anomalies
TODOL which should either be moved to antoher module, or the name of 
this package module should perhaps be renamed.

'''

import numpy as np
import pandas as pd
import itertools
import inspect
import xarray as xr
import xarray_extender as xe
import os, sys
sys.path.append('../')
from functools import partial
from multiprocessing import Pool

import xarray_class_accessors as xca
import utils
from typing import Optional, Union, Dict, Tuple, List, Callable
from numpy.typing import ArrayLike
from sn_typing import AnyXarray

import classes
import stats
from constants import LONGRUNMIP_CHUNKS
logger = utils.get_notebook_logger()


def calculate_ice_earth_fraction(ds: xr.Dataset) -> xr.Dataset:
    '''Calculates the fraction of the earth that is covered in ice for each time step'''
    ocean_as_1_ds = xr.where(np.isfinite(ds), 1, 0).isel(time=0)
    global_frac_ds = ds.sum(dim=['lat', 'lon'])/ocean_as_1_ds.sum(dim=['lat', 'lon'])
    return global_frac_ds


def calculate_global_value_and_anomly(ds: xr.DataArray, control_ds: xr.DataArray, variable:str, lat_bounds:tuple=None,
                          experiment_params=None):
    '''Calculates anomalies and mean.'''

    
    if not lat_bounds and not experiment_params: lat_bounds = (None,None)
    # It's easier to just pass the expereimtn_params_dict
    if isinstance(experiment_params, dict): lat_bounds = constants.HEMISPHERE_LAT[experiment_params['hemisphere']]
    print(lat_bounds)
    
    ds = ds.sel(lat=slice(*lat_bounds))
    control_ds = control_ds.sel(lat=slice(*lat_bounds))
    if variable == 'sic':
        ds_mean = calculate_ice_earth_fraction(ds)
        control_mean = calculate_ice_earth_fraction(control_ds)

    else:
        # Space mean and anomalmies
        ds_anom = ds.clima.anomalies(historical=control_ds)
        
        control_mean = control_ds.clima.space_mean()
        ds_mean = ds_anom.clima.space_mean() 
    
    return ds_mean.compute(), control_mean.compute()

def calculate_global_value(ds: xr.DataArray, variable:str, lat_bounds:tuple=None, experiment_params=None):
    '''Calculates anomalies and mean.'''

    if not lat_bounds and not experiment_params: lat_bounds = (None,None)
    if isinstance(experiment_params, dict): lat_bounds = constants.HEMISPHERE_LAT[experiment_params['hemisphere']]
    print(lat_bounds)
    
    ds = ds.sel(lat=slice(*lat_bounds))
    if variable == 'sic': ds_mean = calculate_ice_earth_fraction(ds)
    else: ds_mean = ds.clima.space_mean() 
    
    return ds_mean.persist()


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
    array = array.rechunk({axis: -1})
    return array.map_blocks(
        np.percentile,
        axis=axis,
        q=q,
        dtype=array.dtype,
        drop_axis=axis)



def calculate_upper_and_lower_bounds(ds:AnyXarray, var:str='signal_to_noise', lower_bound:float=1, upper_bound:float=99, 
                                    logginglevel='ERROR') -> Tuple[xr.DataArray]:
    
    '''
    Caculates the lower and upper bounds using percentiles: Returns [lower_bound, upper_bound]
    
    Variables
    ---------
    
    ds: AnyXarray
        Can be xarray dataset or dataarray. If datset need to specify the variable that the 90th percentile
        is calculated along. This is by default 'signal_to_noise'
    var: str = 'signal_to_noise'
        The variable to calculate this along
  
    Returns
    -------
    (control__lbound, control__ubound): Tuple[xr.DataArray]
        The lower and upper bound for ds for the variable 'var' along the time axis.
    
    '''
    
    utils.change_logging_level(logginglevel)
    logger.info(f'Calculating Upper and lower control bounds')
    
    if isinstance(ds, xr.Dataset):
        if len(list(ds)) >= 2:
            ds = ds[var] # Potential for other variables to be in there.
       
    if ds.chunks:
        logger.info('Map blocks used')
        control__ubound = ds.reduce(xe.dask_percentile, dim='time', q=upper_bound)
        control__lbound = ds.reduce(xe.dask_percentile, dim='time', q=lower_bound) 
    
    else:
        logger.info('np.nanpercentile used')
        control__ubound = ds.reduce(np.nanpercentile, dim='time', q=upper_bound)
        control__lbound = ds.reduce(np.nanpercentile, dim='time', q=lower_bound)
        
    logger.debug(f'{control__lbound.values}  - {control__ubound.values}')
    return (control__lbound, control__ubound)


def generate_windows(windows:Tuple[int]=None, start_window:int=None, end_window:int=None, step_window:int=None):   
    if windows:
        windows = windows
    else:
        if end_window is None: # Only want one window
            windows = [start_window]
        else: # Range of windows
            windows = range(start_window, end_window, step_window)
            
    return windows



def __check_concat(to_concat):
    '''
    Checks if ano obejct should be concatentated or not.S
    '''
    if len(to_concat) > 1: # We have ran more than one window
        print('\nConcatenating objects - PLEASE be patient!')
        return xr.concat(to_concat, dim='window').sortby('window')
    return  to_concat[0] # Only run with one window


def time_slice_da(da, time_slice):
    logger.info(f'slicing time with integers {time_slice}')
    return da.isel(time=slice(*time_slice))

def allocate_data_for_noise_calculation(da:Optional[AnyXarray]=None, da_for_noise:Optional[AnyXarray]=None, 
                                        detrend_kwargs:Dict=dict(), detrend:bool=True, time_slice:Tuple[int]=None,
                                        logginglevel='ERROR') -> Union[xr.DataArray, xr.Dataset]:
    
    utils.change_logging_level(logginglevel)
    logger.info('-- allocate_data_for_noise_calculation')
      
    if isinstance(da_for_noise, xr.DataArray):
        logger.info('Dataset for nosie provided')
        da_for_noise = da_for_noise
        if time_slice is not None: da_for_noise = time_slice_da(da_for_noise, time_slice)
    else: # Need to genereate data for noise
        logger.info('Generating dataset for noise.')
        if time_slice is not None: da = time_slice_da(da, time_slice) # Time slice before detrend
        if detrend:
            logger.info(f'Detrending required. Detrending data using {detrend_kwargs}')
            da_for_noise = da - stats.trend_fit(da, logginglevel=logginglevel, **detrend_kwargs)
        elif not detrend: # detrend === false
            logger.info('Detrending not required - using base data')
            da_for_noise = da
            
        
    return da_for_noise


def noise(da_for_noise:xr.DataArray, rolling_noise:bool=False, window:int=None, logginglevel='ERROR'):
    
    utils.change_logging_level(logginglevel)
    
    logger.info(f'{rolling_noise=}')
    if rolling_noise:
        noise_da = da_for_noise.sn.calculate_rolling_noise(window=window, logginglevel=logginglevel)
    if not rolling_noise:
        noise_da = da_for_noise.std(dim='time')
        noise_da.name = 'noise'
        
    return noise_da


def signal_to_noise(window:int, da:xr.DataArray, da_for_noise:Optional[xr.DataArray]=None, rolling_noise:bool=True,
                    logginglevel='ERROR', detrend:bool=True, detrend_kwargs:Dict=dict(), time_slice:Tuple[int]=None,
                    return_all=False) -> xr.DataArray:
    '''
    Window first for multiprocessing reasons.
    Calculates the rolling signal to nosie with an optional detrend
    
    ! This function cannot use decorators - the function gets called with partial which will error
    '''
    
    utils.change_logging_level(logginglevel)
    if logginglevel == 'ERROR': print(f'{window}, ', end='')
    
    # Signal is always rolling and always the data provided - no static, no detrend etc.
    signal_da = da.sn.rolling_signal(window=window)
    
    da_for_noise = allocate_data_for_noise_calculation(
        da=da, da_for_noise=da_for_noise, detrend_kwargs=detrend_kwargs, detrend=detrend, time_slice=time_slice,
        logginglevel=logginglevel)
    
    
    noise_da = noise(da_for_noise=da_for_noise, window=window, rolling_noise=rolling_noise, logginglevel=logginglevel)
    logger.info('Calculating signal to noise')
    
    sn_da = signal_da/noise_da
    sn_da.name = 'signal_to_noise'
    if return_all:
        da_for_noise.name = 'da_for_noise'
        return xr.merge([signal_da, da_for_noise, noise_da, sn_da])
    return sn_da


def __calculate_multi_window_signal_to_noise(windows:Tuple[int], parallel:bool=False, *args, **kwargs)->List[xr.DataArray]:
    
    to_concat = []
    logger.info(f'{__calculate_multi_window_signal_to_noise.__name__} - {windows=}')
    for window in windows:
        logger.debug(f'{window=}')
        ds_sn = signal_to_noise(*args, window=window, **kwargs)

        to_concat.append(ds_sn)
            
#     if parallel:
#         partial_calculate_rolling_signal_to_noise = partial(signal_to_noise, *args, **kwargs)
#         logger.debug(f'partial function - {partial_calculate_rolling_signal_to_noise.func.__name__} - '
#                  f'{partial_calculate_rolling_signal_to_noise}')
#         with Pool() as pool:
#             to_concat = pool.map(partial_calculate_rolling_signal_to_noise, windows)
        
    return to_concat


def __parallel_signal(window:int, da:xr.DataArray) ->xr.DataArray:
    '''To be used when calling a partial function with parallel multiwindow_signal'''
    print(f'{window}, ', end='')
    return da.sn.rolling_signal(window=window)
    
def multiwindow_signal(da:xr.DataArray, windows:Tuple[int], parallel:bool=False, logginglevel='ERROR')->List[xr.DataArray]:
    '''
    Calculate the signal over multiple window lenghts.
    This is particulary useful when the noise is static: standard deviation of all
    data
    
    '''   
#     if not parallel:
    to_concat = []
    for window in windows:
        print(f'{window}, ', end='')
        to_concat.append(da.sn.rolling_signal(window=window, logginglevel=logginglevel))
#     if parallel:
#         with Pool() as pool:
#             signal_func = partial(__parallel_signal, da=da)
#             to_concat = pool.map(signal_func, windows)
        
    return to_concat


def __rolling_multiwindow_signal_to_noise(*args, **kwargs):
    to_concat = __calculate_multi_window_signal_to_noise(*args, **kwargs)
    
    sn_da = __check_concat(to_concat)
    
    return sn_da


def __static_multiwindow_signal_to_noise(da:xr.DataArray, da_for_noise:xr.DataArray, windows:Tuple[int], parallel:bool=False,
                                         return_all:bool=False, logginglevel='ERROR'):
    
    to_concat = multiwindow_signal(da=da, windows=windows, parallel=parallel, logginglevel=logginglevel)
    
    sig_da_multi_window = __check_concat(to_concat)
    
    noise_da = noise(da_for_noise=da_for_noise, rolling_noise=False, logginglevel=logginglevel)
     
    sn_da = sig_da_multi_window/noise_da
    
    if isinstance(sn_da, xr.DataArray):
        sn_da.name = 'signal_to_noise'
    
    if return_all:
        return xr.merge([sn_da, sig_da_multi_window, noise_da])
    return sn_da

def multiwindow_signal_to_noise(
    da:xr.DataArray, rolling_noise=True, da_for_noise:Optional[xr.DataArray]=None,
    detrend:bool=True, detrend_kwargs:Optional[Dict]=dict(), time_slice:Tuple[int]=None,
    windows:Optional[Tuple[int]]=None, start_window:int=21, end_window:Optional[int]=None, step_window:Optional[int]=None,
    parallel=False, return_all=False, logginglevel='ERROR'):
    '''
    Function that calculates the signal to nosie for multiple different windows. This function is fitted
    with a plethora of different options.
    da: xr.DataArray
        The data to be calculating the signal-to-noise for.
    rolling_noise: bool=True
        Wether the noise calculate will be rolling (a timeseires)(True) or will be a single value (False) 
    windows: Tuple[int]
        The windows to calculate the signal to noise over
    start_window, end_window, step_window: int
        Designed ot be used together to generate a tuple for windows 
    detrend: bool=True
        This specifies if the data should be detrended or not. Shoudl be used in conjunction with detrned_kwargs
    detrend_kwargs: dict
        A dictionary of the way in which the dataset should be detrended.
        {'method':'polynomial', 'order': order}
        {'method': 'lowess', 'lowess_window':50}
    detrended_da: xr.DataArray
        Some filter methods (especially lowess) can be slow. It is quicker just to pass a version of da
        that has already been detrended.  
    parallel: bool = False
        Can run the multiple windows in parallel. Currently this feature is broken.
    ''' 
    utils.change_logging_level(logginglevel)

    windows = generate_windows(windows=windows, start_window=start_window,
                               end_window=end_window, step_window=step_window)
    
    logger.info(f'{windows=}')
 
    # The data will be detrended the same every time - no need to do it for every single window
    da_for_noise = allocate_data_for_noise_calculation(
        da=da, da_for_noise=da_for_noise, time_slice=time_slice,
        detrend_kwargs=detrend_kwargs, detrend=detrend, logginglevel=logginglevel)
    logger.debug(f'da_for_noise =\n{da_for_noise}')
    
    if rolling_noise: # Need to generate new noise each time
        sn_da = __rolling_multiwindow_signal_to_noise(
            da=da, da_for_noise=da_for_noise, windows=windows, parallel=parallel,
            rolling_noise=True, return_all=return_all, logginglevel=logginglevel)
    
    elif not rolling_noise: # Only need to calculate the noise once - same for all datasets (no window)
        sn_da = __static_multiwindow_signal_to_noise(
            da=da, da_for_noise=da_for_noise, windows=windows, parallel=parallel,
            return_all=return_all, logginglevel=logginglevel)
    
    return sn_da
                          
                                                           
def multiwindow_signal_to_nosie_and_bounds(
    experiment_da: xr.DataArray, control_da: xr.DataArray, da_for_noise:Optional[xr.DataArray]=None,
    rolling_noise=True, time_slice:Tuple[int]=None,
    windows:Optional[Tuple[int]]=None, start_window:int=21, end_window:Optional[int]=None, step_window:Optional[int]=None,
    detrend:bool=True, detrend_kwargs:Optional[Dict]=dict(),
    parallel=False, logginglevel='INFO', return_all=False, return_control:bool=False) -> xr.Dataset:
    '''
    Calculates the siganl to nosie for experiment_da and control_da. The signal to noise for
    control_da is then usef ot calculate lbound, and uboud. These bounds are then added to sn_ds, 
    to make a dataset with vars: signal_to_noise, lower_bound, upper_bound.
    Parameters
    
    NEW: Windows can be entered as a list of tuples, and this will be used instead of the range
    e.g. (20, 150, 30)
    
    This can work with a single window, just leave end_window and step_window as None.

    '''
    utils.change_logging_level(logginglevel)
    # This is to be slotted into sn_multi_window
    sn_kwargs = dict(windows=windows, start_window=start_window, end_window=end_window, step_window=step_window,
                     rolling_noise=rolling_noise, parallel=parallel, time_slice=time_slice, logginglevel=logginglevel,
                    return_all=return_all)
    logger.info(f'{sn_kwargs=}')
    
    print('\nExperiment\n--------\n', end='')
    experiment_da_sn = multiwindow_signal_to_noise(
        da=experiment_da, da_for_noise=da_for_noise, detrend=detrend, detrend_kwargs=detrend_kwargs, **sn_kwargs)
    
    print(' - Finished')
    print('\nControl\n------\n')
    # if da_for_noise = control_da then the raw control da will be used.
    control_sn = multiwindow_signal_to_noise(da=control_da, da_for_noise=control_da, detrend=False, **sn_kwargs)

    
    print('Persist')
    if control_sn.chunks:
        chunks = LONGRUNMIP_CHUNKS if ('lat' in list(control_sn.coords) and 'lon' in list(control_sn.coords)) else {'time':-1}
        control_sn = control_sn.chunk(chunks).persist()
    if experiment_da_sn.chunks: experiment_da_sn = experiment_da_sn.persist()
    print('Calculating bounds')
    lower_bound, upper_bound = calculate_upper_and_lower_bounds(control_sn, logginglevel=logginglevel)  
    
    if isinstance(experiment_da_sn, xr.DataArray): experiment_da_sn.to_dataset(name='signal_to_noise')
    
    print('final merge')
    sn_multiwindow_ds = xr.merge(
        [experiment_da_sn,
         lower_bound.to_dataset(name='lower_bound'),
         upper_bound.to_dataset(name='upper_bound')], 
        compat='override')
    
    if return_control:
        return sn_multiwindow_ds, control_sn
    
    return sn_multiwindow_ds.squeeze()



def stability_levels(ds:xr.Dataset) -> xr.DataArray:
    '''
    Dataset needs to have the data_vars: 'signal_to_noise', 'upper_bound', 'lower_bounds'
    Divides the datset into inncreasing unstable, decreasing unstable, and stable.
    
    These can then be counted to view the number of unstable and stable models at
    any point.
    
    '''
    decreasing_unstable_da = ds.where(ds.signal_to_noise < ds.lower_bound).signal_to_noise
    increasing_unstable_da = ds.where(ds.signal_to_noise > ds.upper_bound).signal_to_noise
    
    
    stable_da = ds.utils.between('signal_to_noise',
                                 less_than_var='upper_bound', greater_than_var='lower_bound').signal_to_noise
    unstable_da = ds.utils.above_or_below(
        'signal_to_noise', greater_than_var='upper_bound', less_than_var='lower_bound').signal_to_noise
    
    return xr.concat([decreasing_unstable_da, increasing_unstable_da, unstable_da, stable_da], 
                     pd.Index(['decreasing', 'increasing', 'unstable', 'stable'], name='stability'))



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



def get_stable_arg(values: np.ndarray, window: int) -> int:
    '''
    Calculates when the data first becomes stable given an array of unstable signal to noise
    Parameters
    ----------
    values: np.ndarray
        Unstable signal to noise values
    window: int
        Then window length which this was calculated over.
    
    Returns
    --------
    stable_arg: int
        The arguement when the data first becomes stable. 
    
    '''
    window = window/2
    condition = np.where(np.isfinite(values), True, False)
    
    condition_groupby = []
    for key, group in itertools.groupby(condition):
        condition_groupby.append((key, len(list(group))))

    # The arguements where stablilitity occurs
    condition_groupby_stable_arg = [i for i,(key,length) in enumerate(condition_groupby) 
                           if not key and length > window]
    
    if len(condition_groupby_stable_arg) > 0: condition_groupby_stable_arg = condition_groupby_stable_arg[0]
    else: return np.nan
        
    stable_arg = np.sum([length for key, length in condition_groupby[:condition_groupby_stable_arg]])
    
    return int(stable_arg)

def helper_get_stable_arg(data: np.ndarray, axis: int, window: int) -> np.ndarray:
    '''
    Applies the get_stable_arg function acorrs an axis
    
    Example
    -------
    da.reduce(helper_get_stable_arg, axis=da.get_axis_num('time'), window=window)
    '''
    return np.apply_along_axis(get_stable_arg, axis, data, window)


def get_stable_year(untsable_da: xr.DataArray, window:int, max_effective_length:int=None) -> xr.DataArray:
    
    if 'window' in list(untsable_da.coords): window = int(untsable_da.window.values)
    else: window = window
    da_stable = untsable_da.reduce(helper_get_stable_arg, axis=untsable_da.get_axis_num('time'), window=window)
    
    if max_effective_length is None: max_effective_length = len(untsable_da.time.values)

    print(f'Replacing points greater than {max_effective_length} with {max_effective_length+1}')
    da_stable = xr.where(da_stable > max_effective_length, max_effective_length+1, da_stable)
    
    return da_stable
    

def get_dataarray_stable_year_multi_window(da:xr.DataArray, max_effective_length:int=None) -> xr.DataArray:
    '''
    Loops through all the windows and calculated the first year stable for each window
    
    To apply this to a dataset use: <ds>.apply(get_dataarray_stable_year_multi_window)
    Parameters
    ----------
    da: xr.DataArray
        DataArray with nans where the climate is stable
    max effetive lenght: int
        For some windows there might not be enough data for this to be calculated. This will
        mean that if the climate stabilises at a late point due to trailing nans from a dataset
        being to short, they will just become nan.
    '''
    
    to_concat = []
    
    windows = da.window.values
    windows = np.atleast_1d(windows)
    
    for window in windows:
        da_window = da.sel(window=window)
        da_stable = da_window.reduce(helper_get_stable_arg, axis=da_window.get_axis_num('time'),
                                     window=window)
        to_concat.append(da_stable)
    
    concat_da = xr.concat(to_concat, dim='window')
    concat_da.name = 'time'
    
    if max_effective_length is None:max_effective_length = len(da.time.values)
    
    print(f'Replacing points greater than {max_effective_length} with {max_effective_length}')
    concat_da = xr.where(concat_da > max_effective_length, max_effective_length, concat_da)
    
    # Bug can occur where all nan values become very negative. This just returns them to beign the max
    concat_da = xr.where(concat_da < 0, max_effective_length, concat_da).fillna(max_effective_length)
    
    return concat_da


def get_dataset_stable_year_multi_window(ds:xr.Dataset, max_effective_length:int=None) -> xr.Dataset:
    '''
    Applying the get_dataarray_stable_year_multi_window to a dataset and then
    renaming the coord to time
    '''
    
    get_dataarray_stable_year_multi_window_partial = partial(
        get_dataarray_stable_year_multi_window, max_effective_length=max_effective_length)
    
    return (ds.apply(get_dataarray_stable_year_multi_window_partial)
                     .to_array(dim='variable')
                     .to_dataset(name='time'))

def get_stable_year_ds(sn_multi_ds, max_effective_length:int=None):
    '''
    Gets the year in which the time series first becomes styable
    '''

    unstable_sn_ds = sn_multi_ds.utils.above_or_below(
            'signal_to_noise', greater_than_var = 'upper_bound', less_than_var = 'lower_bound')

    stable_point_ds = get_dataset_stable_year_multi_window(unstable_sn_ds, max_effective_length=max_effective_length)
    return stable_point_ds



def get_stable_average_anomaly_lat_lon(arr1: ArrayLike, arr2:ArrayLike, window:int)->ArrayLike:
    '''
    arr2 is an index of the array arr1. They should have the same dimensions, but arr1
    will have an extra 0th dimension that is time. The index values in arr2 must be at 
    most the length of arr1's 0th dimension - window.
    
    Note: This is not completely general. The array must have lat and lon coords for this 
    to work. This cannot be completely general as a tuple cannot be unpacked into an index.
    E.g. ix_ = (I, J, K); arr1[arr2+i, *ix_] returns an error
    
    Example
    ------    
    arr1 = ds1.values
    arr2 = arg_ds.sel(window=21).squeeze().values.astype(int)
    get_stable_average_anomaly(arr1, arr2, 21)
    '''
    H, I, J, K = np.indices((window, *arr2.shape), sparse=True)
    out = arr1[H + arr2, I, J, K].mean(axis=0)
    
    return out


def get_stable_average_anomaly_global(arr1: ArrayLike, arr2:ArrayLike, window:int)->ArrayLike:
    '''
    arr2 is an index of the array arr1. They should have the same dimensions, but arr1
    will have an extra 0th dimension that is time. The index values in arr2 must be at 
    most the length of arr1's 0th dimension - window.
    
    Note: This is not completely general. The array must have lat and lon coords for this 
    to work. This cannot be completely general as a tuple cannot be unpacked into an index.
    E.g. ix_ = (I, J, K); arr1[arr2+i, *ix_] returns an error
    
    Example
    ------    
    arr1 = ds1.values
    arr2 = arg_ds.sel(window=21).squeeze().values.astype(int)
    get_stable_average_anomaly(arr1, arr2, 21)
    '''
    H, I = np.indices((window, *arr2.shape), sparse=True)
    out = arr1[H + arr2, I].mean(axis=0)
    
    return out

def get_stabel_average_anomaly_over_windows(ds1:xr.DataArray, arg_ds:xr.DataArray, 
                                           local=True) -> xr.DataArray:
    '''
    Loops over all window values and applies the get_stable_average_anomaly_lat_lon function.
    Then results are returned as xarray data array. 
    See get_stable_average_anomaly_lat_lon for details on how this function works.
    '''
    stable_anom_func = get_stable_average_anomaly_lat_lon if local else get_stable_average_anomaly_global
    print(stable_anom_func)
    arr1 = ds1.values
    
    mean_anom_array = []
    for window in arg_ds.window.values:
        arr2 = arg_ds.sel(window=window).squeeze().values.astype(int)
        mean_anom_at_stable_array_2 = stable_anom_func(arr1, arr2, window)
        mean_anom_array.append(mean_anom_at_stable_array_2)

    stable_anom_da = xr.zeros_like(arg_ds.squeeze()).astype(float)
    stable_anom_da.name = 'temp'
    stable_anom_da += mean_anom_array
    return stable_anom_da


def remove_values_before_index(arr1, arr2):
    '''
    arr2 values are all index values for arr1. This will remove all the values in arr1 after the index in arr2. This is done to remove all 
    the points after the model has become stabilised.
    2D example
    arr1 = [1,2,3,4,5,6]
    arr2 = [2]
    out = remove_values_before_index(arr1, arr2)
    out
    >>> [1,2,3, np.nan, np.nan, np.nan]
    https://stackoverflow.com/questions/76022378/numpy-set-all-values-to-np-nan-after-index-in-multi-dimensional-array/76022589#76022589
    '''
    # These are the possible indices along the first axis in arr1
    # Hence shape (100, 1, 1, 1):
    idx = np.arange(arr1.shape[0])[:, None, None, None]
    out = arr1.copy()
    out[idx < arr2] = np.nan
    return out

def remove_all_unstable_points(ds:xr.DataArray, arr_ds:xr.DataArray) -> xr.DataArray:
    '''
    Applies the remove_values_before_index to all windows, and then puts the output back into an xarray data frame
    '''

    windows = ds.window.values
    stor = []
    for window in windows:
        print(window)
        arr1 = ds.sel(window=window).values
        arr2 = arr_ds.sel(window=window).squeeze().values

        out = remove_values_before_index(arr1, arr2)
        stor.append(out)

    stable_da = xr.zeros_like(ds) + stor
    
    return stable_da


def find_stable_and_unstable_years_array(values_1d:List[float], window:int, number_attemps:int=4)-> List[List[int]]:
    """
    Finds the years when the time series data is stable and unstable.

    Args:
        values_1d (np.ndarray): One-dimensional array of time series data.
        window (int): THe window length that this has been calculated over.
        number_attemps (int): Max number of attempts.

    Returns:
        List[List[int]]: A list containing two lists. The first list contains the indices of the stable years,
        and the second list contains the indices of the unstable years.
    """
    values_1d = np.array(values_1d)
    values_1d[np.isfinite(values_1d)] = 1 # Set the finite values to 1
    values_1d[~np.isfinite(values_1d)] = 0 # And non-finite values to 0
    required_stable_perdiod = window/2
    cumulative_length = 0
    stable_args = []; unstable_args = []
    searching_for_stable = True # We are assuming that the time series is initially unstalbe and we want to find when first unstable
    value_checking_for = bool(not searching_for_stable) # The value we want to check for then is 0
    for key, group in itertools.groupby(values_1d):
        group = np.array(list(group)); group_length = len(group)    
        if group_length > required_stable_perdiod and np.all(group==value_checking_for): # The period is stable/unstable
            if searching_for_stable: stable_args.append(cumulative_length) # Assign to the correct array
            elif not searching_for_stable: unstable_args.append(cumulative_length)
            searching_for_stable = not searching_for_stable # Now we are searching for unstable
            value_checking_for = bool(not searching_for_stable)
        cumulative_length += len(group) # Update cumulate length at the end, as we want the first year becomes stable
    
    if len(stable_args) < number_attemps: stable_args = stable_args + [np.nan] * (number_attemps-len(stable_args))
    if len(unstable_args) < number_attemps: unstable_args = unstable_args + [np.nan] * (number_attemps-len(unstable_args))

    return [stable_args, unstable_args]


def find_stable_and_unstable_years(input_da:xr.DataArray, number_attemps:int=4):
    windows = input_da.window.values
    
    to_concat = []
    for window in windows:
        da = input_da.sel(window=window)
        axis_num = da.get_axis_num('time')
        stable_unstable_args = np.apply_along_axis(find_stable_and_unstable_years_array,
                                                   axis=axis_num,
                                                   arr=da.values,
                                                   window=window,
                                                   number_attemps=number_attemps)
        
        dims = np.array(list(da.dims))
        dims = dims[dims != 'time']
        dims = np.concatenate([['stability', 'arg'], dims])
        coords = dict(da.coords)
        coords['arg'] = np.arange(number_attemps)
        coords['stability'] = ['stable', 'unstable']
        del coords['time'] 
        
        stable_unstable_args_da = xr.DataArray(stable_unstable_args, dims=dims, coords=coords)
        to_concat.append(stable_unstable_args_da)
    to_return = xr.concat(to_concat, dim='window')
    to_return.name = 'time'
    return to_return


def remove_unstable_years_advanced(ds:xr.DataArray, arr_ds:xr.DataArray) -> xr.DataArray:
    """
    Removes years from `ds` based on stability information in `arr_ds`.

    Parameters:
    -----------
    ds : xr.DataArray
        The data to filter based on stability information.
    arr_ds : xr.DataArray
        An array with stability information, where the 'stability' dimension
        specifies whether a year is 'stable' or 'unstable', and the 'arg'
        dimension specifies the order in which the stability changes occur.

    Returns:
    --------
    xr.DataArray
        The filtered data with unstable years set to NaN.

    Notes:
    ------
    The function assumes that the `arr_ds` array contains three stability change
    events: (1) the first transition from unstable to stable, (2) the second
    transition from stable to unstable (which rarely happens), and (3) the
    third transition from unstable to stable (which only happens if the data
    destabilizes after the second transition). Years before the first transition
    to stability are set to NaN, years between the first and second transitions
    are set to 1, and years after the second transition to stability are set to NaN.
    """
    fill_value = 99999
    
    out = ds.values.copy()

    idx = np.arange(out.shape[0])[:, None, None, None, None]
    
    first_stable_args = arr_ds.sel(stability='stable', arg=0) # Time when first stabilises
    first_unstable_args = arr_ds.sel(stability='unstable', arg=0) # Time when this will unstabilise (rarely happens)
    second_stable_args = arr_ds.sel(stability='stable', arg=1) # Only happends if detsabilises

    out[idx < np.nan_to_num(first_stable_args, nan=fill_value)] = np.nan # Remove first unstalbe points - not to be counted
    out[np.logical_and(
        idx < np.nan_to_num(first_unstable_args, nan=fill_value), # If re-stabilises, these are to be counted.
        idx > np.nan_to_num(second_stable_args, nan=fill_value)
                      )] = 1
    out[idx > np.nan_to_num(second_stable_args, nan=fill_value)] = np.nan # Has destabilised - not to be counted.

    return (xr.zeros_like(ds) + out)

# def number_finite(da: xr.DataArray, dim:str='model') -> xr.DataArray:
#     '''
    
#     !!!! I don't know why you need this. This is just COUNT.
#     Gets the number of points that are finite .
#     The function gets all points that are finite across the dim 'dim'.
    
#     Paramaters
#     ----------
#     da: xr.Dataset or xr.DataArray (ds will be converted to da). This is the dataset 
#         that the number of finite points across dim.
#     number_present: xr.DataArray - the max number of available observations at each timestep
#     dim: str - the dimension to sum finite points across
    
#     Returns
#     ------
#     da: xr.DataArray - the fraction of finite points.
    
#     '''
    
#     # If da is a dataset, we are going to convert to a data array.
#     if isinstance(da, xr.Dataset):
#         da = da.to_array(dim=dim)
    
#     # The points that are finite become1 , else 0
#     finite_da = xr.where(np.isfinite(da), 1, 0)
    
#     # Summing the number of finite points.
#     number_da = finite_da.sum(dim)
    
    
#     return number_da


# def get_fraction_stable_ds(sn_multi_ds: xr.DataArray, ) -> xr.Dataset:
#     '''
#     Gets the number of models that are both stable and unstable at each time point.
#     Dataset must have dimension model, and must contain variables:
#     signal_to_noise, upper_bound, lower_bound.
#     '''
#     stable_sn_ds = sn_multi_ds.utils.between(
#         'signal_to_noise', less_than_var = 'upper_bound', greater_than_var = 'lower_bound')
    
#     unstable_sn_ds = sn_multi_ds.utils.above_or_below(
#             'signal_to_noise', greater_than_var = 'upper_bound', less_than_var = 'lower_bound')

    
#     stable_number_da = number_finite(stable_sn_ds.signal_to_noise)
#     unstable_number_da = number_finite(unstable_sn_ds.signal_to_noise)
#     stable_number_da.name = 'stable'
#     unstable_number_da.name = 'unstable'
    
#     return xr.merge([stable_number_da, unstable_number_da])



# def count_over_data_vars(ds: xr.Dataset, data_vars: list = None, dim='model') -> xr.DataArray:
#     '''
#     Counts the number of data vars that are present. 
    
#     Parameters
#     ----------
#     ds (xr.Dataset): the dataset to count over
#     data_vars (list): the data vars that need to be coutned.
#     dim (str): the dimenesion to be counted over
    
#     Returns
#     -------
#     number_da (xr.Dataarray): the number of occurences accross the data vars
    
#     '''
    
#     # If data_vars is none then we want all the data vars from out dataset
#     if data_vars is None:
#         data_vars = ds.data_vars
    
#     # Subsetting the desired data vars and then counting along a dimenstion. 
#     da = ds[data_vars].to_array(dim=dim) 
#     # This is the nubmer of models peresent at each timestep.
#     number_da = da.count(dim=dim)
#     # In the multi-window function, time has been changed to time.year, so must be done here as well
#     # Note: This may be removed in future.
#     number_da['time'] = ds.time.dt.year
#     number_da.name = f'number_of_{dim}'
#     return number_da
