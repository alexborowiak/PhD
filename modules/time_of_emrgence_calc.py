import os, sys
from functools import partial
from itertools import groupby
from typing import Optional, Callable, Union

import xarray as xr
import numpy as np
import dask.array as daskarray
from scipy.stats import anderson_ksamp, ks_2samp,ttest_ind
# from dask.array.stats import ttest_ind
from numpy.typing import ArrayLike

sys.path.append('../')
import signal_to_noise as sn



def return_ttest_pvalue(test_arr, base_arr):
    """
    Compute T-Test p-value between two arrays.

    Parameters:
        test_arr (ArrayLike): Array to test against base_arr.
        base_arr (ArrayLike): Base array to compare against.

    Returns:
        float: T-Test p-value.
    """
    return ttest_ind(test_arr, base_arr, nan_policy='omit').pvalue

def return_ks_pvalue(test_arr, base_arr):
    """
    Compute Kolmogorov-Smirnov test p-value between two arrays.

    Parameters:
        test_arr (ArrayLike): Array to test against base_arr.
        base_arr (ArrayLike): Base array to compare against.

    Returns:
        float: Kolmogorov-Smirnov test p-value.
    """
    return ks_2samp(test_arr, base_arr).pvalue


def return_anderson_pvalue(test_arr, base_arr):
    """
    Compute Anderson-Darling test p-value between two arrays.

    Parameters:
        test_arr (ArrayLike): Array to test against base_arr.
        base_arr (ArrayLike): Base array to compare against.

    Returns:
        float: Anderson-Darling test p-value.
    """
    if all(np.isnan(test_arr)) or all(np.isnan(base_arr)): return np.nan
    # print(test_arr.shape, base_arr.shape)
    return anderson_ksamp([test_arr, base_arr]).pvalue

TEST_NAME_MAPPING = {
    return_ttest_pvalue:'ttest',
    return_ks_pvalue: 'ks',
    return_anderson_pvalue: 'anderson_darling'
}

def stats_test_with_ufunc(da: xr.DataArray, window: int, base_period_ds: xr.DataArray, statistic_func:Callable) -> xr.DataArray:
    """
    Apply statistical test using xarray's apply_ufunc.

    Parameters:
        da (xr.DataArray): Data to apply the test to.
        window (int): Size of the rolling window for the test.
        base_period_ds (xr.DataArray): Base period data for comparison.
        statistic_func (Callable): Statistical function to use.

    Returns:
        xr.DataArray: DataArray containing the p-values.
    """

    assert isinstance(da, xr.DataArray)
    output_da = xr.apply_ufunc(
        statistic_func,
        da.rolling(time=window).construct('window_dim')[(window-1):],
        base_period_ds.rename({'time':'window_dim'}),
        input_core_dims=[['window_dim'], ['window_dim']],
        exclude_dims={'window_dim'},
        vectorize=True,
        dask='parallelized'#''
    )
    output_da.attrs = {'longname': TEST_NAME_MAPPING.get(statistic_func, 'p-value')}
    return output_da


def stats_test_1d_array(arr, stats_func:Callable, window: int=20, base_period_length:int = 50):
    """
    Apply Kolmogorov-Smirnov test along a 1D array.

    Parameters:
        arr (ArrayLike): 1D array to apply the test to.
        window (int): Size of the rolling window for the test.
        base_period_length (int, optional): Length of the base period. Defaults to 50.

    Returns:
        ArrayLike: Array of p-values.
    """
    # The data to use for the base period
    base_list = arr[:base_period_length]
    # Stop when there are not enough points left
    number_iterations = arr.shape[0] - window
    pval_array = np.zeros(number_iterations)
    
    for t in np.arange(number_iterations):
        arr_subset = arr[t: t+window]
        p_value = stats_func(base_list, arr_subset) # return_ttest_pvalue
        pval_array[t] = p_value

    # TODO: This could be done in the apply_ufunc
    lenghth_diff = arr.shape[0] - pval_array.shape[0]
    pval_array = np.append(pval_array, np.array([np.nan] *lenghth_diff))
    return pval_array 


# def stats_test_1d_array(arr: ArrayLike, window: int, stats_test:Callable,
#                     base_period_length:int = 50) -> ArrayLike:
#     """
#     Apply Kolmogorov-Smirnov test along a 1D array.

#     Parameters:
#         arr (ArrayLike): 1D array to apply the test to.
#         window (int): Size of the rolling window for the test.
#         base_period_length (int, optional): Length of the base period. Defaults to 50.

#     Returns:
#         ArrayLike: Array of p-values.
#     """
#     # The data to use for the base period
#     base_list = arr[:base_period_length]
#     # Fill function with base data - this needs to be done each time as the base period is unique
#     stats_test_partial = partial(return_ttest_pvalue, base_list) #statstest
#     # Stop when there are not enough points left
#     number_iterations = arr.shape[0] - window
#     #print(number_iterations)
#     kstest_array = np.zeros(number_iterations)
#     for t in np.arange(number_iterations):
#         arr_subset = arr[t:t+window]
#         ks_value = stats_test_partial(arr_subset)#.pvalue
#         #if isinstance(ks_value, (list, np.ndarray)):
#         #   ks_value = ks_value[0]  # Taking the first element
#         #print(ks_value)
#         kstest_array[t] = ks_value
#     return kstest_array

# def apply_stats_test_along_array(ds: xr.DataArray, window: int, stats_test:Callable, 
#                                  base_period_length: int = 50, 
#                                 ) -> xr.DataArray:
#     """
#     Apply Kolmogorov-Smirnov test along a 2D xarray.DataArray.

#     Parameters:
#         ds (xr.DataArray): DataArray to apply the test to.
#         window (int): Size of the rolling window for the test.
#         base_period_length (int, optional): Length of the base period. Defaults to 50.

#     Returns:
#         xr.DataArray: DataArray containing the p-values.
#     """
#     stats_test_1d_array_partial = partial(
#         stats_test_1d_array,
#         stats_test=stats_test,
#         window=window,
#         base_period_length=base_period_length)

#     ds_data = ds.data
#     time_axis_num = ds.get_axis_num('time')
#     local_ks_np = np.apply_along_axis(
#         stats_test_1d_array_partial, 
#         time_axis_num,
#         ds.data,
#     )

#     time_axis_num = ds.get_axis_num('time')
#     local_ks_ds = xr.zeros_like(ds.isel(time=slice(0, local_ks_np.shape[time_axis_num])))
    
#     local_ks_ds += local_ks_np

#     return local_ks_ds


def get_exceedance_arg(arr, time, threshold, comparison_func):
    """
    Get the index of the first occurrence where arr exceeds a threshold.

    Parameters:
        arr (array-like): 1D array of values.
        time (array-like): Corresponding 1D array of time values.
        threshold (float): Threshold value for comparison.
        comparison_func (function): Function to compare arr with the threshold.

    Returns:
        float: The time corresponding to the first exceedance of the threshold.
               If there is no exceedance, returns np.nan.

    Example:
        data = [False, False, False, False, False, False,
                False, False, False, False, True, False, True, 
                True, True]

        # Group consecutive True and False values
        groups = [(key, len(list(group))) for key, group in groupby(data)]
        print(groups)
        >>> [(False, 10), (True, 1), (False, 1), (True, 3)]
        # Check if the last group is True
        groups[-1][0] == True
        # Compute the index of the first exceedance
        first_exceedance_arg = int(np.sum(list(map(lambda x: x[1], groups))[:-1]))
        print(first_exceedance_arg)
        >>> 12
    """
    # Entire nan slice, return nan
    if all(np.isnan(arr)):
        return np.nan

    # Find indices where values exceed threshold
    greater_than_arg_list = comparison_func(arr, threshold)

    # If no value exceeds threshold, return nan
    if np.all(greater_than_arg_list == False):
        return np.nan

    # Group consecutive True and False values
    groups = [(key, len(list(group))) for key, group in groupby(greater_than_arg_list)]

    # If the last group is False, there is no exceedance, return nan
    if groups[-1][0] == False:
        return np.nan

    # The argument will be the sum of all the other group lengths up to the last group
    first_exceedance_arg = int(np.sum(list(map(lambda x: x[1], groups))[:-1]))

    # Get the time corresponding to the first exceedance
    first_exceedance = time[first_exceedance_arg]

    return first_exceedance

def get_permanent_exceedance(ds: xr.DataArray, threshold: Union[int, float], comparison_func: Callable,
                             time: Optional[xr.DataArray] = None)-> xr.DataArray:
    """
    Calculate the time of the first permanent exceedance for each point in a DataArray.

    This function calculates the time of the first permanent exceedance (defined as when a value exceeds a threshold
    and never goes below it again) for each point in a DataArray.

    Parameters:
        ds (xr.DataArray): Input data.
        threshold (Union[int, float]): Threshold value for exceedance.
        comparison_func (Callable): Function to compare values with the threshold.
        time (Optional[xr.DataArray]): Optional array of time values corresponding to the data. 
                                        If not provided, it will default to the 'year' component of ds's time.

    Returns:
        xr.DataArray: DataArray containing the time of the first permanent exceedance for each point.
    """
    # If time is not provided, use 'year' component of ds's time
    if time is None:
        time = ds.time.dt.year.values
        
    # Partial function to compute the exceedance argument
    partial_exceedance_func = partial(get_exceedance_arg, time=time, threshold=threshold, comparison_func=comparison_func)
               
    # Dictionary of options for xr.apply_ufunc
    exceedance_dict = dict(
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True, 
        dask='parallelized',
        output_dtypes=[float]
    )

    # Apply the partial function to compute the permanent exceedance
    return xr.apply_ufunc(
        partial_exceedance_func, 
        ds, 
        **exceedance_dict
    )

def create_exceedance_single_point_dict(toe_ds, timeseries_ds):
    """
    Creates a dictionary with year, corresponding datetime, and value from two datasets.

    Parameters:
        toe_ds (xarray.Dataset): Dataset containing a single value representing a year.
        timeseries_ds (xarray.Dataset): Dataset containing a time series.

    Returns:
        dict: A dictionary with keys 'year', 'year_datetime', and 'val'.

    Note:
        This function assumes both datasets are xarray Datasets.

    Example:
        create_exceedance_single_point_dict(toe_dataset, timeseries_dataset)
    """
    
    # Extract the year from toe_ds values
    year = toe_ds.values
    
    # Find the datetime corresponding to the extracted year in timeseries_ds
    year_datetime = timeseries_ds.sel(time=timeseries_ds.time.dt.year==int(year)).time.values[0]
    
    # Find the value corresponding to the extracted year in timeseries_ds
    val = timeseries_ds.sel(time=timeseries_ds.time.dt.year==int(year)).values[0]
    
    # Create and return the dictionary
    return {
        'year': year,
        'year_datetime': year_datetime,
        'val': val
    }


# def return_statistic_func_pvalue(statistic_func, test_arr, base_arr):
#     if statistic_func == anderson_ksamp: statistic_func = statistic_func([base_arr, test_arr])
#     else:  statistic_func = statistic_func(base_arr, test_arr)
#     return statistic_func.pvalue

# ttest_ind_partial = partial(toe.return_statistic_func_pvalue, statistic_func=ttest_ind)
# anderson_ksamp_partial = partial(toe.return_statistic_func_pvalue, statistic_func=anderson_ksamp)
# ks_2samp_ind_partial = partial(toe.return_statistic_func_pvalue, statistic_func=ks_2samp)



# from scipy.stats import ttest_ind

# def return_ttest_pvalue(test_arr, base_arr):
#     """
#     Compute T-Test p-value between two arrays.

#     Parameters:
#         test_arr (ArrayLike): Array to test against base_arr.
#         base_arr (ArrayLike): Base array to compare against.

#     Returns:
#         float: T-Test p-value.
#     """
#     return ttest_ind(test_arr, base_arr, nan_policy='omit').pvalue

# def stats_test_1d_array(arr, window: int=20, base_period_length:int = 50):
#     """
#     Apply Kolmogorov-Smirnov test along a 1D array.

#     Parameters:
#         arr (ArrayLike): 1D array to apply the test to.
#         window (int): Size of the rolling window for the test.
#         base_period_length (int, optional): Length of the base period. Defaults to 50.

#     Returns:
#         ArrayLike: Array of p-values.
#     """
#     # The data to use for the base period
#     base_list = arr[:base_period_length]
#     # Stop when there are not enough points left
#     number_iterations = arr.shape[0] - window
#     pval_array = np.zeros(number_iterations)
    
#     for t in np.arange(number_iterations):
#         arr_subset = arr[t: t+window]
#         p_value = return_ttest_pvalue(base_list, arr_subset)
#         pval_array[t] = p_value

#     # TODO: This could be done in the apply_ufunc
#     lenghth_diff = arr.shape[0] - pval_array.shape[0]
#     pval_array = np.append(pval_array, np.array([np.nan] *lenghth_diff))
#     return pval_array 