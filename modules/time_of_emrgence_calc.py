import os, sys
from functools import partial
from itertools import groupby
from typing import Optional, Callable, Union, Tuple

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



def stats_test_1d_array(arr, stats_func:Callable, window: int=20, base_period_length:int = 50):
    """
    Apply stats_func test along a 1D array.

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



def return_hawkins_signal_and_noise(lt: ArrayLike, gt: ArrayLike, return_reconstruction:bool=False) -> Tuple[ArrayLike, ArrayLike]:
    """
    Calculate the signal and noise using the Hawkins method.

    Parameters:
        lt (ArrayLike): Time series data to be filtered.
        gt (ArrayLike): Time series used as the reference for filtering.
        return_reconstruction (Tuple) = False:
            Returns the reconstruction of the local time series.
            This is optional, as the reconstruction series is only needed for verification purposes                                   

    Returns:
        Tuple[ArrayLike, ArrayLike]: A tuple containing the filtered signal and noise.

    If either `lt` or `gt` contains all NaN values, it returns `lt` as both the signal and noise.

    The Hawkins method removes NaNs from the start and end of `lt` and `gt` to align the series.
    It then calculates the gradient `grad` and y-intercept `yint` of the linear fit between `gt` and `lt`.
    The signal is calculated as `signal = grad * gt`.
    The noise is calculated as the difference between `lt` and `signal`.

    NaN values are padded back to the filtered signal and noise arrays to match the original input length.
    """

    if np.all(np.isnan(lt)) or np.all(np.isnan(gt)):
        # If either series is all NaN, return lt as both signal and noise
        return lt, lt

    # If either is nan we want to drop
    nan_locs = np.isnan(lt)#  | np.isnan(gt)

    lt_no_nan = lt[~nan_locs]
    gt_no_nan = gt[~nan_locs]

    # Calculate the gradient and y-intercept
    grad, yint = np.polyfit(gt_no_nan, lt_no_nan, deg=1)

    # Calculate signal and noise
    signal = grad * gt_no_nan
    noise = lt_no_nan - signal

    signal_to_return = np.empty_like(gt)
    noise_to_return = np.empty_like(lt)
    
    signal_to_return.fill(np.nan)
    noise_to_return.fill(np.nan)

    signal_to_return[~nan_locs] = signal
    noise_to_return[~nan_locs] = noise

    
    if return_reconstruction:
        reconstructed_lt = grad * gt + yint
        return signal_to_return, noise_to_return, reconstructed_lt
    return signal_to_return, noise_to_return

    # # Pad NaNs back to the filtered signal and noise arrays
    # signal = np.concatenate([[np.nan] * number_nans_at_start, signal, [np.nan] * number_nans_at_end])
    # noise = np.concatenate([[np.nan] * number_nans_at_start, noise, [np.nan] * number_nans_at_end])

    # # Find the number of NaNs at the start and end of lt
    # number_nans_at_start = np.where(~np.isnan(lt))[0][0]
    # number_nans_at_end = np.where(~np.isnan(lt[::-1]))[0][0]

    # # Remove start NaNs
    # lt = lt[number_nans_at_start:]
    # gt = gt[number_nans_at_start:]

    # # Remove end NaNs if there are any
    # if number_nans_at_end > 0:
    #     lt = lt[:-number_nans_at_end]
    #     gt = gt[:-number_nans_at_end]

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


# TEST_NAME_MAPPING = {
#     return_ttest_pvalue:'ttest',
#     return_ks_pvalue: 'ks',
#     return_anderson_pvalue: 'anderson_darling'
# }


    
# def stats_test_with_ufunc(da: xr.DataArray, window: int, base_period_ds: xr.DataArray, statistic_func:Callable) -> xr.DataArray:
#     """
#     Apply statistical test using xarray's apply_ufunc.

#     Parameters:
#         da (xr.DataArray): Data to apply the test to.
#         window (int): Size of the rolling window for the test.
#         base_period_ds (xr.DataArray): Base period data for comparison.
#         statistic_func (Callable): Statistical function to use.

#     Returns:
#         xr.DataArray: DataArray containing the p-values.
#     """

#     assert isinstance(da, xr.DataArray)
#     output_da = xr.apply_ufunc(
#         statistic_func,
#         da.rolling(time=window).construct('window_dim')[(window-1):],
#         base_period_ds.rename({'time':'window_dim'}),
#         input_core_dims=[['window_dim'], ['window_dim']],
#         exclude_dims={'window_dim'},
#         vectorize=True,
#         dask='parallelized'#''
#     )
#     output_da.attrs = {'longname': TEST_NAME_MAPPING.get(statistic_func, 'p-value')}
#     return output_da

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