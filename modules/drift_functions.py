import numpy as np
import xarray as xr
import pymannkendall

from typing import List

def grid_trend(x: np.ndarray,t: np.ndarray):
    '''
    Calculates the trend for each individaul grid cell
    '''
    # If every point is just a nan values. We don't want to do the polyfit calculation. Just return nan
    if np.all(np.isnan(x)):
        return float('nan')
    
    # Getting the gradient of a linear interpolation
    idx = np.isfinite(x) & np.isfinite(t) #checking where the nans are for both
    x = x[idx]
    t = t[idx]
    if len(t) < 5 or len(x) < 5:
        return np.nan
    grad = np.polyfit(t,x,1)[0]
    return grad

#calculate_trend
def calculate_trend(data: xr.DataArray):
    '''
    Calcualtes the gradient of the trend along the year axis.
    '''
    
    # The axis number that year is
    axis_num = data.get_axis_num('time')
    
    # Applying trends along each grid cell
    trend_meta = np.apply_along_axis(grid_trend, axis_num, data.values, t = np.arange(len(data.time.values)))
    # Adding back to xarray data array.
    trend_da = xr.zeros_like(data.isel(time=0).drop('time').squeeze(), dtype=np.float64)+trend_meta

    return trend_da


def apply_mannkendall_test(y:List[float]) -> float:
    return pymannkendall.original_test(y).p

def calculate_pvals(da: xr.DataArray):
    '''
    Calcualtes the gradient of the trend along the year axis.
    '''
    import pymannkendall
    # The axis number that year is
    axis_num = da.get_axis_num('time')
    
    # Applying trends along each grid cell
    data_np = np.apply_along_axis(apply_mannkendall_test, da.get_axis_num('time'), da.values)
    # Adding back to xarray data array.
    trend_da = xr.zeros_like(da.isel(time=0).drop('time').squeeze(), dtype=np.float64)+data_np

    return trend_da