import xarray as xr
import numpy as np

import utils
logger = utils.get_notebook_logger()

from typing import Dict


def xr_dict_to_xr_dataset(data: Dict[str, xr.Dataset]):
    '''Takes a dicionary that has the model name and then model values
    as the value and merges them into an xarray dataset with the data_vars as the mode
    Parameters
    ----------
    dict '''
    to_merge = []
    for model, value in data.items():
        value.name = model
        to_merge.append(value)

    return xr.merge(to_merge, compat='override')


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
        np.nanpercentile,
        axis=axis,
        q=q,
        dtype=array.dtype,
        drop_axis=axis)


def percentile(da: xr.DataArray, dim: str, q: float) -> xr.DataArray:
    """
    Calculate the percentile along the specified dimension of the DataArray.

    Parameters:
        da (xarray.DataArray): The input DataArray.
        dim (str): The dimension along which to compute the percentile.
        q (float): The percentile value to calculate (between 0 and 100).

    Returns:
        xarray.DataArray: A new DataArray containing the computed percentile values.

    Note:
        If the input DataArray (`da`) is chunked (i.e., a dask array), this function
        will use `dask.array.nanpercentile` to calculate the percentile, otherwise,
        it will use `numpy.nanpercentile`.

    Example:
        import xarray as xr
        import numpy as np

        # Create a sample DataArray
        data = xr.DataArray(np.random.rand(10, 20), dims=['time', 'space'])

        # Calculate the 75th percentile along the 'time' dimension
        result = percentile(data, dim='time', q=75)
    """

    # Check if the DataArray is chunked
    if da.chunks:
        # If chunked (dask array), use dask_percentile from Xarray
        return da.reduce(dask_percentile, dim=dim, q=q)
    # If not chunked, use numpy.nanpercentile
    return da.reduce(np.nanpercentile, dim=dim, q=q)

def convert_dimension_to_data_vars(da: xr.DataArray, dim:str) -> xr.Dataset:
    '''Given a data array that has a dimension dim. Change this dimension 
    to variables
    Example
    -------
    print(da)
    
    >>> <xarray.DataArray 'tas' (time: 150, model: 8)>
        array([[1.84572607, 1.93748659, 1.33901388, ..., 1.47518842, 1.40841897,
               [1.69576504,        nan, 1.2413853 , ..., 1.25982132,        nan,
                2.92434189]])
        Coordinates:
          * time     (time) object 0001-12-31 00:00:00 ... 0150-12-31 00:00:00
          * model    (model) object 'CESM2' 'CanESM5' ... 'NorESM2-LM' 'UKESM1-0-LL'
            height   float64 2.0



    print(convert_dimension_to_data_vars(da, 'model'))
    
    >>> <xarray.Dataset>
        Dimensions:         (time: 150)
        Coordinates:
          * time            (time) object 0001-12-31 00:00:00 ... 0150-12-31 00:00:00
            model           <U5 'CESM2'
            height          float64 2.0
        Data variables:
            CESM2           (time) float64 1.846 1.997 1.995 1.799 ... 1.872 1.986 1.696
            CanESM5         (time) float64 1.937 2.008 2.094 2.04 ... nan nan nan nan
            GFDL-ESM4       (time) float64 1.339 1.407 1.5 1.418 ... 1.229 1.207 1.241
            GISS-E2-1-G-CC  (time) float64 2.397 2.338 2.085 2.068 ... 2.427 2.472 2.139
            MIROC-ES2L      (time) float64 1.3 1.408 1.601 1.648 ... 1.179 0.9098 0.8843
            MPI-ESM1-2-LR   (time) float64 1.475 1.543 1.621 1.515 ... 1.28 1.324 1.26
            NorESM2-LM      (time) float64 1.408 1.222 1.192 1.529 ... nan nan nan nan
            UKESM1-0-LL     (time) float64 2.681 2.827 2.819 2.842 ... 2.888 2.884 2.924
      
    '''
    dim_values = da[dim].values

    to_merge = []
    for d in dim_values:
        # Need to use loc as dim is a variable
        sub_da = da.loc[{dim:d}]
        sub_da.name = d
        to_merge.append(sub_da)
        
    merged_ds = xr.merge(to_merge, compat='override')
    return merged_ds



def add_lower_upper_to_dataset(da: xr.DataArray, lower: xr.DataArray, upper:xr.DataArray) -> xr.Dataset:
    '''Convert the dataarray to dataset, then ddd the lower and upper bounds
    as a variable.'''
    ds = da.to_dataset(name='signal_to_noise')
    
    ds['lower_bound'] = lower
    ds['upper_bound'] = upper
    
    return ds


# previosly get_median_and_uncertainty_across_dim
def get_average_and_uncertainty_across_dim(ds:xr.Dataset, dim:str, var:str='time', logginglevel='INFO',
                                           averaging_method='mean', uncertainty_method:str='percent'):
    '''
    Calcualtes the median and unvertainty
    uncertainty_method: str
        Options: percent, max_min
    '''
    
    utils.change_logginglevel(logginglevel)
    logger.info(f'{averaging_method=}, {uncertainty_method=}')
    if averaging_method == 'median':
        average_ds = ds.median(dim=dim)
    elif averaging_method == 'mean':
        average_ds = ds.mean(dim=dim)
        
    if uncertainty_method == 'percent':
        uncertainty_ds = np.abs((ds.max(dim=dim) - ds.min(dim=dim)) *100/average_ds)
    elif uncertainty_method == 'max_min':
        uncertainty_ds = np.abs((ds.max(dim=dim) - ds.min(dim=dim))/2)
    
    average_ds = average_ds.rename({var: 'average_value'})
    uncertainty_ds = uncertainty_ds.rename({var: 'uncertainty'})

    return xr.merge([average_ds, uncertainty_ds])



@xr.register_dataset_accessor('utils')
class Utils:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def var_comparison(self, main_var: str, other_var: str, func: callable):
        '''Compares one variable to another.
        This is useful for getting where the signal_to_nosie is above the 
        upper bound (unstable)
        
        '''
        ds = self._obj
        
        return ds[main_var].where(func(ds[main_var], ds[other_var])).to_dataset()
    
    @staticmethod
    def _deal_with_bounds(ds, main_var, greater_than_var, less_than_var, func):
        '''
        func = np.logical_and for when getting stable data
        (greater than lower AND less than upper)
        func = np.logical_or for when getting unstable
        (greater than upper OR less than lower)
        '''
        
        return ds[main_var].where(func(
            ds[main_var] > ds[greater_than_var], ds[main_var] < ds[less_than_var])).to_dataset()
    
    def between(self, main_var: str, greater_than_var: str, less_than_var: str):
        '''
        Bounds on variable by two other. Useful for getting where data is stable

    
        stable_sn_multi_window_da = sn_multiwindow_ds.utils.between(
                                'signal_to_noise', less_than_var = 'upper_bound', greater_than_var='lower_bound')
        '''
        ds = self._obj
        
        return self._deal_with_bounds(ds, main_var, greater_than_var, less_than_var, np.logical_and)
    
    def above_or_below(self, main_var: str, greater_than_var: str, less_than_var: str):
        '''
         Bounds on variable by two other. Useful for getting where data is stable
            
         Example
         -------
         unstable_sn_multi_window_da = sn_multiwindow_ds.utils.above_or_below(
                                'signal_to_noise', greater_than_var='upper_bound', less_than_var='lower_bound')
        '''
        ds = self._obj
        
        return self._deal_with_bounds(ds, main_var, greater_than_var, less_than_var, np.logical_or)
        
  