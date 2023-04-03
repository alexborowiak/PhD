import numpy as np
import xarray as xr
from functools import partial

import classes
import utils

import statsmodels.api as sm 
lowess = sm.nonparametric.lowess


from numpy.typing import ArrayLike
from typing import Optional, Dict, Callable


logger = utils.get_notebook_logger()


@utils.function_details
def polynomaial_fit(y: ArrayLike, x:Optional[ArrayLike]=None, order:float=1)->ArrayLike:
    '''
    Polynomial fit for line y using the Vandermone matrix method.
    https://en.wikiversity.org/wiki/Numerical_Analysis/Vandermonde_example
    y: ArrayLike
        y-values to be used
    x: Optional[ArrayLike] = None
        Optional x-values that can be used. These values are only needed if they are
        not linearly increasing
    order: float
        The order of the polnomial
    '''
    x = np.arange(len(y)) if x is None else x
    coeff = np.polyfit(x, y, deg=order)

    fitted_line = np.polyval(coeff, x)
    
    #x = np.arange(len(y)) if x is None else x
    #vandermond = np.vander(x, N=order+1)
    #coefficient_matrix = np.linalg.pinv(vandermond).dot(y)
    #fitted_line = np.dot(vandermond, coefficient_matrix)
    
    return fitted_line

@utils.function_details
def lowess_fit(exog: Callable, window:int=50) -> Callable:
    '''
    A function to fill the lowess function with exog (x values) and the fraction
    '''
    return partial(lowess, exog=exog, frac=window/len(exog), return_sorted=False)


@utils.function_details
def apply_detrend_as_ufunc(
    da: xr.DataArray, func1d: Callable, func_kwargs:Optional[Dict]=None, debug=False) -> xr.DataArray:
    '''
    Applies the detrending funcs as a ufunc.
    '''
    
    if debug: print(func_kwargs)
    if func_kwargs is not None:
        func1d = partial(func1d, **func_kwargs)
        
    
    ufunc_dict = dict(input_core_dims=[['time']],
                      output_core_dims=[['time']], 
                      vectorize=True,
                      output_dtypes=float,
                      dask='parallelized')
    
    try:
        to_return = xr.apply_ufunc(func1d, da, **ufunc_dict)
    except ValueError as e:
        logger.debug(e)
        ufunc_dict.pop('output_dtypes')
        logger.debug(f'Trying again without output type specifiction {ufunc_dict}')
        to_return = xr.apply_ufunc(func1d, da, **ufunc_dict)
    
    return to_return
       

# @utils.function_details
def trend_fit(da:xr.DataArray, method:str=None, order:int=1, lowess_window:int=30, func_kwargs:Optional[Dict]={},
             logginglevel='ERROR'):
    '''
    Genertes a trend line for each grid cell in a given dataset
    da: xr.DataArray
        The data array to calcualted the trend line along
    method: str
        The method used to detrend. For options see detrendingMethods ckass
    order: int
        ONLY for polynomial fitting. The order of the polynomial
    window: int
        ONLY for lowess fitting. The window to take filter over.
    '''
    utils.change_logging_level(logginglevel)
    
    if not method:
        raise ValueError(f'method must be specified. Method options:  {[i.value for i in classes.detrendingMethods]}')

    if da.chunks is not None:
        da = da.unify_chunks()#.chunk({'time':-1})
    
    logger.debug(f'{method=}\n data \n {da}')
    
    method = classes.detrendingMethods[method.upper()]
    
    detrend_log_info = f'{order=}' if method == classes.detrendingMethods.POLYNOMIAL else f'{lowess_window=}'
    logger.info(f'Detrending data using {method.name} with ' + detrend_log_info)
    
    if method == classes.detrendingMethods.POLYNOMIAL:
        func1d = partial(polynomaial_fit, order=order)

    elif method == classes.detrendingMethods.LOWESS:
        func1d = partial(lowess, exog=da.time.values, frac=lowess_window/len(da.time.values), return_sorted=False)
    
    logger.debug(f'func1d = {func1d.func.__name__}\n{func1d}')
    da_trend = apply_detrend_as_ufunc(da, func1d, **func_kwargs)
    
    return da_trend


