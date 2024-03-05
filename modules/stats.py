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
def polynomial_fit(y: ArrayLike, x:Optional[ArrayLike] = None, order:float=None, deg:float=None) -> ArrayLike:
    """
    Perform a polynomial fit for line y using the Vandermonde matrix method.
    
    Args:
        y (ArrayLike): y-values to be used for fitting.
        x (Optional[ArrayLike]): Optional x-values that can be used. These values are only needed if they are
            not linearly increasing.
        order/deg (float): The order of the polynomial.
    
    Returns:
        ArrayLike: The fitted line.
    """
    # If x is not provided, generate linearly increasing values
    x = np.arange(len(y)) if x is None else x
    # Perform polynomial fit using numpy's polyfit function
    deg = order if order is not None else deg
    deg = 1 if deg is None else deg
    coeff = np.polyfit(x, y, deg=deg)
    # Generate the fitted line using the computed coefficients
    fitted_line = np.polyval(coeff, x)
    
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
        
    
    ufunc_dict = dict(input_core_dims=[['time']], output_core_dims=[['time']], vectorize=True,
                      output_dtypes=float, dask='parallelized')
    
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
    Generate a trend line for each grid cell in a given dataset.

    Parameters:
    da : xr.DataArray
        The data array to calculate the trend line along.
    method : str
        The method used to detrend. Options: [POLYNOMIAL, LOWESS]
    order : int
        Only for polynomial fitting. The order of the polynomial.
    lowess_window : int
        Only for LOWESS fitting. The window to take filter over.
    func_kwargs : dict, optional
        Additional keyword arguments for the detrending function.
    logginglevel : str
        The desired logging level.

    Returns:
    xr.DataArray
        The detrended data array.
    '''
    utils.change_logging_level(logginglevel)
    
    if not method:
        raise ValueError(f'method must be specified. Method options:  {[i.value for i in classes.detrendingMethods]}')

    if da.chunks is not None:
        da = da.unify_chunks()
    
    logger.debug(f'{method=}\n data \n {da}')
    
    method = classes.detrendingMethods[method.upper()]
    
    detrend_log_info = f'{order=}' if method == classes.detrendingMethods.POLYNOMIAL else f'{lowess_window=}'
    logger.info(f'Detrending data using {method.name} with ' + detrend_log_info)
    
    if method == classes.detrendingMethods.POLYNOMIAL:
        func1d = partial(polynomial_fit, order=order)

    elif method == classes.detrendingMethods.LOWESS:
        func1d = partial(lowess, exog=np.arange(len(da.time.values)), frac=lowess_window/len(da.time.values), return_sorted=False)
    
    logger.debug(f'func1d = {func1d.func.__name__}\n{func1d}')
    da_trend = apply_detrend_as_ufunc(da, func1d, **func_kwargs)
    
    return da_trend


