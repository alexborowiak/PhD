import os
import sys
import cftime

import numpy as np
import xarray as xr

from typing import List, Union

import logging
logging.basicConfig(format="- %(message)s", filemode='w', stream=sys.stdout)
logger = logging.getLogger()


import inspect

def function_name():
    caller_name = inspect.currentframe().f_back.f_code.co_name
    return f"**Calling function: {caller_name}"

def print_function_name():
    print(function_name())

def function_details(func):
    '''
    Designed to be used as a wrapper to functions that contain
    logging. When logginglevel='DEBUG' functions wrapped with this
    function will contain a log statement printing which function has been
    called.
    
    NOTE: The func doesn't actually need to have a logginglevel arguement
    NOTE2: A default arguement of 'logginglevel="DEBUG"' will not trigger this.
           This must be input into the function.
    
    Example
    @function_detials
    def my_func():
        pass
    
    my_func(logginglevel='DEBUG')
    
    '''
    def inner_func(*args, **kwargs):
        
        if 'logginglevel' in kwargs:
            logginglevel = kwargs['logginglevel']
        else: 
            logginglevel = 'ERROR'
        change_logging_level(logginglevel)
        fname = func.__name__
        logger.info(f'--- Running function {fname!r}')
        return func(*args, **kwargs)
    return inner_func

def get_notebook_logger():
    import logging, sys
    logging.basicConfig(format=" - %(message)s", filemode='w', stream=sys.stdout)
    logger = logging.getLogger()
    return logger

    

def change_logging_level(logginglevel: str):
    eval(f'logging.getLogger().setLevel(logging.{logginglevel})')
    
def change_logginglevel(logginglevel: str):
    change_logging_level(logginglevel)
    
    
change_logginglevel = change_logging_level

def create_period_list(step: int, end:int,  start:int = 0):
    '''Creates a list of tuples between start and end, with step 'step'
    
    Reason
    -------
    This is used in 07_exploring_consecutive_metrics_all_models_(nb_none) for
    getting the different period in time to calculate the percent of points
    unstable
    Example
    --------
    create_period_list(step = 25, end = 2, start = 0) 
    >> [(0, 24), (25, 49)]
    
    '''
    return [(i * step, (i+1) * step - 1) for i in range(start,end)]



def convert_period_string(period):
    '''
    Converts the periods created by create_period_list to a string.
    Reason
    ------
    This is used in 07_exploring_consecutive_metrics_all_models_(nb_none) 
    
    '''
    period_list = period.split('_')
    return f"Years {int(period_list[0]) + 1} to {int(period_list[1]) + 1}"


def pprint_list_string(
    l: List[Union[str, int, float]],
    num_start_items: int = 2,
    num_end_items: int = 0
) -> str:
    '''
    Generate a formatted string representation of a list, with specified number
    of start and end items included.
    
    Args:
        l: The list to be printed.
        num_start_items: The number of items to include from the start of the list.
        num_end_items: The number of items to include from the end of the list.
    
    Returns:
        A formatted string representation of the list.
    '''
    length = len(l)
    start_items = [f'{i}. {str(item)}' for i, item in enumerate(l[:num_start_items])]
    end_items = [f'{-j}. {str(item)}' for j, item in enumerate(l[-num_end_items:], start=1)]
    
    to_print = f'length = {length}\n'
    to_print += '\n'.join(start_items)
    
    if num_end_items:
        to_print += '\n...\n'
        to_print += '\n'.join(end_items)
        
    return to_print


def pprint_list(
    *args: List[Union[str, int, float]],
    **kwargs: Union[int]
) -> None:
    '''
    Print a formatted representation of a list with additional information.
    
    Args:
        l: The list to be printed.
        num_start_items: The number of items to include from the start of the list.
        num_end_items: The number of items to include from the end of the list.
    '''
    output = pprint_list_string(*args, **kwargs)
    print(output)

    
    
def mkdir_no_error(ROOT_DIR):
    try:
        os.mkdir(ROOT_DIR)
    except FileExistsError as e:
        pass
    
    
    
def ceil_to_base(values: Union[np.ndarray, float, int], base: int) -> np.ndarray:
    '''
    Ceil to the nearest base.
    E.g. 29 will ceil to 30 with base 10.
    '''
    return np.ceil(values/base) * base



def floor_to_base(values: Union[np.ndarray, float, int], base: int)-> np.ndarray: 
    '''
    Floor to the nearest base.
    E.g. 29 will ceil to 20 with base 10.
    '''
    return np.floor(values/base) * base


def get_tick_locator(vals: np.ndarray, num_major_ticks: int=10, fraction_minor_ticks:int=2) -> tuple:
    '''
    Based upon the range of values get the major and minor tick location spacing. 
    These are float values to be used with
    ax.yaxis.set_major_locator(mticker.MultipleLocator(major_locations))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(minor_location))
    
    Parameters
    ---------
    vals: np.ndarray
        Numpy array of any shape to base the values upon
    num_major_ticks: int
        The number of major ticks that are wanted on the axis
    fraction_minor_ticks: int
        How many minor ticks between each major tick
    '''
    # Range of values
    vals_range = np.nanmax(vals) - np.nanmin(vals)
    # The order of magnitude
    order_of_magnitude = np.floor(np.log10(np.array(vals_range)))
    # The ceiling of this.
    ceil_range = ceil_to_base(vals_range, 10 ** order_of_magnitude)
    
    # The range divided by the number of desired ticks
    major_locations = ceil_range/num_major_ticks
    
    major_locations = np.ceil(major_locations)
    
    # Minor ticks occur fraction_minor more often
    minor_location = major_locations/fraction_minor_ticks
    
    return (major_locations, minor_location)





def convert_to_0_start_cftime(time, freq='Y'):
    """
    Convert time values to a new time range with a starting year of 0 (year 1 AD).

    This function takes an array of time values, adjusts them to start from year 0 (1 AD),
    and returns a new time range based on the adjusted values.

    Args:
        time (numpy.ndarray): Array of time values.
        freq (str, optional): Frequency string for the new time range. Default is 'Y' (yearly).

    Returns:
        pandas.DatetimeIndex: A new time range starting from year 0 with the adjusted time values.
    """

    t0, tf = np.take(time, [0, -1])

    # Define the new start time as year 0
    t0_new = cftime.datetime(1, 1, 1, 0, 0, 0, 0, calendar='gregorian')

    # Calculate the new end time based on the difference between the original end time and start time
    tf_new = t0_new + (tf - t0) if freq is None else None

    # Generate a new time range using xarray's cftime_range
    new_time = xr.cftime_range(start=t0_new, end=tf_new, periods=len(time), freq=freq)

    return new_time



def reset_time_to_0_start(ds: Union[xr.Dataset, xr.DataArray]) ->  Union[xr.Dataset, xr.DataArray]:
    """
    Reset the time values of an xarray Dataset or DataArray to start from year 0 (1 AD).

    This function takes an xarray Dataset or DataArray and adjusts its time values to start from year 0 (1 AD).

    Args:
        ds (Union[xr.Dataset, xr.DataArray]): The xarray Dataset or DataArray with time values to be reset.

    Returns:
        Union[xr.Dataset, xr.DataArray]: The input xarray Dataset or DataArray with adjusted time values.
    """

    # Call the convert_to_0_start_cftime function to adjust time values
    ds['time'] = convert_to_0_start_cftime(ds.time.values)
    return ds