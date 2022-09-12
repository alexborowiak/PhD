import os
import logging, sys

import numpy as np

from typing import Union

logging.basicConfig(format="- %(message)s", filemode='w', stream=sys.stdout)
logger = logging.getLogger()


def get_notebook_logger():
    import logging, sys
    logging.basicConfig(format=" - %(message)s", filemode='w', stream=sys.stdout)
    logger = logging.getLogger()
    return logger

    

def change_logging_level(logginglevel: str):
    eval(f'logging.getLogger().setLevel(logging.{logginglevel})')

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


def pprint_list_string(l: list, num_start_items=2, no_end_items=0):
    '''This version is useful for logging moduel'''
    
    to_print = f'lenght = {len(l)}'
    for i in range(num_start_items):
        to_print += f'\n {i}. {str(l[i])}'
        
    if no_end_items:
        to_print += '\n...\n'
    for j in range(1, no_end_items+1):
        to_print += f'\n {-j}. {str(l[-j])}'
        
    return to_print


def pprint_list(*args, **kwargs) -> None:
    '''A nicer print of a list with more information'''

    print(pprint_list_string(*args, **kwargs))
    
    
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