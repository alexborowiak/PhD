from typing import Callable, Tuple, Literal

import numpy as np
import xarray as xr


def find_nth_extreme_location(ds:xr.Dataset, method=Literal['max', 'min'], nth=1, 
                             output_dtype=Literal['tuple', 'dict']):
    """
    Find the location (lat, lon) of the nth extreme value in an xarray Dataset.
    
    Parameters:
        ds (xarray.Dataset): The dataset to search.
        func (function): A function to apply to the dataset to find the extreme value.
        nth (int): The order of the extreme value to find (1 for min, 2 for 2nd smallest, etc.).
    
    Returns:
        Tuple: (lat, lon) of the nth extreme location.
    """
    # Get the nth extreme value

    # Sort the values
    sorted_arr = np.sort(ds.values.flatten())
    sorted_arr = sorted_arr[np.isfinite(sorted_arr)]
    # The sorting is from smalest to largest, so reverse if we want the max
    if method=='max': sorted_arr = sorted_arr[::-1]    
    nth_extreme_value = sorted_arr[nth]

    # Find location corresponding to nth extreme value
    extreme_loc = ds.where(ds == nth_extreme_value, drop=True)
    lat = extreme_loc.lat.values[0]
    lon = extreme_loc.lon.values[0]
    return (lat, lon) if output_dtype == 'tuple' else {'lat':lat, 'lon':lon}

