# +
import xarray as xr
import numpy as np

from typing import Dict


# -

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
#     array = array.rechunk({axis: -1})
    return array.map_blocks(
        np.percentile,
        axis=axis,
        q=q,
        dtype=array.dtype,
        drop_axis=axis)
