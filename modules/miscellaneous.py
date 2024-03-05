import xarray as xr
from typing import Union

import utils

logger = utils.get_notebook_logger()

# def convert_ds_units(ds: Union[xr.Dataset, xr.DataArray], variable:str, logginlevel='ERROR'):
#     if variable == 'tas':
#         print()