import numpy as np
import pandas as pd
import os
from glob import glob
import xarray as xr



def read_and_merge_netcdfs(fnames: list, ROOT_DIR: str, var='tas', length=999) -> xr.Dataset:
    '''
    Opens a list of fnames found in a common directory. Then merges these files
    together.
    
    Parameters
    ----------
    fnames: list of all the names of all the files
    ROOT_DIR: the directory all the file names are found in
    var: the variable to be loaded in 
    legnth: the minimum acceptable legnth for a dataset
    '''
    
    def _open_da(root_fname, model, var, length ):
        
        # Opening the dataset, renaming the varialbe to the model name,
        # (for merging) then converting to dataset.
        da = xr.open_dataset(root_fname).rename({var: model})[model]
        
        
        wanted_coords = ['lat', 'lon', 'time']
        
        # Getting the items that are different
        unwanted_coords = list(set(da.coords) - set(wanted_coords))
        if len(unwanted_coords) > 0: # Extra vars detected
            print(f'Dropping coords {unwanted_coords}')
            da = da.drop(unwanted_coords).squeeze()
        
        # Rejecting any datasets that have are shorter than lenght.
        if len(da.time.values) > length:
            return da
    
    data = []
    for fname in fnames:
        # Need to open da and alter length and names of vars
        model = fname.split('_')[2].lower()
        da = _open_da(os.path.join(ROOT_DIR, fname), model, var, length)
        
        # Some files will be rejected if they are too short.
        print(f'{fname}: ', end = '')
        if type(da) is not type(None):
            data.append(da)
            print('Y')
        else:
            print('N')
    
    return xr.merge(data, compat='override') 