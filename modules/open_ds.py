'''
Standardising the format of all data. 

'''


import numpy as np
import pandas as pd
import xarray as xr
import cftime
from typing import List, Dict
import os
from glob import glob
import constants
import json


def open_dataset(fpath: str) -> xr.Dataset:
    '''
    Tries to open with cf_time, otherwise will not. Works for multi-dataset
    '''
    
    open_function = xr.open_mfdataset if isinstance(fpath, list) else xr.open_dataset
     
    # TODO: Need to figure out what the error is with files having a string as timestep.
    try:
        ds = open_function(fpath, use_cftime=True)
        return ds
    except ValueError as e:
        print(f'{os.path.basename(fpath)} has failed with ValueError')
        
#     except:
#         # Certain files have an object (string) as the datetime - they will error with cftime.
#         ds = open_function(fpath)
        
#     return ds


def read_and_merge_netcdfs(fnames: List[str], ROOT_DIR: str = '',
                           var:str ='tas', model_index:int = 2, verbose=False) -> xr.Dataset:
    '''
    Opens a list of fnames found in a common directory. Then merges these files
    together.
    
    Parameters
    ----------
    fnames: list[str]
        list of all the names of all the files
    ROOT_DIR: str
        the directory all the file names are found in
    var: string
        the variable to be loaded in 
    model_index: int
        When splitting by "_" where does the model name appear in the lsit

        
    Example
    --------
    
    ROOT_DIR = <path_to_files>
    fnames = os.listdir(ROOT_DIR)
    var = 'tas'
    read_and_merge_netcdfs(fnames, ROOT_DIR, var,)
    '''
    
    def _open_da(root_fname, model, var, model_index = 2, verbose=False):
        
        # Opening the dataset, renaming the varialbe to the model name,
        # (for merging) then converting to dataset.
        da = xr.open_dataset(root_fname).rename({var: model})[model]
        
        if verbose:
            print(da)
            print('-----------')
        
        wanted_coords = ['lat', 'lon', 'time']
        
        # Getting the items that are different
        unwanted_coords = list(set(da.coords) - set(wanted_coords))
        if len(unwanted_coords) > 0: # Extra vars detected
            print(f' - Dropping coords {unwanted_coords}')
            da = da.drop(unwanted_coords).squeeze()
        

        return da
    
    data = []
    for fname in fnames:
        print(f'{fname}')

        # Need to open da and alter length and names of vars
        model = fname.split('_')[2].lower()
        da = _open_da(os.path.join(ROOT_DIR, fname), model, var, model_index, verbose)
        
        # Some files will be rejected if they are too short.
        data.append(da)
        da.attrs['length'] = len(da.time.values)

    
    return xr.merge(data, compat='override') 


def convert_numpy_to_cf_datetime(t_input):
    '''This function converts a numpy datetime to cftime.'''
    # Converting to pandas datetime, then to tuple, then getting
    # the first four elements (year, month, day, hour)
    t_tuple = list(pd.to_datetime(t_input).timetuple())[:4]
    # Converting to cftime
    t_output = cftime.datetime(*t_tuple, calendar='gregorian')
        
    return t_output


def refactor_dims(ds:xr.Dataset) -> xr.Dataset:
        
    # The name of the dim can be different in each file. So need to get the dim that isn't lat or lon.
    dims =  np.array(list(ds.dims.keys()))
    
    # This should be the time dim in the dataset.
    possible_non_time_dims = ['lon', 'lat', 'long', 'longitude', 'latitude', 'lev', 'bnds','bounds', 'model']
    time_dim = dims[~np.isin(dims, possible_non_time_dims)][0]
    
    # Time dime is not called time
    if time_dim != 'time':
        print(f'Chaning {time_dim} to time')
        ds = ds.rename({time_dim: 'time'})
    if 'longitude' in dims:
        ds = ds.rename({'longitude': 'lon'})
    if 'latitude' in dims:
        ds = ds.rename({'latitude': 'lat'})
        
    return ds



def make_new_time(ds: xr.Dataset, freq:str=None, debug=True)-> List: 
    '''
    Create a new time dimensions for the dataset starting at 1-1-1 in cftime.
    This is done to standardise athe data.
    '''
    
    time = ds.time.values
    t0, tf = np.take(time, [0,-1])
    

    t0_new = cftime.datetime(1, 1, 1, 0, 0, 0, 0, calendar='gregorian')
    # New end time is the total length of the old dataset added to t0
    

    tf_new = t0_new + (tf - t0) if freq is None else None


    new_time = xr.cftime_range(start = t0_new, end = tf_new, periods = len(ds['time'].values), 
                               freq=freq)
    
    if debug:
        print(f'Chaning time to range between {new_time[0]} and {new_time[1]} with length = {len(new_time)}')
    
    return new_time

def correct_dataset(ds, debug=False, **kwargs):
    
    '''This function makes a dataset of into standard format.'''

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    
    # Making sure main dims are lat, lon and time. 
    ds = refactor_dims(ds)

    # New time range
    # Defining the start of the new time series.
    t0 = ds.time.values[0] # First time in dataset
    tf = ds.time.values[-1] # Final time in dataset
    # If numpy datetime, we want to convert to cftime
    if ds.time.dtype == np.dtype('<M8[ns]'):
        if debug:
            print('Converting from numpy to cftime')
        t0 = convert_numpy_to_cf_datetime(t0)
        tf = convert_numpy_to_cf_datetime(tf)
    
    if debug:
        print(f'Dataset ranges between {t0} and {tf}')

        
    # TODO: The can be change to the make_new_time function
    # New start time is zero
    t0_new = cftime.datetime(1, 1, 1, 0, 0, 0, 0, calendar='gregorian')
    # New end time is the total length of the old dataset added to t0
    tf_new = t0_new + (tf - t0)
    if debug:
        print(f'New time dim will range between {t0_new} and {tf_new}')
    new_time = xr.cftime_range(start = t0_new, end = tf_new, periods = len(ds['time'].values), 
                               freq=None)
    
    if debug:
        print('Old time values')
        print(ds['time'].values[:5])
        print(ds['time'].values[-5:])
        print('New time values')
        print(new_time[:4])
        print(new_time[-4:])
    
    ds['time'] =  new_time
    
    if debug:
        print('Resampling to yearly data')
    ds = ds.resample(time='Y').mean()
        
    print('\n\n\Data correction successfull')
    

    return ds


def zecmip_open_matching_picontrol(fpath: str, 
                            unique_picontrol_paths: List[str],
                            experiment:str='esm-1pct-brch-1000PgC', debug=False) ->xr.DataArray:
    
    '''This function takes a path, and finds the matching path for piControl from a list of paths
    containing piControl paths.
    
    Parameters
    ----------
    fpath: the path to the dataset that you want to match to piControl.
    unique_picontrol_paths: paths to different piControl runs.
    experiment: the experiment of what you are matching.
    
    
    
    Sample of split path
    fpath = '/g/data/oi10/replicas/CMIP6/C4MIP/CCCma/CanESM5/esm-1pct-brch-1000PgC/r1i1p2f1/Amon/tas/gn/v20190429'
    fpath.split('/')
    0-, 1-g, 2-data, 3-oi10, 4-replicas, 5-CMIP6, 6-C4MIP, 7-CCCma, 8-CanESM5, 9-esm-1pct-brch-1000PgC, 10-r1i1p2f1, 11-Amon,       12-tas, 13-gn, 14-v20190429
    
    '''
    import re
    print(f'Attempting to open matching picontrol for:\n{fpath}')
        
    
    path_branch = '/'.join(fpath.split('/')[7:-2])
    if debug:
        print(f'{path_branch=}')
    path_branch = path_branch.replace(experiment, '\w+')#'piControl|esm-piControl')
    if debug:
        print(f'Searching for branch containing\n{path_branch}')
        
    PATH_FOUND = False
    for picontrol_path in unique_picontrol_paths:
        #if path_branch in pi_path:
        core_picontrol_path = '/'.join(picontrol_path.split('/')[7:-2])
        
        if debug:
            print(f'{path_branch=} - {core_picontrol_path=}')

        if re.search(path_branch, core_picontrol_path):
            PATH_FOUND = True
            if debug:
                print(f'Found branch:\n{path_branch}')
            break
    if not PATH_FOUND:
        raise Exception('No match found')
        
    # Opening dataset
    if debug:  
        print(f'Found path:\n{picontrol_path}')
    picontrol_ds = xr.open_mfdataset(os.path.join(picontrol_path, '*.nc'), use_cftime='True')
    picontrol_ds = refactor_dims(picontrol_ds)

    return picontrol_ds


def remove_unwated_coords(da):
    '''
    Removes all coords that aren't lat, lon and time.
    '''
    wanted_coords = ['time', 'lon', 'lat']
    coords = list(da.coords)
    unwanted_coords = [c for c in coords if c not in wanted_coords]
    if unwanted_coords:
        print(f'Removing coords - {unwanted_coords}')
        for c in unwanted_coords:
            da = da.drop(c)
    return da


def open_and_concat_nc_files(nc_files: List[str]):
    '''
    Purpose
    -------
    Opens all the listed files ina  directory and concatenates them together. ALos removes unwanted
    coords. 
    Reaon
    ------
    This funcion was created as part of '07_exploring_consecutive_metrics_all_models_(nb_none) as there
    was a need to open a different datasets for different models. Could not be variables each model
    as there are many variables in each data set. They also couldn't be merged as there where
    conflicting dimensions. 
    
    Parameters
    ---------
    nc_files: List[str]
        List of all the files (with directory attached).
        
    '''
    xr_files = {}
    for f in nc_files:
        print(f'Opening {f}')
        model = os.path.basename(f).split('_')[-1].split('.')[0]
        da = xr.open_dataset(f)
        da = remove_unwated_coords(da)

        xr_files[model] = da
    print(f'Merging together {list(xr_files)}')
    ds = xr.concat(xr_files.values(), dim = pd.Index(list(xr_files.keys()), name='model'))

    return ds


def get_exeriment_file_names(debug=False) -> Dict[str, List[str]]:
    '''
    Gets all the file names sotres in LONGRUNMIP_RETIMED_DIR and LONRUNMIP_LOESS_DIR
    for abrupt4x and control runs.
    Reason:
    A list of the file names for each experiment
    - Created in 06_saving_consecutive_metrics_all_models_(nb24)
    
    Returns
    -------
    FILE_NAME_DICT: Dict[str, List[str]]
        A dictionary with keys as the different experiments 
        (abrupt4x_raw, abrupt4x_loess, control_raw, control_loess)
    '''
    FILE_NAME_METADATA_DICT = {'base_paths':{
                        'raw': constants.LONGRUNMIP_RETIMED_DIR,
                        'loess':constants.LONRUNMIP_LOESS_DIR
                                                }, 
                        'experiments' :['abrupt4x', 'control']}

    
    with open('data/longrunmip_model_lengths.json') as f:
        good_models = list(json.loads(f.read())['good_models'])
        if debug:
            print(f'Models must be one of\n{good_models}')
            print('--------')
            
    FILE_NAME_DICT = {}
    for name, base_path in FILE_NAME_METADATA_DICT['base_paths'].items():
        for exp in FILE_NAME_METADATA_DICT['experiments']:
            full_name = f'{exp}_{name}'
            if debug:
                print(full_name)
            fnames = list(map(os.path.basename, glob(os.path.join(base_path, f'*{exp}*'))))
            
            accepted_model_paths = []
            if debug:
                print('- Getting rid of models  - ', end = '')
            for fname in fnames:
                model = fname.split('_')[2]
                if model.lower() not in good_models:
                    print(f'{model}, ', end='')
                else:
                    accepted_model_paths.append(fname)
                    
            print(f'\n- Fraction of good models {len(accepted_model_paths)/ len(good_models)}')
            print('------')
            FILE_NAME_DICT[full_name] = {'base_path': base_path, 'file_names': accepted_model_paths}
            
    return FILE_NAME_DICT


def get_all_file_names_for_model(model: str, FILE_NAME_DICT: Dict[str, List[str]], 
                                debug = 0) -> Dict[str, str]:
    '''
    Given a model name and a FILE_NAME_DICT of a certain strucutre, this returns the file path
    for all the experminets in the FILE_NAME_DICT keys.
    
    Reaons
    -------
    - Gets all the different files names for a model. This is useful as all these files will often
    be used in conjuntion with each other. 
    - Created for 06_saving_consecutive_metrics_all_models_(nb24)
    
    Parameters
    ----------
    model: str
    FILE_NAME_DICT: dict
        A dictionary with keys as the different experiments, and the values as a list of
        all the different model file pahts for this experiment.
    '''
    model_fname_dict = {}
    
    # Looping through experimentins (keys) and getting the file name for each experiment for model.
    if debug:
        print('Model file names:')
    for exp_type in FILE_NAME_DICT:
        file_to_get = [f for f in FILE_NAME_DICT[exp_type]['file_names'] if model in f][0]
        if debug:
            print(f'     + {exp_type} = {file_to_get}')
        model_fname_dict[exp_type] = file_to_get
        
    return model_fname_dict
