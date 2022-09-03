import numpy as np
import pandas as pd
import xarray as xr
import cftime
from typing import List, Dict, Union
import os
from glob import glob
import constants
import json
from enum import Enum
from classes import ExperimentTypes, LongRunMIPError

import utils
logger = utils.get_notebook_logger()



### Longrunmip

def get_models_longer_than_length(experiment_length: int = 700, control_length: int = 500, debug=False) -> List[str]:
    '''
    Gets all the file names for the models longer than a certain length.
    '''
    from utils import pprint_list
    
    # A list of all the models and how long the runs for 'tas' go for.
    with open('data/longrunmip_model_lengths.json') as f:
        longrunmip_model_lengths = json.loads(f.read())
        
        # Gtting only the models where the controla dn 4xCO2 are longer than requested_length
        good_models = {model: len_obj for model, len_obj in longrunmip_model_lengths.items() 
                       if len_obj['control'] > control_length
                       and len_obj['4xCO2'] > experiment_length}
        
        good_models = list(good_models.keys())
        
        # The model famous is not wanted
        good_models = np.array(good_models)[np.array(good_models) != 'famous']
        if debug:
            print(f'Models with min length {requested_legnth}:')
            pprint_list(good_models)
                            
    return good_models



    
def get_file_names_from_from_directory(ROOT_DIR, experiment: ExperimentTypes, 
                                       models: List[str], logginglevel='ERROR') -> List[str]:
    '''Gets all file names for a model from a particular diretory'''
    
    utils.change_logging_level(logginglevel)
    
    if 'signal_to_noise' in experiment.value:
        ROOT_DIR = os.path.join(ROOT_DIR, experiment.value)
        
    logger.info(f'Getting files from {ROOT_DIR}')
    files_in_directory = os.listdir(ROOT_DIR)
    logger.debug(utils.pprint_list_string(files_in_directory))
    paths_to_return = []
    
    for model in models:
        model = model.lower()
        found_fname = None
        for fname in files_in_directory:
            logger.debug(f'{model} - {experiment.value.lower()} - {fname}')
            if model in fname.lower() and experiment.value.lower() in fname.lower():
                logger.debug('Found match')
                found_fname = fname 
                break
                
        if found_fname:
            paths_to_return.append(found_fname)
            
            logger.debug(f'{model=} - {found_fname=}')
        else:
            logger.error(f'{model=} - {found_fname=} - No file found')
            
    if 'signal_to_noise' in experiment.value:
        paths_to_return =[os.path.join(experiment.value, fname) for fname in paths_to_return]

        
    return paths_to_return


def open_dataset(fpath: str) -> xr.Dataset:
    '''
    Tries to open with cf_time, otherwise will not. Works for multi-dataset
    '''
    
    open_function = xr.open_mfdataset if isinstance(fpath, list) else xr.open_dataset
     
    # TODO: Need to figure out what the error is with files having a string as timestep.
    try:
        ds = open_function(fpath, use_cftime=True)
        return ds.squeeze()
    except ValueError as e:
        print(f'{os.path.basename(fpath)} has failed with ValueError')
        
#     except:
#         # Certain files have an object (string) as the datetime - they will error with cftime.
#         ds = open_function(fpath)
        
#     return ds


def convert_units(ds: xr.Dataset, variable: str, logginglevel='ERROR'):
    utils.change_logging_level(logginglevel)
    SECONDS_IN_YEAR = 365 * 24 * 60 * 60
    KELVIN_TO_DEGC = 273.15
    logger.debug(f'{ds}')
    if variable == 'tas':
        logger.info('Converting from Kelvin to C')
        return ds-KELVIN_TO_DEGC
    if variable == 'pr':
        logger.info('Converting from per second to yearly total')
        ds = ds * SECONDS_IN_YEAR
        return ds
    
    if variable == 'tos':
        if ds.to_array().min().values > 200:
            logger.info('Units are in Kelvin. Converting to DegC')
            
            return ds-KELVIN_TO_DEGC
    
    if variable == 'sic':
        # This means value is in percent not as a fraction
        if ds.to_array().max().values > 1.5:
            return ds/100
            
    return ds
    
    


def get_requested_length(fname: str):
    if 'control' in fname:
        return 100
    return 800



def get_mask_for_model(model: str) -> xr.Dataset:
    '''
    Opens the land sea mask for a specific model found in the directory
    constants.LONGRUNMIP_MASK_DIR
    '''
    mask_list = os.listdir(constants.LONGRUNMIP_MASK_DIR)
    model_mask_name = [fname for fname in mask_list if model.lower() in fname.lower()]
    
    if len(model_mask_name) == 0:
        raise IndexError(f'No mask found for model {model_mask_name=}')

    model_mask_name = model_mask_name[0]
 
    mask_ds = xr.open_dataset(os.path.join(constants.LONGRUNMIP_MASK_DIR, model_mask_name))
    return mask_ds

def  apply_landsea_mask(ds: xr.Dataset, model:str, mask: Union['land', 'sea'] = None):
    '''
    Applies either a land or sea mask to the dataset. 
    '''
    mask_ds = get_mask_for_model(model)
    
    if mask == 'land':
        return ds.where(mask_ds.mask == 1)
    if  mask == 'sea':
        return ds.where(mask_ds.mask != 1)
    raise ValueError(f'{mask} is not a valid mask option. Please use either [land, sea]')
        
def read_longrunmip_netcdf(fname: str, ROOT_DIR: str = '',
                           var:str = None, model_index:int = 2, 
                           requested_length:int=None, max_length: int = 1200,
                           chunks = {'lat':72/4,'lon':144/4,'time':-1},
                           mask: Union['land', 'sea'] = None,
                           logginglevel='ERROR') -> xr.Dataset:
    
    utils.change_logging_level(logginglevel)
    
    fpath = os.path.join(ROOT_DIR, fname)
    
    logger.info(f'Opening files {fpath}')
    
    if not requested_length:
        requested_length = get_requested_length(fpath)
        logger.info(f'{requested_length=}')


    # Need to open da and alter length and names of vars
    model = fname.split('_')[model_index].lower()
    
    # Ocassionally fname can contain parts of a path
    model = os.path.basename(model)
    
    ds = xr.open_dataset(fpath)
    
    def remove_start_time_steps(ds, number_to_remove:int):
        logger.error(f'Removing first 10 steps for {fname}')
        new_time = ds.time.values[:-number_to_remove]
        ds = ds.isel(time=slice(number_to_remove, None))
        ds['time'] = new_time
        
        return ds
    
        
    
    # First few time stemps of ccsm3 control are not in equilibrium
    # TODO: Better to just remove this in the procesing step
    if 'control' in fname.lower() and 'ccsm3' in fname.lower():
        ds = remove_start_time_steps(ds, 10)
    
    if 'tos' in fname.lower() and 'abrupt4x' in fname.lower() and 'ipslcm5a' in fname.lower():
        ds = remove_start_time_steps(ds, 200)

    
    time_length = len(ds.time.values)
    if time_length < requested_length:
        raise LongRunMIPError(f"{model=} is too short has {time_length=} < {requested_length=}\n({fname=})")
    
    if var is None:
        var = list(ds.data_vars)[0]
                
    logger.info(f'Rename {var=} to {model}')

    logger.debug(ds)

    ds = ds.rename({var: model})[[model]]

    # Some dataset are too long. Getting all data is a waste
    ds = ds.isel(time=slice(None, max_length))
    
    ds = ds.squeeze()
    
    if mask:
        ds = apply_landsea_mask(ds, model, mask)
    
    ds = convert_units(ds, var, logginglevel)
     
    ds.attrs = {**ds.attrs, **{'length':time_length}}

    return ds

    
#     if var is None:
#         var = fname[0].split('_')[0]
#         logger.info(f'Making assumptions on var {var=}')
        

def read_and_merge_netcdfs(fnames: List[str], ROOT_DIR: str = '', var:str=None,
#                            var:str = None, model_index:int = 2, 
#                            requested_length:int=None, max_length: int = 1200,
                           logginglevel='ERROR',*args, **kwargs) -> xr.Dataset:
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
 
    utils.change_logging_level(logginglevel)
    
    logger.info(f'Opening files in {ROOT_DIR}')
     
    to_merge = []
    for fname in fnames:
        try:
            da = read_longrunmip_netcdf(fname=fname, ROOT_DIR=ROOT_DIR, var=var, 
                                        logginglevel = logginglevel, *args, **kwargs)
#                            var, model_index, 
#                            requested_length, max_length,
#                            logginglevel)
            to_merge.append(da)
            
        except LongRunMIPError as e:
            logger.error(e)
            
    if len(to_merge) == 0:
        raise LongRunMIPError('No files found')
           
    merged_ds = xr.merge(to_merge, compat='override') 
       
    return merged_ds



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
    possible_non_time_dims = ['lon', 'lat', 'long', 'longitude', 'latitude', 'lev', 'bnds','bounds', 'model',
                             'LON', 'DEPTH', 'depth', 'LAT', 'LATITUDE', 'LAT', 'height', 'z']
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


def open_and_concat_nc_files(nc_files: List[str], ROOT_DIR: str='', model_index=-1, logginglevel='ERROR'):
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
    
    utils.change_logging_level(logginglevel)
    xr_files = {}
    
    
    for f in nc_files:
        logger.info(f'Opening {f}')
        model = os.path.basename(f).split('_')[model_index]
        if '.' in model:
            model = model.split('.')[0]
        logger.debug(f'{model=}')
        da = xr.open_dataset(os.path.join(ROOT_DIR, f))
        da = remove_unwated_coords(da)

        xr_files[model] = da
    logger.debug(f'Merging together {list(xr_files)}')
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



