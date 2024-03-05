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
import signal_to_noise
import utils
logger = utils.get_notebook_logger()
from utils import pprint_list

### ZECMIP
def open_and_rename(fname, open_func, model:str=None):
    '''Open the file, then add the model dimension'''
    ds = open_func(fname)
    
    if not model: model = fname.split('/')[8]
    print(f'{model} ({fname})')
    ds = ds.expand_dims('model').assign_coords(model=('model', [model]))

    return ds

def open_mfdataset_nc(path, dropna=True, to_array=True):
    path = os.path.join(path, '*.nc')
    ds = xr.open_mfdataset(path)
    if dropna:
        da = ds.dropna(dim='time')
    if to_array:
        da = da.to_array()
    return da.squeeze()

def reformat_giss_key_for_onepct(onepct_zec_da):
    '''
    The one percent for GISS did not have a -G version. Thus, all this function does is swap the key/name around
    '''

    # TODO: This code is not very safe. 
    model_values = onepct_zec_da.model.values.astype('U16')
    model_values[np.where(model_values == 'GISS-E2-1-G')[0][0]] = 'GISS-E2-1-G-CC'

    onepct_zec_da['model'] = xr.DataArray(model_values, dims='model')
    return onepct_zec_da


### Longrunmip

def get_models_longer_than_length(experiment_length:int = 700, control_length: int = 500, debug=False) -> List[str]:
    '''
    Gets all the file names for the models longer than a certain length.
    '''
    from utils import pprint_list
    
    # A list of all the models and how long the runs for 'tas' go for.
    with open('/home/563/ab2313/Documents/PhD/data/longrunmip_model_lengths.json') as f:
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
    
    return ds


def convert_units(ds: xr.Dataset, variable: str, logginglevel='ERROR'):
    utils.change_logging_level(logginglevel)
    SECONDS_IN_YEAR = 365 * 24 * 60 * 60
    KELVIN_TO_DEGC = 273.15
    logger.debug(f'{ds}')
    if variable == 'tas':
        logger.debug('Converting from Kelvin to C')
        return ds-KELVIN_TO_DEGC
    if variable == 'pr':
        logger.info('Converting from per second to yearly total')
        ds = ds * SECONDS_IN_YEAR
        return ds
    
    if variable == 'tos':
        if ds.to_array().min().values > 200:
            logger.debug('Units are in Kelvin. Converting to DegC')
            
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

@utils.function_details
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

@utils.function_details
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
    
    
@utils.function_details    
def remove_start_time_steps(ds, number_to_remove:int):
    '''Removes the start number_to_remove points and adjsuts the time accordingly.'''
    logger.debug(f'Removing first {number_to_remove} steps')
    new_time = ds.time.values[:-number_to_remove]
    ds = ds.isel(time=slice(number_to_remove, None))
    ds['time'] = new_time

    return ds

@utils.function_details
def read_longrunmip_netcdf(fname: str, ROOT_DIR: str = '',
                           var:str = None, model_index:int = 2, 
                           requested_length:int=None, max_length: int = 1200,
                           chunks = {'lat':72/4,'lon':144/4,'time':-1},
                           mask: Union['land', 'sea'] = None,
                           logginglevel='INFO') -> xr.Dataset:
    
    utils.change_logging_level(logginglevel)
    
    fpath = os.path.join(ROOT_DIR, fname)
    logger.info(f'Opening files {fpath}')
    
    if not requested_length:
        requested_length = get_requested_length(fpath)
        logger.debug(f'{requested_length=}')

    model = fname.split('_')[model_index].lower() # Need to open da and alter length and names of vars
    model = os.path.basename(model) # Ocassionally fname can contain parts of a path
    logger.debug(f'{model=}')
    
    ds = xr.open_dataset(fpath)
        
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
                
    logger.debug(f'Rename {var=} to {model}')
    logger.debug(ds)

    ds = ds.rename({var: model})[[model]]
    ds = ds.isel(time=slice(None, max_length))
    ds = ds.squeeze()
    
    if mask:
        ds = apply_landsea_mask(ds, model, mask)
    
    ds = convert_units(ds, var, logginglevel)
    ds.attrs = {**ds.attrs, **{'length':time_length}}

    return ds

    
@utils.function_details
def read_and_merge_netcdfs(fnames: List[str], ROOT_DIR:str='', var:str=None, no_time_extension:bool=True,
                           logginglevel='INFO',*args, **kwargs) -> xr.Dataset:
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
    logger.debug(fnames)
     
    to_merge = []
    logger.debug('Time lengths')
    if no_time_extension: time_length_list = []
        
    for fname in fnames:
        try:
            ds = read_longrunmip_netcdf(fname=fname, ROOT_DIR=ROOT_DIR, var=var, 
                                        logginglevel = logginglevel, *args, **kwargs)
            time_length = len(ds.time.values)
            logger.debug(f'{fname} - {time_length}')
            if no_time_extension: time_length_list.append(time_length)
                
            to_merge.append(ds)
            
        except LongRunMIPError as e:
            logger.error(e)
            
    if len(to_merge) == 0: raise LongRunMIPError('No files found')
        
    if no_time_extension: 
        max_all_models_pressent = np.min(time_length_list)
        logger.info(f'Time length with all models present {max_all_models_pressent}')
        to_merge = [ds.isel(time=slice(None,max_all_models_pressent)) for ds in to_merge]
           
    merged_ds = xr.merge(to_merge, compat='override') 
    
    logger.info(f'Length of final dataset {len(merged_ds.time.values)}\n'
                f'Dataset has coords {list(merged_ds.coords)})\n'
                f'Dataset has data vars {list(merged_ds.data_vars)}')
       
    return merged_ds



@utils.function_details
def open_experiment_files(experiment_params: dict, experiment: ExperimentTypes, folder:str='regrid_retimestamped', models_to_get:List[str]=None,
                          max_length:int=None, no_time_extension:bool=True, 
                          combine_to_coord:bool=True, new_coord_name:str = 'model',
                          keep_time_stamp:bool=True, logginglevel='INFO'):
    
    '''
    Gets all the models for an experiment type.
    Use this in conjunction with constants.EXPERIMENTS_TO_RUN. These are the different experiment params that 
    get used.
    
    
    experiment: bool = True
        This will match the lengths of all the datasets. E.g. the model that has the shorted
        run will be the final length of the data set. This is needed for the detredning methods
        that will not work if nans have been inserted onto the end.
    keep_time_stamp: bool=False
        For some reason the cftime is now not plotting. Also, having an actual time stamp doesn't matter.
        Can just use an integer value anyway
    no_time_extension:bool=True
        Do not extend the time for any dataset. Clip the dataset to the shortest data set
    combine_to_coord: bool
        Combine all data vars into a new coordinate with name 'new_coord_name' (default = 'model')
    '''
    print('\n')
    
    utils.change_logging_level(logginglevel)
    logger.info(f'Opening {experiment}')

    if not models_to_get: models_to_get = get_models_longer_than_length()
    logger.info(f'{models_to_get}=')
        
    ROOT_DIR = os.path.join(constants.LONGRUNMIP_DIR, experiment_params["variable"], folder)
    logger.info(f'{ROOT_DIR}=')
    
    files_to_open = get_file_names_from_from_directory(ROOT_DIR, experiment, models_to_get)
    logger.debug(f'{files_to_open}=')

    ds = read_and_merge_netcdfs(files_to_open, ROOT_DIR, mask=experiment_params['mask'], max_length=max_length, 
                               no_time_extension=no_time_extension)
    
    if not keep_time_stamp:
        logger.debug('Chning time from time stamp to integer value between o and length of dataset')
        ds['time'] = range(len(ds.time.values)) # Integer value between 0 and length of dataset

    if combine_to_coord:
        ds = ds.to_array(name=experiment_params['variable']).rename({'variable': new_coord_name})
#         ds.name = experiment_params['variable']
        logger.info(f'Converted to datarray with {new_coord_name=} and name')
    
    return ds


    

def get_mean_for_experiment(ROOT_DIR:str, experiment_params:Dict[str, Union[str, float]], models_to_get:List[str],
                            max_length:int=None):
    '''Gets the global mean/value of experiment and picontrol. '''
    
    control_ds = open_experiment_files(experiment_params, ExperimentTypes.CONTROL, models_to_get=models_to_get)
    abrupt4x_ds = open_experiment_files(experiment_params, ExperimentTypes.ABRUPT4X, models_to_get=models_to_get)

    # Getting the glboal value (mean for all other variables other than sic)
    abrupt4x_mean,control_ds_mean = signal_to_noise.calculate_global_value(
        abrupt4x_ds, control_ds, experiment_params["variable"],
        constants.HEMISPHERE_LAT[experiment_params['hemisphere']])
    
    return abrupt4x_mean,control_ds_mean


def get_experiment_name_from_params(experiment_params: Dict[str, str]) -> str:
    '''Converts the expeimenet dict name to a string name of just the values.'''
    experiment_name = '' 

    for key, value in experiment_params.items():
        value = value + '_' if value else ''
        experiment_name = f'{experiment_name}{value}'
    experiment_name= experiment_name[:-1]
    
    return experiment_name


def get_all_experiment_ds(experiments_to_run, directory, models_to_get, max_length:int=None):
    '''
    Get all the different expereiments and merges them into a xr.Dataset. This Dataset
    has a model coodinate, and all the data vars are for each periment. The name of each si 
    created using get_experiment_name_from_params (e.g. tas_land_global)
    '''

    to_merge_experiment = []
    to_merge_control = []
    for experiment_params in experiments_to_run:
        print(f'\n- {experiment_params}')

        abrupt4x_mean, control_ds_mean = get_mean_for_experiment(
            directory, experiment_params, models_to_get, max_length=max_length)
        
        experiment_name = get_experiment_name_from_params(experiment_params)
        
        # Change models from data_var to coord. Then make into dataset.
        experiment_ds = abrupt4x_mean.to_array(dim='model').to_dataset(name=experiment_name)
        control_ds = control_ds_mean.to_array(dim='model').to_dataset(name=experiment_name)
        to_merge_experiment.append(experiment_ds)
        to_merge_control.append(control_ds)

    all_experiment_ds = xr.merge(to_merge_experiment).drop(['height', 'depth'], errors='ignore')
    all_control_ds = xr.merge(to_merge_control).drop(['height', 'depth'], errors='ignore')
    
    return all_experiment_ds, all_control_ds

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

    if isinstance(ds, xr.DataArray): ds = ds.to_dataset()
    
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
    # TODO: The month and hour should be 1
    t0_new = cftime.datetime(1, 1, 1, 0, 0, 0, 0, calendar='gregorian')
    # New end time is the total length of the old dataset added to t0
    tf_new = t0_new + (tf - t0)
    if debug: print(f'New time dim will range between {t0_new} and {tf_new}')
    
    new_time = xr.cftime_range(start=t0_new, end=tf_new, periods=len(ds['time'].values), freq=None)
    
    if debug:
        print('Old time values')
        print(ds['time'].values[:5])
        print(ds['time'].values[-5:])
        print('New time values')
        print(new_time[:4])
        print(new_time[-4:])
    
    ds['time'] = new_time
    
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
#         da = remove_unwated_coords(da)

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

def open_signal_to_noise_dataset(files, ROOT_DIR):
    '''
    Simple open and contat all files in a directory with a rename and squeeze
    TODO: Remove this once figured out why model is called variable.
    '''
    to_concat = []

    for file in files:
        ds = (xr.open_dataset(os.path.join(ROOT_DIR, file))
            .rename({'variable':'model'}))

        to_concat.append(ds)
        
    return xr.concat(to_concat, dim='model')

