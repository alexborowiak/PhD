import json
from pathlib import Path
from functools import partial
import xarray as xr

import utils
from listXarray import listXarray
from typing import Dict, Optional, List

logger = utils.get_notebook_logger()

import xarray as xr

def __preprocess(ds: xr.Dataset, variable: str, model: str, ensemble:str) -> xr.Dataset:
    """
    Preprocesses a dataset for a specific variable and model.
    This function performs various preprocessing steps on the dataset, including:
    - Selecting the specified variable from the dataset and removing any extra dimensions (squeeze).
    - Expanding the dataset with a new 'model' dimension and setting 'model' coordinate.
    - Removing the 'height' variable if it exists in the dataset.
    Parameters:
        ds (xr.Dataset): The xarray dataset to preprocess.
        variable (str): The name of the variable to select and process.
        model (str): The model name associated with the dataset.
    Returns: xr.Dataset: The preprocessed xarray dataset.
    """
    ds = ds[variable].drop(['height'], errors='ignore').squeeze()
    
    # Expand the dataset with a new 'model' dimension and set 'model' coordinate.
    ds = ds.expand_dims('model').assign_coords(model=('model', [model]))
    ds = ds.expand_dims('ensemble').assign_coords(ensemble=('ensemble', [ensemble]))

    return ds



def extract_experiment_into_xrlist(
    experiment_str: str, variable:str, zecmip_model_paths: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    logginglevel:str='ERROR', chunks:Dict[str, str]='auto', 
) -> listXarray:
    """
    Extracts experiment data into a list of xarray datasets for each model and ensemble.

    Parameters:
        experiment_str (str): The experiment name to extract data for.
        zecmip_model_paths (dict, optional): A dictionary containing model paths for zecmip experiments.
            If not provided, it will read the paths from a default JSON file.

    Returns:
        list: A list of xarray datasets, each representing data for a specific model.
    """
    
    utils.change_logginglevel(logginglevel)
    if zecmip_model_paths is None:
        default_json_path = Path.cwd() / 'Documents' / 'PhD' / 'data' / f'zecmip_experiment_paths_ensemble_sorted_{variable}.json'
        with default_json_path.open('r') as f:
            zecmip_model_paths = json.load(f)

    model_datasets = []
    
    for model, ensemble_paths in zecmip_model_paths.items():
        logger.info(model)
        ensemble_datasets = []
        for ensemble, path in ensemble_paths.get(experiment_str, {}).items():
            path = str(Path(path).joinpath('*.nc')) # convert the path string to a path object and join with nc
            logger.info(f'     {ensemble} - {path}')
            
            preprocess = partial(__preprocess, variable=variable, model=model, ensemble=ensemble)
            try:
                ds = xr.open_mfdataset(path, preprocess=preprocess, chunks=chunks, use_cftime=True)
            except TypeError:
                ds = xr.open_mfdataset(path, chunks=chunks, preprocess=preprocess) 
            ensemble_datasets.append(ds)
        
        model_datasets.append(xr.concat(ensemble_datasets, dim='ensemble'))
    
    return listXarray(model_datasets, 'model')


