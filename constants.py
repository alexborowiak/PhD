import os
from typing import NamedTuple

import numpy as np

PHD_ROOT_DIR = '/g/data/w40/ab2313/PhD'

LONGRUNMIP_DIR = PHD_ROOT_DIR + '/longrunmip'
ZECMIP_LOCAL_DIR = os.path.join(PHD_ROOT_DIR, 'zecmip')
ZECMIP_LOCAL_REGRIDDED_DIR = os.path.join(ZECMIP_LOCAL_DIR, 'regridded')
LONGRUNMIP_MASK_DIR = os.path.join(LONGRUNMIP_DIR, 'landesea_masks')


ZECMIP_DIR = '/g/data/oi10/replicas/CMIP6/C4MIP'
DECK_DIR = '/g/data/oi10/replicas/CMIP6/CMIP'

IMAGE_SAVE_DIR_INIT = '/g/data/w40/ab2313/images/PhD/init'
IMAGE_SAVE_DIR_TOP_LEVEL = '/g/data/w40/ab2313/images/PhD/top_level'

MODULE_DIR = '/home/563/ab2313/Documents/PhD/modules'


class PlotConfig(NamedTuple):
    title_size = 20
    label_size = 16
    cmap_title_size = 16
    legend_title_size = 16
    tick_size = 14
    legend_text_size = 14
    
# Chunks to be used for LongRunMIP when all models are loaded in. This is used
# as a lot of the calculations used with LongRunMIP rolling time.
LONGRUNMIP_CHUNKS = {'time':-1, 'lon': 144/4, 'lat': 72/2}


# Note the untis are not the base units, but the units I desire
VARIABLE_INFO = variables = {
    'tas': 
    {
        'longname': 'Near-Surface\nAir Temperature',
        'units': r'$^{\circ}C$'
    }, 
    'pr':
    {
        'longname': 'Precipitation',
        'units' : 'mm\year'
    }, 
    'netTOA':
    {
        'longname': 'Net TOA flux'
    },
    'sic': 
    {
        'longname': "Sea Ice Area Fraction",
        'units': 'fraction'
    }, 
    'psl': 
    {
        'longname': 'Sea level pressure'
    }, 
    'tos': 
    {
        'longname': 'Sea Surface Temperature',
        'units':  r'$^{\circ}C$'
        
    },
    'surf': 
    {
        'longname': 'Neat Ocean Heat Uptake',
        'units': 'W/m^2'
    }
           }

# This was created using: open_ds.get_models_longer_than_length()
# 'cesm104' removed as it has too short of a length for all variables but temperature.
LONGRUNMIP_MODELS = ['ccsm3', 'cnrmcm61', 'hadcm3l', 'ipslcm5a', 'mpiesm11', 'mpiesm12']




# NOTE: this comes from zecmip_stability_global_(single_ensemble).ipynb
ZECMIP_MODEL_PARAMS = {'GISS-E2-1-G-CC': {'value': -0.09, 'color': '#add8e6', 'linestyle': '-'},
 'CanESM5': {'value': -0.101, 'color': '#87ceeb', 'linestyle': '--'},
 'MIROC-ES2L': {'value': -0.108, 'color': '#6495ed', 'linestyle': '-'},
 'GFDL-ESM4': {'value': -0.204, 'color': '#4169e1', 'linestyle': '--'},
 'MPI-ESM1-2-LR': {'value': -0.27, 'color': '#1e90ff', 'linestyle': '-'},
 'CESM2': {'value': -0.31, 'color': '#0000cd', 'linestyle':'--'},
 'NorESM2-LM': {'value': -0.333, 'color': '#00008b','linestyle': '-'},
 'ACCESS-ESM1-5': {'value': 0.011,
  'color': [0.99137255, 0.6972549 , 0.48392157, 1.], 'linestyle': '--'},
 'UKESM1-0-LL': {'value': 0.288,
  'color': [0.78666667, 0.11294118, 0.07294118, 1.], 'linestyle': '-'}}






LONGRUNMIP_MODEL_PARAMS = {
    'gisse2r':  {'ECS' : 2.44, 'color': '#94FB33'},
    'ccsm3':    {'ECS' : 2.57, 'color': '#A0E24C'},
    'mpiesm12': {'ECS' : 2.94, 'color': '#ACC865'},
    'cesm104':  {'ECS' : 3.20, 'color': '#B8AF7E'}, 
    'hadcm3l':  {'ECS' : 3.34, 'color': '#C49597'}, 
    'mpiesm11': {'ECS' : 3.42, 'color': '#D07CB0'},
    'ipslcm5a': {'ECS' : 4.76, 'color': '#DC62C9'},
    'cnrmcm61': {'ECS' : 4.83, 'color': '#F42FFB'},
}


RANDOM_COLOR_LIST = ['springgreen', 'limegreen', 'goldenrod', 'goldenrod', 'blueviolet', 
                        'forestgreen','chartreuse', 'olive', 'fuchsia']

HEMISPHERE_LAT = {'northern_hemisphere': (0,90), 'southern_hemisphere': (-90,0), 'global': (None, None)}


EXPERIMENTS_TO_RUN = [
    {'variable': 'tas', 'mask': None, 'hemisphere': 'global'},
    {'variable': 'tas', 'mask': 'land', 'hemisphere': 'global'},
    {'variable': 'pr', 'mask': None, 'hemisphere': 'global'},
    {'variable': 'pr', 'mask': 'land', 'hemisphere': 'global'},
    {'variable': 'tos', 'mask': 'sea', 'hemisphere': 'global'},
    {'variable': 'surf', 'mask': 'sea', 'hemisphere': 'global'},
    {'variable': 'surf', 'mask': 'sea', 'hemisphere': 'northern_hemisphere'},
    {'variable': 'surf', 'mask': 'sea', 'hemisphere': 'southern_hemisphere'},
]




MULTI_WINDOW_RUN_PARAMS = dict(start_window = 11, end_window = 153, step_window=2)
ZECMIP_MULTI_WINDOW_RUN_PARAMS = {'start_window': 10, 'end_window': 41, 'step_window': 1}

# Windows that have interesing properties. These windows were decided upong from
# the graphs of the year when models and variables stabailise in the global mean.
WINDOWS_OF_INTEREST = (21, 81, 151)
LONGRUNMIP_WINDOWS = (21, 41, 81)
ZECMIP_LOCAL_RUN_WINDOWS = (20, 40)

LONGRUNMIP_LENGTH = 700 # The length of the longrunmip simulations to use
LONGRUMMIP_CONTROL_LENGHT = 500

# Need to make sure all the windows have a full length
LONGRUNMIP_EFFECTIVE_LENGTH = LONGRUNMIP_LENGTH - WINDOWS_OF_INTEREST[-1]

###### KWARGS

save_kwargs = dict(dpi=500, bbox_inches='tight')




