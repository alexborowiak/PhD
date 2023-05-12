import os
from typing import NamedTuple

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
ZECMIP_MODELS = ['CESM2',
                 'CanESM5', # Model is short. Lenght ~ 100 years
                 'GFDL-ESM4',
                 'GISS-E2-1-G-CC',
                 'MIROC-ES2L',
                 'MPI-ESM1-2-LR',
                 'NorESM2-LM' #Model is  short. Lenght ~ 100 years
                 'UKESM1-0-LL']



ZECMIP_MODEL_PARAMS = {
     'NorESM2-LM':     {'ECS': 2.54, 'color': '#FDFD96'},
     'MIROC-ES2L':     {'ECS': 2.7,  'color': '#FFC926'},
     'MPI-ESM1-2-LR':  {'ECS': 2.83, 'color': '#FF8C00'},
     'GISS-E2-1-G-CC': {'ECS': 2.9,  'color': '#FF5733'}, # ECS value needs revision
     'GFDL-ESM4':      {'ECS': 3.1,  'color': '#FF2A00'},
     'CESM2':          {'ECS': 5.1,  'color': '#B90000'},
     'UKESM1-0-LL':    {'ECS': 5.4,  'color': '#7F0000'},
     'CanESM5':        {'ECS': 5.7,  'color': '#3F0000'},
 }

# ZECMIP_MODEL_PARAMS = {
#      'NorESM2-LM':     {'ECS': 2.54, 'color': '#94FB33', 'linestyle': 'solid'},
#      'MIROC-ES2L':     {'ECS': 2.7,  'color': '#A0E24C', 'linestyle': 'dashed'},
#      'MPI-ESM1-2-LR':  {'ECS': 2.83, 'color': '#ACC865', 'linestyle': 'dashdot'},
#      'GISS-E2-1-G-CC': {'ECS': 2.9,  'color': '#B8AF7E', 'linestyle': 'dotted'}, # ECS value needs revision
#      'GFDL-ESM4':      {'ECS': 3.1,  'color': '#C49597', 'linestyle': (0, (3, 1, 1, 1))},
#      'CESM2':          {'ECS': 5.1,  'color': '#D07CB0', 'linestyle': (0, (3, 1, 1, 1, 1, 1))},
#      'UKESM1-0-LL':    {'ECS': 5.4,  'color': '#DC62C9', 'linestyle': (0, (1, 1))},
#      'CanESM5':        {'ECS': 5.7,  'color': '#F42FFB', 'linestyle': (0, (1, 3))},
# }

# ZECMIP_MODEL_PARAMS = {
#      'NorESM2-LM':     {'ECS': 2.54, 'color': '#94FB33'},
#      'MIROC-ES2L':     {'ECS': 2.7,  'color': '#A0E24C'},
#      'MPI-ESM1-2-LR':  {'ECS': 2.83, 'color': '#ACC865'},
#      'GISS-E2-1-G-CC': {'ECS': 2.9,  'color': '#B8AF7E'}, # ECS value needs revision
#      'GFDL-ESM4':      {'ECS': 3.1,  'color': '#C49597'},
#      'CESM2':          {'ECS': 5.1,  'color': '#D07CB0'},
#      'UKESM1-0-LL':    {'ECS': 5.4,  'color': '#DC62C9'},
#      'CanESM5':        {'ECS': 5.7,  'color': '#F42FFB'},
#  }

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

# SIC GLobal calculations are incorrect. Removing unitl fix has been implemented.
#     {'variable': 'sic', 'mask': 'sea', 'hemisphere': 'global'},
#     {'variable': 'sic', 'mask': 'sea', 'hemisphere': 'northern_hemisphere'},
#     {'variable': 'sic', 'mask': 'sea', 'hemisphere': 'southern_hemisphere'}





MULTI_WINDOW_RUN_PARAMS = dict(start_window = 11, end_window = 153, step_window=2)
ZECMIP_MULTI_WINDOW_RUN_PARAMS = dict(start_window = 11, end_window = 51, step_window=2)

# Windows that have interesing properties. These windows were decided upong from
# the graphs of the year when models and variables stabailise in the global mean.
WINDOWS_OF_INTEREST = (21, 81, 151)
LONGRUNMIP_WINDOWS = (21, 41, 81)
ZECMIP_WINDOWS = (21, 41)
LONGRUNMIP_LENGTH = 800 # The length of the longrunmip simulations to use
# Need to make sure all the windows have a full length
LONGRUNMIP_EFFECTIVE_LENGTH = LONGRUNMIP_LENGTH - WINDOWS_OF_INTEREST[-1]

###### KWARGS

save_kwargs = dict(dpi=500, bbox_inches='tight')
