# +
# Directories
import os
PHD_ROOT_DIR = '/g/data/w40/ab2313/PhD'

LONGRUNMIP_DIR = PHD_ROOT_DIR + '/longrunmip'
ZECMIP_LOCAL_DIR = os.path.join(PHD_ROOT_DIR, 'zecmip')
ZECMIP_LOCAL_REGRIDDED_DIR = os.path.join(ZECMIP_LOCAL_DIR, 'regridded')
LONGRUNMIP_MASK_DIR = os.path.join(LONGRUNMIP_DIR, 'landesea_masks')


ZECMIP_DIR = '/g/data/oi10/replicas/CMIP6/C4MIP'
DECK_DIR = '/g/data/oi10/replicas/CMIP6/CMIP'

IMAGE_SAVE_DIR_INIT = '/home/563/ab2313/gdata/images/PhD/init'
IMAGE_SAVE_DIR_TOP_LEVEL = '/home/563/ab2313/gdata/images/PhD/top_level'

MODULE_DIR = '/home/563/ab2313/Documents/PhD/modules'


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



MODEL_PARAMS = {
    'gisse2r':  {'ECS' : 2.44, 'color': '#94FB33'}, # springgreen
    'ccsm3':    {'ECS' : 2.57, 'color': '#A0E24C'},#limegreen
    'mpiesm12': {'ECS' : 2.94, 'color': '#ACC865'},#goldenrod
    'cesm104':  {'ECS' : 3.20, 'color': '#B8AF7E'}, #blueviolet
    'hadcm3l':  {'ECS' : 3.34, 'color': '#C49597'}, #forestgreen
    'mpiesm11': {'ECS' : 3.42, 'color': '#D07CB0'},#chartreuse
    'ipslcm5a': {'ECS' : 4.76, 'color': '#DC62C9'},#olive
    'cnrmcm61': {'ECS' : 4.83, 'color': '#F42FFB'},#fuchsia

}

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



# This was created using: open_ds.get_models_longer_than_length()
# 'cesm104' removed as it has too short of a length for all variables but temperature.
LONGRUNMIP_MODELS = ['ccsm3', 'cnrmcm61', 'hadcm3l', 'ipslcm5a', 'mpiesm11', 'mpiesm12']
ZECMIP_MODELS = ['CESM2',
                 #'CanESM5', Model is too short. Lenght ~ 100 years
                 'GFDL-ESM4',
                 'GISS-E2-1-G-CC',
                 'MIROC-ES2L',
                 'MPI-ESM1-2-LR',
                 # 'NorESM2-LM',Model is too short. Lenght ~ 100 years
                 'UKESM1-0-LL']

MULTI_WINDOW_RUN_PARAMS = dict(start_window = 10, end_window = 152, step_window=1)


# Windows that have interesing properties. These windows were decided upong from
# the graphs of the year when models and variables stabailise in the global mean.
WINDOWS_OF_INTEREST = [20, 80, 150]#[20, 150, 300]
LONGRUNMIP_LENGTH = 800 # The length of the longrunmip simulations to use
# Need to make sure all the windows have a full length
LONGRUNMIP_EFFECTIVE_LENGTH = LONGRUNMIP_LENGTH - WINDOWS_OF_INTEREST[-1]



###### KWARGS

save_kwargs = dict(dpi=500, bbox_inches='tight')
