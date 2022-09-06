# +
# Directories
import os
PHD_ROOT_DIR = '/g/data/w40/ab2313/PhD'

LONGRUNMIP_DIR = PHD_ROOT_DIR + '/longrunmip'
ZECMIP_LOCAL_DIR = os.path.join(PHD_ROOT_DIR, 'zecmip')
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
        'longname': 'Near-surface air temperature',
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
        'longname': 'Sea surface temperature',
        'units':  r'$^{\circ}C$'
        
    },
    'surf': 
    {
        'longname': 'Neat Ocean Heat Uptake'
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
]

# SIC GLobal calculations are incorrect. Removing unitl fix has been implemented.
#     {'variable': 'sic', 'mask': 'sea', 'hemisphere': 'global'},
#     {'variable': 'sic', 'mask': 'sea', 'hemisphere': 'northern_hemisphere'},
#     {'variable': 'sic', 'mask': 'sea', 'hemisphere': 'southern_hemisphere'}



# This was created using: open_ds.get_models_longer_than_length()
# 'cesm104' removed as it has too short of a length for all variables but temperature.
LONGRUNMIP_MODELS = ['ccsm3', 'cnrmcm61', 'hadcm3l', 'ipslcm5a', 'mpiesm11', 'mpiesm12']





# Old

# LONGRUNMIP_RETIMED_DIR = LONGRUNMIP_DIR + '/regrid_retimestamped'
# LONRUNMIP_LOESS_DIR = LONGRUNMIP_RETIMED_DIR + '/loess'
# LONGRUNMIP_CONSECMET_DIR = os.path.join(LONGRUNMIP_DIR, 'consecutive_metrics')
# ZECMIP_LOCAL_REGRIDDED_DIR= os.path.join(ZECMIP_LOCAL_DIR, 'regridded')

# Subset of the models that have at least 1850 years in length
# LONGRUMIP_MODELS_MIN_1850 = ['ccsm3', 'cesm104', 'cnrmcm61', 'famous', 'gisse2r', 'mpiesm11']
# LONGRUNMIP_MODELS = []

# Models that do not have a long enough control run.
# MODELS_TO_DROP = ['hadgem2', 'echam5mpiom']
