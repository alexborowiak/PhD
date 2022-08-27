# +
# Directories
import os
PHD_ROOT_DIR = '/g/data/w40/ab2313/PhD'

LONGRUNMIP_DIR = PHD_ROOT_DIR + '/longrunmip'
ZECMIP_LOCAL_DIR = os.path.join(PHD_ROOT_DIR, 'zecmip')

ZECMIP_DIR = '/g/data/oi10/replicas/CMIP6/C4MIP'
DECK_DIR = '/g/data/oi10/replicas/CMIP6/CMIP'

IMAGE_SAVE_DIR_INIT = '/home/563/ab2313/gdata/images/PhD/init'
IMAGE_SAVE_DIR_TOP_LEVEL = '/home/563/ab2313/gdata/images/PhD/top_level'

MODULE_DIR = '/home/563/ab2313/Documents/PhD/modules'

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
