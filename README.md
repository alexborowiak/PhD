# Signal to Noise


Signal to noise analysis of LongRunMIP and ZECMIP Data.

=================================

## LongRunMIP


** Opening Data **

In 00_3_make_longrunmip_length_file_and_sample_open_ds.ipynb a file is made called data/longrunmip_model_lengths.json.
This file contains the lengths of the different model lengths for tas variable. Functions using this file can then be
called with modules/open_ds.py

```
import open_ds
import utils
from classes import ExperimentTypes

=========

# Gets all models with default requested_legnth = 700.
models_to_get = open_ds.get_models_longer_than_length()
utils.pprint_list(models_to_get, num_start_items=7)
>>> lenght = 7
     0. ccsm3
     1. cesm104
     2. cnrmcm61
     3. hadcm3l
     
     
========

files_to_open = open_ds.get_file_names_from_from_directory(ROOT_DIR, ExperimentTypes.ABRUPT4X, models_to_get)
utils.pprint_list(files_to_open)
>>> lenght = 7
     0. pr_mon_CCSM3_abrupt4x_2120_g025.nc
     1. pr_mon_CESM104_abrupt4x_5900_g025.nc
     
========     

# The root dir should contain exactly what directory. THe pr is needed to be passed in 
# for renaming of the variable to the model name. This is needed to be done as they are 
# to large to merge into a single coordinate all the files.
ds = open_ds.read_and_merge_netcdfs(files_to_open, ROOT_DIR, var='pr')
print(ds)

>>> <xarray.Dataset>
    Dimensions:   (time: 4459, lon: 144, lat: 72)
    Coordinates:
      * time      (time) object 0001-12-31 00:00:00 ... 4459-12-31 00:00:00
      * lon       (lon) float32 1.25 3.75 6.25 8.75 ... 351.2 353.8 356.2 358.8
      * lat       (lat) float32 -88.75 -86.25 -83.75 -81.25 ... 83.75 86.25 88.75
    Data variables:
        ccsm3     (time, lat, lon) float32 3.206e-06 3.112e-06 3.112e-06 ... nan nan
        cesm104   (time, lat, lon) float32 2.112e-06 2.105e-06 2.092e-06 ... nan nan
        cnrmcm61  (time, lat, lon) float32 2.589e-06 2.589e-06 2.597e-06 ... nan nan
        hadcm3l   (time, lat, lon) float64 6.821e-07 6.706e-07 6.619e-07 ... nan nan
        ipslcm5a  (time, lat, lon) float32 1.623e-06 1.594e-06 1.589e-06 ... nan nan
        mpiesm11  (time, lat, lon) float32 ...
        mpiesm12  (time, lat, lon) float32 1.35e-06 1.354e-06 1.363e-06 ... nan nan
    Attributes:
        length:   2120

```