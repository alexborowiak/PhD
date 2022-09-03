#!/bin/sh


# FPATH="tas_mon_ECHAM5MPIOM_control_100_g025.nc"
FPATH=$1

python3 -W ignore open_xarray_and_do_something.py $FPATH