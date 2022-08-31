import xarray as xr
import numpy as np
import sys, os

fname = sys.argv[1]


def do_something(ds):
    ds = ds
    return ds + 1

def open_ds(fname):
    return xr.open_dataset(fname)


def main():
    ds = open_ds(fname)
    ds = do_something(ds)
    ds.to_netcdf('done_something' + fname)
    
    
    
if __name__ == '__main__':
    main()