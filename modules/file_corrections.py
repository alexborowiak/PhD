'''
Standardising the format of all data. 

''''



import xarray as xr


def open_and_correct_dataset(ROOT_DIR, fname, **kwargs)
    
    '''This function opens and makes a dataset of a standard format.'''
    
    # Multipole datasets - want to use multifile open
    if len(fname) > 1:
        open_function = xr.open_mfdataset
    else:
        open_function = xr.open_dataset
    
    # Opening dataset
    try:
        ds = open_function(os.path.join(ROOT_DIR, fname), use_cftime=True)
    except:
        # Certain files have an object (string) as the datetime - they will error with cftime.
        ds = open_function(os.path.join(ROOT_DIR, fname))
        
    ### Correcting timesteps
    if 'mon' in fname:
        print(' monthly')
        freq = 'M'
        
    else:
        print(' annual')
        freq = 'Y'
        
    # The name of the dim can be different in each file. So need to get the dim that isn't lat or lon.
    dims =  np.array(list(ds.dims.keys()))
    
    # This should be the time dim in the dataset.
    dim_to_override = dims[~np.isin(dims, ['lon', 'lat', 'long', 'longitude', 'latitude', 'lev'])][0]
    
    
    # News time range
    # Defining the start of the new time series.
    t0 = cftime.DatetimeNoLeap(0, 1, 1, 0, 0, 0, 0, has_year_zero=True)
    new_time = xr.cftime_range(t0, periods = len(ds[dim_to_override].values), freq=freq)
    
    
    print(f'- Time dim is currently: {dim_to_override}')
    print(ds[dim_to_override].values[:5])
    print(ds[dim_to_override].values[-5:])
    print('- New time dim of')
    print(new_time[:2])
    print(new_time[-2:])
    
    ds[dim_to_override] =  new_time
    
    ds = ds.rename({dim_to_override: 'time'})
    
    if 'mon' in file:
        ds = ds.resample(time='Y').mean()
        
    print(ds)
    
    # Saving the dataset
    if 'OUTPUT_DIR' in kwargs.keys():
        ds.to_netcdf(os.path.join(kwargs['OUTPUT_DIR'], file))
    # Returning the dataset for local use.  
    else:
        return ds

    