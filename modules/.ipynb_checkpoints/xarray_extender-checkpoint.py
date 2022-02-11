import xarray as xr

def xr_dict_to_xr_dataset(data:dict):
    '''Takes a dicionary that has the model name and then model values
    as the value and merges them into an xarray dataset with the data_vars as the mode
    Parameters
    ----------
    dict '''
    to_merge = []
    for model, value in data.items():
        value.name = model
        to_merge.append(value)

    return xr.merge(to_merge, compat='override')
