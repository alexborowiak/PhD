import os
import json

import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Callable, Union, Dict, Optional
from numpy.typing import ArrayLike

import utils
import xarray_extender

logger = utils.get_notebook_logger()

class listXarray():
    def __init__(self, xr_list:List, key_dim:List[str]=None, refkeys:Optional[ArrayLike]=None, logginglevel:str='ERROR') -> None:
        """
        Initialize the listXarray object.
        Parameters:
            xr_list (List[xr.DataArray]): A list of xarray DataArrays.
            key_dim (List[str], optional): List of key dimensions for reference. Defaults to None.
            
        TODO: This is potentially a bad way of doing this. A dictionary might have been better.
        """
        
        utils.change_logginglevel(logginglevel)
        self.xr_list = xr_list
        self.key_dim = key_dim
        
        logger.info(self.key_dim)
        logger.debug(self.xr_list)

        if refkeys is None: # Only try this if not manually setting refkeys
            if key_dim:
                try:
                    self.refkeys = np.array([ds[key_dim].values[0] for ds in xr_list])
                except IndexError:
                    # Potential that the model values aren't stored as numpy arrays.
                    self.refkeys = np.array([str(ds[key_dim].values) for ds in xr_list])
            else:
                self.refkeys = None
        else:
            # Refkeys have been manually assigned
            self.refkeys = refkeys
            
        logger.info(self.refkeys)

    def set_refkeys(self, key_dim, refkeys=None, logginglevel:str='ERROR'):
        """
        Set new key dimensions for the listXarray.
        Parameters:
            key_dim (List[str]): List of key dimensions for reference.
        Returns:
            listXarray: A new instance of listXarray with the updated key dimensions.
        """
        return listXarray(self.xr_list, key_dim, refkeys=None, logginglevel='ERROR')
    
    def sort_by_refkey(self):
        '''
        Makes the refkeys and the corresponding ds sorted by the refkey
        '''

        refkeys = self.refkeys

        # Pair each DataArray with its corresponding refkey
        paired_list = list(zip(refkeys, self.xr_list))

        # Sort the paired list by refkeys
        sorted_paired_list = sorted(paired_list, key=lambda x: x[0])

        # Extract the sorted DataArrays from the paired list
        sorted_xr_list = [item[1] for item in sorted_paired_list]

        return listXarray(sorted_xr_list, self.key_dim)
    
        
    def dim(self, dim:str, output='print'):
        '''
        Will print out the values for dim. This can be put into a
        dictionary or can be just printed
        '''
        if output == 'dict': dobj = {}
        for refkey, ds in zip(self.refkeys, self.xr_list):
            if output=='print': print(f'{refkey} - {ds[dim].values}')
            if output == 'dict': dobj[refkey] = ds[dim].values
        if output == 'dict': return dobj
    
    def rename(self, name_dict:dict):
        '''
        Rename the variable name from old to new name
        '''
        to_return  = []
        for ds in self.xr_list:
            try:
                ds = ds.rename(name_dict)
            except ValueError:
                pass
            to_return.append(ds)
        return listXarray(to_return, self.key_dim)
            
    
    def cross_match_dim(self, listxr_to_match, dim:str):
        '''
        This is to be applied across different listXarray objects.
        Makes sure that for each data set in each of the listXarray object the dimension
        are the same. E.g. if they both have an enesemble dimension, this will make the ensembles
        match. 
        
        Parameters:
            listxr_to_match (listXarray): The list xarray with the desired dim
            dim (str): The dim to be mached
        
        '''
        to_return = []
        for refkey in self.refkeys:
            da1 = self[refkey]
            da2 = listxr_to_match[refkey]
            da1 = da1.where(da1[dim].isin(da2[dim].values), drop=True)
            to_return.append(da1)
        return listXarray(to_return, self.key_dim)
            
    def __repr__(self) -> str:
        """
        Get the string representation of the listXarray.
        Returns:
            str: A string representation of the listXarray.
        """
        string = ''
        string += f'Number of da/ds: {str(len(self.xr_list))}\n---------------\n'
        active_key_dim = f'key_dim = {self.key_dim}' if type(self.key_dim) is not type(None) else 'key_dim = None'
        print(active_key_dim)
        longest_key = np.max(list(map(len, self.refkeys)))
        for refkey, da in zip(self.refkeys, self.xr_list):
            # if isinstance(da, (xr.Dataset, xr.DataArray)):
            #     if 'model' in da.coords:
            #         model_name = str(da.model.values)
            #         string += model_name + (16-len(model_name)) * ' '
            string += refkey + (longest_key-len(refkey)) * ' '
            string += str(da.sizes).replace('Frozen', '')
            string += '\n'
        return string

    def to_dict(self, logginglevel='ERROR') -> Dict:
        """
        Convert the data to a nested dictionary.
    
        :return: A nested dictionary representation of the data.
        """

        utils.change_logginglevel(logginglevel)
        result_dict = {}
    
        for key, ds in self:
            logger.info(key)
            key_dict = {}
            for coord in ds.coords:
                coord_dict = {}
                coord_values = ds[coord].values
                logger.debug(coord_values)
                for coord_val in coord_values:
                    coord_val_data = ds.sel(**{coord: coord_val}).values
                    logger.debug(coord_val_data.ndim)
                    if coord_val_data.ndim == 0:
                        pass
                    elif coord_val_data.ndim < 2:
                        coord_val_data = coord_val_data[0]
                    coord_dict[coord_val] = coord_val_data
    
                # Avoid adding the key as a sub-dictionary
                if coord_val != key:
                    key_dict[coord] = coord_dict
    
            result_dict[key] = key_dict
    
        return result_dict
        
    def to_simple_dict(self) -> Dict:
        """
        Convert the data to a simplified dictionary.

        :return: A simplified dictionary representation of the data.
        """
        result_dict = {}

        # All are shape of 1
        if all([ds.shape == (1,)for key, ds in self]):
            return {key: float(ds.values) for key, ds in self}
            
        for key, ds in self:
            for coord in ds.coords:
                coord_values = ds[coord].values
                for coord_val in coord_values:
                    coord_val_data = ds.sel(**{coord: coord_val}).values
                    if coord_val_data.ndim < 2:
                        coord_val_data = coord_val_data[0]
                    if coord_val != key:
                        result_dict[f'{key}_{coord_val}'] = coord_val_data

        return result_dict

    def to_pandas(self) -> pd.DataFrame:
        '''
        Converts to pandas dataframe is the simple dict configuration can be used
        '''
        return pd.DataFrame(
            {self.key_dim: list(self.to_simple_dict()),
             'values': list(self.to_simple_dict().values())}).set_index(self.key_dim)
    
    def __getitem__(self, key:Union[str, int]):
        """
        Get a DataArray from the listXarray using the specified key.
        Parameters:
            key: The key for the DataArray.
        Returns:
            xr.DataArray: The corresponding DataArray.
        Raises:
            KeyError: If the key is not found in the reference keys.
        """
        # If the key is a string, then this is a reference to the key dim
        if isinstance(key, str):
            arg = np.where(self.refkeys == key)[0][0]
        # If an int, then this is getting the number in they keydim
        elif isinstance(key, int):
            arg = key

        ds = self.xr_list[arg]
        return ds
        
    def __setitem__(self, key, value):
        """
        Set a new DataArray in the listXarray using the specified key.
        Parameters:
            key: The key for the DataArray.
            value (xr.DataArray): The new DataArray.
        Returns:
            listXarray: A new instance of listXarray with the updated DataArray.
        Raises:
            KeyError: If the key is not found in the reference keys.
        """
        arg = np.where(self.refkeys == key)[0][0]
        xr_list = self.xr_list
        xr_list[arg] = value
        return listXarray(xr_list, self.key_dim)
    
    def __len__(self):
        return len(self.xr_list)
    
    def __iter__(self):
        # Create and return an iterator object.
        return self.XarrayIterator(self.xr_list, self.refkeys)

    class XarrayIterator:
        def __init__(self, xr_list, refkeys):
            self.xr_list = xr_list
            self.refkeys = refkeys
            self.index = 0

        def __iter__(self):
            # The iterator object is already its own iterator, so return itself.
            return self

        def __next__(self):
            # Check if there are more elements to iterate through.
            if self.index < len(self.xr_list):
                ds = self.xr_list[self.index]
                refkey = self.refkeys[self.index] if self.refkeys is not None else None
                self.index += 1
                return refkey, ds
            else:
                # Raise StopIteration when there are no more elements.
                raise StopIteration
       
    
    @staticmethod
    def __reconcile(self, other, func):
        """
        Helper method to perform element-wise comparison or operation between elements of two listXarray instances.
        Parameters:
            self (listXarray): The first listXarray instance.
            other (listXarray or float or int): The second listXarray instance or value.
            func (function): The comparison or operation function to be applied element-wise.
        Returns:
            listXarray: A new instance with the result of the operation.
        """
        if isinstance(other, (float, int)):
            new_xr_list = [func(data, other) for data in self.xr_list]
        elif isinstance(other, listXarray):
            new_xr_list = []
            for refkey in self.refkeys:
                output = func(self[refkey], other[refkey])
                new_xr_list.append(output)
        return listXarray(new_xr_list, self.key_dim)

    def __lt__(self, other):
        """
        Perform element-wise less than comparison between two listXarray instances.
        Parameters:
            other (listXarray): Another listXarray instance.
        Returns:
            listXarray: A new instance with boolean values indicating whether each element is less than the corresponding element in 'other'.
        """
        func = np.less
        return self.__reconcile(self, other, func)
    
    def __gt__(self, other):
        """
        Perform element-wise greater than comparison between two listXarray instances.
        Parameters:
            other (listXarray): Another listXarray instance.
        Returns:
            listXarray: A new instance with boolean values indicating whether each element is greater than the corresponding element in 'other'.
        """
        func = np.greater
        return self.__reconcile(self, other, func)
    
    def __ge__(self, other):
        """
        Perform element-wise greater than or equal to comparison between two listXarray instances.
        Parameters:
            other (listXarray): Another listXarray instance.
        Returns:
            listXarray: A new instance with boolean values indicating whether each element is greater than or equal to the corresponding element in 'other'.
        """
        func = np.greater_equal
        return self.__reconcile(self, other, func)
    
    def __le__(self, other):
        """
        Perform element-wise less than or equal to comparison between two listXarray instances.
        Parameters:
            other (listXarray): Another listXarray instance.
        Returns:
            listXarray: A new instance with boolean values indicating whether each element is less than or equal to the corresponding element in 'other'.
        """
        func = np.less_equal
        return self.__reconcile(self, other, func)
    
    
    
    def __add__(self, other):
        """
        Add a given value from all items in the list.
        Parameters:
            value (float or int): The value to be added.
        Returns:
            listXarray: A new instance with the updated values.
        """
        if isinstance(other, (float, int)):
            new_xr_list = [data + other for data in self.xr_list]
        if isinstance(other, listXarray):
            new_xr_list = []
            for refkey in self.refkeys:
                da1 = self[refkey]
                da2 = other[refkey]
                new_xr_list.append(da1+da2)                
        return listXarray(new_xr_list, self.key_dim)
    
    def __sub__(self, other):
        """
        Subtract a given value from all items in the list.
        Parameters:
            value (float or int): The value to be subtracted.
        Returns:
            listXarray: A new instance with the updated values.
        """
        if isinstance(other, (float, int, np.float64, np.float32)):
            new_xr_list = [data - other for data in self.xr_list]
        elif isinstance(other, listXarray):
            new_xr_list = []
            for refkey in self.refkeys:
                da1 = self[refkey]
                da2 = other[refkey]
                new_xr_list.append(da1-da2)    
        else:
            raise TypeError(f'Type is not registerd self {type(self)} other {type(other)}')
        return listXarray(new_xr_list, self.key_dim)
    
    def __mul__(self, other):
        """
        Multiply all values in the xarray list by a certain valuee.
        Parameters:
            value (float or int): The value to be multiplied.
        Returns:
            listXarray: A new instance with the updated values.
        """
        if isinstance(other, (float, int)):
            new_xr_list = [data * other for data in self.xr_list]
        if isinstance(other, listXarray):
            new_xr_list = []
            for refkey in self.refkeys:
                da1 = self[refkey]
                da2 = other[refkey]
                new_xr_list.append(da1 / da2)                
        return listXarray(new_xr_list, self.key_dim)
    
    def __truediv__(self, other):
        """
        Divide all values in the xarray list by a certain valuee.
        Parameters:
            value (float or int): The value to be divided by.
        Returns:
            listXarray: A new instance with the updated values.
        """
        if isinstance(other, (float, int)):
            new_xr_list = [data / other for data in self.xr_list]
        if isinstance(other, listXarray):
            new_xr_list = []
            for refkey in self.refkeys:
                da1 = self[refkey]
                da2 = other[refkey]
                new_xr_list.append(da1 / da2)                
        return listXarray(new_xr_list, self.key_dim)
    
    def apply(self, func: Callable, *args, **kwargs) -> 'listXarray':
        """
        Apply a function to each DataArray in the list and create a new listXarray.

        Parameters:
            func (Callable): The function to apply to each DataArray.
            *args: Additional positional arguments for the function.
            **kwargs: Additional keyword arguments for the function.
        Returns:
            listXarray: A new listXarray with the results of the function applied to each DataArray.
        """
        
        if 'debug' in kwargs:
            debug = kwargs['debug']
            del kwargs['debug']
        else:
            debug=False
            
        to_return = []
        for da in self.xr_list:
            to_append = func(da, *args, **kwargs)
            if debug: print(to_append)
            to_return.append(to_append)
        if debug: return to_return
        return listXarray(to_return, self.key_dim)
        
    
    def reduce(self, func: Callable, *args, **kwargs) -> 'listXarray':
        """
        Reduce each xarray object in the list using the provided function.

        Parameters:
        - func (Callable): The function to apply for reduction.
        - *args: Positional arguments to pass to the function.
        - **kwargs: Keyword arguments to pass to the function.

        Returns:
        - 'listXarray': A new listXarray object containing the reduced xarray objects.
        """
        # Check if 'debug' keyword argument is provided, default to False if not.
        debug = kwargs.pop('debug', False)
        
        # Initialize a list to store the reduced xarray objects.
        to_return = []
        
        # Iterate through each xarray object in the list.
        for da in self.xr_list:
            # Apply the reduce function to the xarray object with provided arguments and keyword arguments.
            to_append = da.reduce(func, *args, **kwargs)
            
            # If debug mode is enabled, print the reduced result.
            if debug:
                print(to_append)
            
            # Append the reduced xarray object to the result list.
            to_return.append(to_append)
        
        # If debug mode is enabled, return the list of reduced xarray objects.
        if debug:
            return to_return
        
        # Otherwise, create a new listXarray object with the reduced xarray objects and return it.
        return listXarray(to_return, self.key_dim)
    
    def to_list(self):
        """
        Convert the listXarray to a list of xarray DataArrays.
        Returns:
            List[xr.DataArray]: The list of xarray DataArrays.
        """
        return self.xr_list
    def merge_dim_to_refkey(self, dim_to_remove: str, logginglevel='ERROR'):
        """
        Merge a specified dimension into the reference keys, creating new reference keys
        based on the removed dimension's values.
    
        Parameters:
            dim_to_remove (str): The name of the dimension to merge into reference keys.
            logginglevel (str): Logging level for messages (default is 'ERROR').
    
        Returns:
            listXarray: A new listXarray object with merged dimensions in reference keys.
        """
        
        # Initialize empty lists to store new Xarray objects and corresponding reference keys
        new_xr_list = []
        new_refkey_list = []
        
        # Iterate over each reference key
        for refkey in self.refkeys:
            # Get the Xarray object for the current reference key
            da = self[refkey]
            
            # Get the values of the dimension to remove from the current Xarray object
            da_remove_dims_vals = da[dim_to_remove].values
        
            # Iterate over each value of the dimension to remove
            for dtr in da_remove_dims_vals:
                # Create a subset of the Xarray object with only the current dimension value
                da_sub = da.loc[{dim_to_remove: dtr}]
                
                # Append the subset to the list of new Xarray objects
                new_xr_list.append(da_sub)
                
                # Create a new reference key by appending the current dimension value to the original reference key
                new_refkey_list.append(f'{refkey}_{dtr}')
        
        # Create a new listXarray object with the merged dimensions in reference keys
        return listXarray(new_xr_list, refkeys=new_refkey_list, key_dim=self.key_dim)
    
    def squeeze(self, *args, **kwargs):
        """
        Squeeze each DataArray in the list.
        Returns:
            listXarray: A new listXarray with squeezed DataArrays.
        """
        to_return = []
        for ds in self.xr_list:
            try: ds = ds.squeeze(*args, **kwargs)
            except KeyError: ds = ds
            to_return.append(ds)
        new_listxr = listXarray(to_return, self.key_dim)
        return new_listxr
    
    def drop(self, to_drop:List[str]):
        """
        Drop specified dimensions from each DataArray in the list.
        Parameters:
            to_drop (List[str]): List of dimensions to be dropped.
        Returns:
            listXarray: A new listXarray with dropped dimensions.
        """
        return listXarray([ds.drop(to_drop, errors='ignore') for ds in self.xr_list], self.key_dim)
    
    def to_dataarray(self, data_var:str):
        """
        Extract a specific data variable from each DataArray in the list.

        Parameters:
            data_var (str): The name of the data variable.
        Returns:
            listXarray: A new listXarray with the extracted data variables.
        """
        return listXarray([ds[data_var] for ds in self.xr_list], self.key_dim)
        
        
    
    def sample_values(self, ivalue: int = 0):
        """
        Print the values of a specific DataArray in the list.

        Parameters:
            ivalue (int): The index of the DataArray to print.
        """
        print(np.take(self.xr_list[ivalue].values, [[0, 1, 2, 3, 4]]))
        print(self.xr_list[ivalue])
        
    def single_xarray(self, ivalue: int = 0):
        """
        Print the values of a specific DataArray in the list.

        Parameters:
            ivalue (int): The index of the DataArray to print.
        """
        return self.xr_list[ivalue]
    
    def resample(self, time:str):
        """
        Resample each DataArray in the list.

        Parameters:
            time (str): The time frequency for resampling.
        Returns:
            listXarray: A new listXarray with resampled DataArrays.
        """
        return listXarray([ds.resample(time=time) for ds in self.xr_list], None)
    
    def _apply_reduction(self, dim, reduction_func):
        """
        Apply a reduction function along specified dimensions for each DataArray in the listXarray.

        Parameters:
            dim (List[str]): List of dimensions along which to apply the reduction function.
            reduction_func (function): The reduction function to apply (e.g., mean, sum).
        Returns:
            listXarray: A new listXarray with the calculated reduction applied to each DataArray.
        """
        reduced_list = []
        for ds in self.xr_list:
            try:
                ds_reduced = reduction_func(ds, dim=dim)
            except ValueError:
                ds_reduced = ds
            reduced_list.append(ds_reduced)
            
        return listXarray(reduced_list, self.key_dim)
    
    def mean(self, dim):
        return self._apply_reduction(dim, reduction_func=lambda ds, dim: ds.mean(dim=dim))
    
    def sum(self, dim):
        return self._apply_reduction(dim, reduction_func=lambda ds, dim: ds.sum(dim=dim))

    def count(self, dim):
        return self._apply_reduction(dim, reduction_func=lambda ds, dim: ds.count(dim=dim))
    
    def dropna(self, dim:Union[List[str], str]):
        """
        Remove missing values along specified dimensions for each DataArray in the list.
        
        Parameters:
            dim (Union[List[str], str]): Dimension or list of dimensions along which to remove missing values.
        Returns:
            listXarray: A new listXarray with missing values removed.
        """
        return listXarray([ds.dropna(dim=dim) for ds in self.xr_list], self.key_dim)
    
    def isel(self, **kwargs):
        """
        Index each DataArray in the list along specified dimensions.

        Parameters:
            **kwargs: Keyword arguments to be passed to the `isel` method of each DataArray.
        Returns:
            listXarray: A new listXarray with the indexed DataArrays.
        """
        new_xr_list = []
        for ds in self.xr_list:
            try: ds = ds.isel(**kwargs)
            except (ValueError, KeyError): ds = ds # Sometime there may be a dim that is not with the other datasets
            new_xr_list.append(ds)
        
        return listXarray(new_xr_list, self.key_dim)
    def compute(self):
        '''
        Loads all Datsets into memory
        Returns:
            listXarray: A new listXarray with computed DataArrays.
        '''
        return listXarray([ds.compute() for ds in self.xr_list], self.key_dim)
    
    def persist(self):
        '''
        Applies persist method to all data arrays/data sets
        Returns:
            listXarray: A new listXarray with persisted DataArrays.
        '''
        return listXarray([ds.persist() for ds in self.xr_list], self.key_dim)


    def copy(self, *args, **kwargs):
        """
        Create a copy of the ListXArray object.

        Parameters:
        - args: Positional arguments to pass to xarray.Dataset.copy method.
        - kwargs: Keyword arguments to pass to xarray.Dataset.copy method.

        Returns:
        - ListXArray: Copy of the ListXArray object.

        
        # Example usage:
        # list_xarray = ListXArray(xr_list=[dataset1, dataset2], key_dim='time')
        # copied_list_xarray = list_xarray.copy
        """
        # Use list comprehension to create a copy of each xarray.Dataset in the list
        copied_datasets = [ds.copy(*args, **kwargs) for ds in self.xr_list]
        
        # Create and return a new ListXArray object with the copied datasets
        return listXarray(copied_datasets, self.key_dim)
    
    
    def chunk(self, chunk_dict):
        """
        Chunk each DataArray in the list.

        Parameters:
            chunk_dict: Dictionary specifying the chunk sizes for each dimension.
        Returns:
            listXarray: A new listXarray with chunked DataArrays.
        """
        return listXarray([ds.chunk(chunk_dict) for ds in self.xr_list], self.key_dim)
    
    
    def unify_chunks(self):
        """
        Unify Chunks for each DataArray in the list.

        Returns:
            listXarray: A new listXarray with chunked DataArrays.
        """
        return listXarray([ds.unify_chunks() for ds in self.xr_list], self.key_dim)
    
    def greater_than(self, other, true_fill=1, false_fill=0):
        """
        Compare each DataArray in the list with the corresponding DataArray of another listXarray using greater-than operation.

        Parameters:
            other (listXarray): The other listXarray for comparison.
            true_fill (int or float): Value to fill where the condition is True.
            false_fill (int or float): Value to fill where the condition is False.
        Returns:
            listXarray: A new listXarray with the comparison results.
        """
        to_return = []
        for refkey in self.refkeys:
            da1 = self[refkey]
            da2 = other[refkey]
            gt_da = xr.where(da1 > da2, true_fill, false_fill)
            to_return.append(gt_da)
        return listXarray(to_return, self.key_dim)
    
    def less_than(self, other, true_fill=1, false_fill=0):
        """
        Compare each DataArray in the list with the corresponding DataArray of another listXarray using less-than operation.

        Parameters:
            other (listXarray): The other listXarray for comparison.
            true_fill (int or float): Value to fill where the condition is True.
            false_fill (int or float): Value to fill where the condition is False.
        Returns:
            listXarray: A new listXarray with the comparison results.
        """
        if isinstance(other, listXarray):
            to_return = []
            for refkey in self.refkeys:
                da1 = self[refkey]
                da2 = other[refkey]
                lt_da = xr.where(da1 < da2, true_fill, false_fill)
                to_return.append(lt_da)
                
            return listXarray(to_return, self.key_dim)
        
        elif isinstance(other, (float, int)):
            return listXarray([xr.where(ds < other, true_fill, false_fill) for ds in self.xr_list], self.key_dim)
        else:
            raise TypeError('Non-valid type for comparison')
            
    def where(self, bool_xrlist, true_fill=1, false_fill=0):
        """
        Apply the 'where' function element-wise to the elements of the listXarray.
        Parameters:
            bool_xrlist (listXarray): A listXarray instance containing boolean values.
            true_fill: Value to fill where 'bool_xrlist' is True.
            false_fill: Value to fill where 'bool_xrlist' is False.
        Returns:
            listXarray: A new instance with elements modified based on the 'where' operation.
        """
        if not isinstance(bool_xrlist, listXarray):
            raise ValueError("'bool_xrlist' must be a listXarray instance.")
        
        if len(self.xr_list) != len(bool_xrlist.xr_list):
            raise ValueError("Both listXarray instances must have the same length for element-wise operations.")
        
        new_xr_list = []
        for ds, bool_ds in zip(self.xr_list, bool_xrlist.xr_list):
            ds_out = ds.where(bool_ds, true_fill, false_fill)
            new_xr_list.append(ds_out)
        return listXarray(new_xr_list, self.key_dim)
    
    
    def __and__(self, other):
        """
        Perform element-wise logical AND between two listXarray instances.
        Parameters:
            other (listXarray): Another listXarray instance or a boolean array.
        Returns:
            listXarray: A new instance with boolean values resulting from the logical AND operation.
        """
        if isinstance(other, listXarray):
            if len(self.xr_list) != len(other.xr_list):
                raise ValueError("Both listXarray instances must have the same length for element-wise operations.")
            new_xr_list = [data and other_data for data, other_data in zip(self.xr_list, other.xr_list)]
        elif isinstance(other, np.ndarray) and other.dtype == bool:
            if len(self.xr_list) != len(other):
                raise ValueError("The boolean array must have the same length as the listXarray for element-wise operations.")
            new_xr_list = [data and other_data for data, other_data in zip(self.xr_list, other)]
        else:
            raise ValueError("The '&' operator is supported between listXarray instances or a boolean array.")

        return listXarray(new_xr_list, self.key_dim)
            
    def regrid(self, target_key: str, method: str) -> List[xr.DataArray]:
        """
        Regrids the DataArrays in the object to a common target grid.

        Parameters:
            target_key (str): The key of the DataArray to which all other DataArrays will be regridded.
            method (str): The regridding method to be used. Should be one of the supported methods by xESMF.

        Returns:
            List[xarray.DataArray]: A list of regridded DataArrays, including the target grid DataArray.
        """
        from xesmf import Regridder

        if target_key not in self.refkeys:
            raise TypeError(f'target_key must be in {self.refkeys} ({target_key=})')

        target_grid = self[target_key]
        regridded_data_arrays = []

        for refkey in self.refkeys:
            if refkey == target_key:
                continue
            else:
                ds = self[refkey]
                regridder = Regridder(ds, target_grid, method=method)
                regridded_ds = regridder(ds)
                regridded_data_arrays.append(regridded_ds)

        # Add the target grid DataArray to the list of regridded DataArrays
        regridded_data_arrays.insert(0, target_grid)

        return listXarray(regridded_data_arrays)
    
    def concat(self, dim:str):
        '''
        Converst the list xarray object into an xarray object. 
        Note: All dims on the object must be the same
        
        '''
        
        return xr.concat(self.xr_list, dim)
    
    def above_or_below(self, main_var: str, greater_than_var: str, less_than_var: str):
        '''
         Bounds on variable by two other. Useful for getting where data is stable
            
         Example
         -------
         unstable_sn_multi_window_da = sn_multiwindow_ds.utils.above_or_below(
                                'signal_to_noise', greater_than_var='upper_bound', less_than_var='lower_bound')
        '''
        
        new_xr_list = []
        for ds in self.xr_list:
            ds_final = ds.utils.above_or_below(
                main_var, greater_than_var=greater_than_var, less_than_var=less_than_var)
            
            new_xr_list.append(ds_final)
            
        return listXarray(new_xr_list, self.key_dim)

    def to_netcdf(self, fname, force=False, logginglevel='ERROR'):
        """
        Save xarray datasets to NetCDF files in a specified directory and
        save the key dimension information in a JSON file within the same directory.

        Args:
        - fname: The directory where the data and key dimension information will be saved.
        """
        utils.change_logginglevel(logginglevel)
        if self.key_dim is None:
            raise ValueError("Key dimension is not specified. Please provide a key dimension before saving the data.")

        try:
            os.mkdir(fname)  # Create the directory if it doesn't exist
        except FileExistsError:
            pass

        # Convert the NumPy array to a regular Python list
        individual_fnames = self.refkeys.tolist()

        name_file_tup_list = list(zip(individual_fnames, self.xr_list))

        # Remove the files that have already been saved
        if force: name_file_tup_list = [tup for tup in name_file_tup_list if f'{tup[0]}.nc' not in os.listdir(fname)]
        logger.debug(list(map(lambda x: x[0], name_file_tup_list)))


        # Save key dimension information in a JSON file within the specified directory
        key_dim_info = {
            'key_dim_name': self.key_dim,
            'datasets': individual_fnames
        }
        json_file_path = os.path.join(fname, 'key_dim_info.json')
        # If we are not forcing and the path does not exists
        # Dont run if: we are forcing and the path exists 
        logger.debug(f'{force=}    path_extist = {os.path.exists(json_file_path)}')
        if not force or not os.path.exists(json_file_path): # If we are forcing, we don't need to create this file again and again
            logger.info('Creating key_dim_info json file')
            if os.path.exists(json_file_path): os.remove(json_file_path)
            
            with open(json_file_path, 'w') as f:
                json.dump(key_dim_info, f)

        for indiv_fname, ds in name_file_tup_list:
            save_name = os.path.join(fname, f'{indiv_fname}.nc')
            logger.info(indiv_fname)
            logger.debug(save_name)
            # Save each xarray dataset to a NetCDF file in the specified directory
            ds.to_netcdf(save_name)


def read_listxarray(fname, *args, **kwargs):
    """
    Open data and key dimension information from a specified directory.

    Args:
    - fname: The directory where the data and key dimension information are stored.

    Returns:
    - An instance of listXarray with loaded data and key dimension information.
    """
    # Load key dimension information from the JSON file
    if 'logginglevel' in kwargs:
        logginglevel=kwargs['logginglevel']
        utils.change_logginglevel(logginglevel)
        kwargs.pop('logginglevel')
    with open(os.path.join(fname, 'key_dim_info.json'), 'r') as f:
        key_dim_info = json.load(f)
        logger.info(key_dim_info)

    # Load individual NetCDF files
    xr_list = []
    for indiv_fname in key_dim_info['datasets']:
        logger.debug(indiv_fname)
        ds = xr.open_dataset(os.path.join(fname, f'{indiv_fname}.nc'), *args, **kwargs)
        xr_list.append(ds)

    # Create an instance of listXarray with loaded data and key dimension
    return listXarray(xr_list, key_dim=key_dim_info['key_dim_name'])

            
def where(xrlist, true_fill=1, false_fill=0):
    """
    Apply the 'where' function element-wise to the elements of a listXarray instance.
    Parameters:
        xrlist (listXarray): A listXarray instance containing data to be modified.
        true_fill: Value to fill where the 'xrlist' is True.
        false_fill: Value to fill where the 'xrlist' is False.
    Returns:
        listXarray: A new instance with elements modified based on the 'where' operation.
    """
    
    new_xr_list = []
    for key, ds in xrlist:
        new_xr_list.append(xr.where(ds, true_fill, false_fill))
    return listXarray(new_xr_list, xrlist.key_dim)

        
    