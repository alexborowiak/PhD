import xarray as xr
import numpy as np
import itertools


@xr.register_dataarray_accessor('sn')
class SignalToNoise:
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def consecutive_counter(self, bound):
        '''
        Calculates two array. The first is the start of all the instances of 
        exceeding a threshold. The other is the consecutive length that the 
        threshold.
        TODO: Need to adds in the rolling timeframe. The data is not just unstable
        starting at a specific point, but for the entire time. 

        Parameters
        ----------
        data: np.ndarray
              Groups of booleans.

        Returns
        -------
        consec_start: An array of all start times of consecuitve sequences.
        consec_len: The length of all the exceedneces.

        TODO: Could this be accelerated with numba.njit???? The arrays will 
        always be of unkonw length.
        '''
        condition = self._obj.values >= bound
        print('Alex')
        #condition = data >= stable_bound

        consec_start_arg = []
        consec_len = []

        # Arg will keep track of looping through the list.
        arg = 0

        # This loop will grup the array of Boleans together.  Key is the first value in the
        # group and group will be the list of similar values.
        for key, group in itertools.groupby(condition):

            # Consec needs to be defined here for the arg
            consec = len(list(group))

            if key:
                consec_start_arg.append(arg)
                consec_len.append(consec)

            arg += consec

        return np.array(consec_start_arg), np.array(consec_len)

    

