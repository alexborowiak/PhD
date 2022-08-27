# +
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
import xarray_extender as xce
import signal_to_noise as sn
import signal_to_noise as sn

import statsmodels.api as sm 
lowess = sm.nonparametric.lowess
# -

import logging, sys
logging.basicConfig(format="%(message)s", filemode='w', stream=sys.stdout)
logger = logging.getLogger()


@xr.register_dataarray_accessor('correct_data')
class CorrectData:
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def apply_corrections(self, freq='M'):
        data = self._obj
        
        if freq == 'M':
            print('Testing months in each year...')
            data = self._test_correct_months_in_years(data)
        return data

    @staticmethod     
    def _test_correct_months_in_years(data):
        
        print(f'Inital time length: {len(data.time.values)}')
        year, count = np.unique(data.time.dt.year.values, return_counts=True)
        
        # If the first year has less than 12 months.
        if count[0] < 12:
            data = data.where(~data.time.dt.year.isin(year[0]), drop = True)
            print(f'- First year removed: {count[0]} month(s) in year {year[0]}')
            
        # If the last year has less than 12 months.
        if count[-1] < 12:
            data = data.where(~data.time.dt.year.isin(year[-1]), drop = True)
            print(f'- Last year removed:  {count[-1]} month(s) in year {year[-1]}')
        
        # If there is a year that has failed, the whole time needs to be redone.
        if np.unique(count[1:-1])[0] != 12:
            fixed_time = xr.cftime_range(start=data.time.values[0],
                                        periods=len(data.time.values),
                                        freq='1M')
            data['time'] = fixed_time
            print('- Incorrect year detected and time overridden')
       
        print('\nData Correction complete - all years now have 12 months')
        print(f'Final time length: {len(data.time.values)}')
            
        return data




@xr.register_dataarray_accessor('clima')
class ClimatologyFunction:
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def climatology(self, start = 1850, end = 1901):
        '''
        CLIMATOLOGY
        Getting just the years for climatology. This should be for each pixel, the mean temperature
        from 1850 to 1900.

        Parameters
        ----------
        hist: xarray dataset with dimension time
        start: float/int of the start year.
        end: float/ind of the end year

        Returns:
        climatologyL xarray dataset with the mean of the time dimension for just the years from 
        start to end. Still contains all other dimensions (e.g. lat and lon) if passed in.

        '''
        data = self._obj
        climatology = data.where(data.time.dt.year.isin(np.arange(start,end)), drop = True)\
                            .mean(dim = 'time')

        return climatology
    
    
    
    # TODO: Need kwargs for this.
    def anomalies(self, start = 1850, end = 1901, **kwargs): # hist

        data = self._obj
        
        
        if 'historical' in kwargs:
            print('Using historical dataset')
            climatology = kwargs['historical'].clima.climatology(start = start, end = end)
            
        else:
            climatology = data.clima.climatology(start = start, end = end)

        data_anom = (data - climatology).chunk({'time':8})
    
        if 'debug' in kwargs:
            return data_anom, climatology

        return data_anom

    def space_mean(self):
        '''
        When calculating the space mean, the mean needs to be weighted by latitude.

        Parameters
        ----------
        data: xr.Dataset with both lat and lon dimension

        Returns
        -------
        xr.Dataset that has has the weighted space mean applied.

        '''

        data = self._obj

        # Lat weights
        weights = np.cos(np.deg2rad(data.lat))
        weights.name = 'weights'

        # Calculating the weighted mean.
        data_wmean = data.weighted(weights).mean(dim = ['lat','lon'])
        
        return data_wmean

    




@xr.register_dataarray_accessor('sn')
class SignalToNoise:
    
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    @staticmethod    
    def loess_filter(y: np.array, step_size = 60):
        '''
        Applies the loess filter to a 1D numpy array.

        Parameters
        -----------
        data: the 1D array of values to apply the loess filter to
        step_size: the number of steps in each of the loess filter. The default is 50 points 
        in each window.

        Returns
        -------
        yhat: the data but, the loess version.

        Example
        -------
        >>> mean_temp = data.temp.values
        >>> mean_temp_loess = loess_filter(mean_temp)
        >>> 
        >>> # The mean temperature that has been detrended using the loess method.
        >>> mean_temp_loess_detrend = mean_temp - mean_temp_loess

        '''

        # Removign the nans (this is important as if two dataarrays where together in dataset
        # one might have been longer than the other, leaving a trail of NaNs at the end.)
        idy = np.isfinite(y)
        y = y[idy]

        # The equally spaced x-values.
        x =  np.arange(len(y))


        # The fraction to consider the linear trend of each time.
        frac = step_size/len(y)
        # The fraction is just the whole length of y
        # frac = 1#len(y)
    

        #yhat is the loess version of y - this is the final product.
        yhat = lowess(y, x, frac  = frac)

        return yhat[:,1]

    def loess_grid(self,
                      step_size = 60, 
                      min_periods = 0) -> xr.DataArray:
        
        '''Applies the loess filter static method to an array through time'''
        
        data = self._obj
        
        # (25th March) The loess filter will do this automatically and will cause dim 
        # mismatch if not added here. 
        data = data.dropna(dim='time')
       
        # Loess filter
        loess = np.apply_along_axis(self.loess_filter, data.get_axis_num('time'), data.values,
                                    step_size = step_size)

        # Detredning with the loess filer.
        loess_detrend = data - loess

        return loess_detrend
    
    
    
    @staticmethod
    def _grid_trend(x, use = [0][0]):
        '''
        Parameters
        ----------
        x: the y values of our trend
        use: 
        [0][0] will just return the gradient
        [0,1] will return the gradient and y-intercept.
        '''
        if all(~np.isfinite(x)):
            return np.nan

        # If every point is just a nan values. We don't want to do the polyfit calculation. Just return nan
        t = np.arange(len(x))

        # Getting the gradient of a linear interpolation
        idx = np.isfinite(x) #checking where the nans.
        x = x[idx]
        t = t[idx]

        if len(x) < 3:
            return np.nan

        poly = np.polyfit(t,x,1)

        return poly[use]
    
    @staticmethod
    def _apply_along_helper(arr, axis, func1d):
        '''
        Parameters
        -------
        arr : an array
        axis: the axix to apply the grid_noise function along


        Example
        --------
        >>> ipsl_anom_smean.rolling(time = ROLL_PERIOD, min_periods = MIN_PERIODS, center = True)\
        >>>    .reduce(apply_along_helper, grid_noise_detrend)
        '''

        # func1ds, axis, arr 
        return np.apply_along_axis(func1d, axis[0], arr)
    
    
    def adjust_time_from_rolling(self, roll_period):
        
        data = self._obj

        # This will get rid of all the NaN points on either side that arrises due to min_periods.
        data_adjusted = data.isel(time = slice(
                                   int((roll_period - 1)/2),
                                    -int((roll_period - 1)/2)
                                  )
                    )

        # We want the time to match what the data is (will be shifter otherwise).
        data_adjusted['time'] = data.time.values[:len(data_adjusted.time.values)]

        return data_adjusted
    
    
    def signal_grad(self,
                      roll_period = 61, 
                      min_periods = 0) -> xr.DataArray:

        '''
        '''
        data = self._obj
        
        # If no min_periods, then min_periods is just roll_period.
        if ~min_periods:
            min_periods = roll_period


        # Getting the graident at each point with the rolling function. Then multipolying 
        # by the number of points to get the signal.
        signal = data.rolling(time = roll_period, min_periods = min_periods, center = True)\
            .reduce(self._apply_along_helper, func1d = self._grid_trend) * roll_period
    
        signal = signal.sn.adjust_time_from_rolling(roll_period = roll_period)
    
        signal.name = 'signal'
    
        return signal

    
    def noise_grad(self,
                      roll_period = 61, 
                      min_periods = 0) -> xr.DataArray:
        
        data = self._obj
        
        # If no min_periods, then min_periods is just roll_period.
        if ~min_periods:
            min_periods = roll_period
        
        noise = \
           data.rolling(time = roll_period, min_periods = min_periods, center = True).std()
            
        noise = noise.sn.adjust_time_from_rolling(roll_period = roll_period) 
        
        noise.name = 'noise'
        
        return noise
    
    
    
    @staticmethod
    def _consecutive_counter(data: np.array) -> np.array:
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

        # Data should have nans where conditionsis not met.
        if all(np.isnan(data)):
            logger.debug('All are nan values - returning nan values')
            return np.array([np.nan] * 5)
            
        condition = np.where(np.isfinite(data), True, False)
        #condition = data >= stable_bound

        consec_start_arg = []
        consec_len = []

        # Arg will keep track of looping through the list.
        arg = 0

        # This loop will grup the array of Boleans together.  Key is the first value in the
        # group and group will be the list of similar values.
        #[True, True, True, False, True, False, True, True] will have the groups that will be accepted 
        # by 'if key':
        #[[True, True, True], [True], [True, True]]
#         print(data)
        for key, group in itertools.groupby(condition):
            # Consec needs to be defined here for the arg
            consec = len(list(group))
            
            # If they key is true, this means
            if key:
                consec_start_arg.append(arg)
                consec_len.append(consec)

            arg += consec

        # First time stable
        first_stable = consec_start_arg[0]

        # Average lenght of period
        average_consec_length = np.mean(consec_len)

        # Total number periods
        number_consec = len(consec_len)

        # Sum of all period
        total_consec = np.sum(consec_len)

        # Fraction of total where condition is met
        frac_total = total_consec * 100 / len(data)
        
        return np.array([first_stable, average_consec_length,number_consec, total_consec, frac_total])
        
    def calculate_consecutive_metrics(self, logginglevel='INFO'):
    
        eval(f'logging.getLogger().setLevel(logging.{logginglevel})')
        
        data = self._obj
        
        # Applying the consecitve_counter function along the time axis.
        output = np.apply_along_axis(
                            self._consecutive_counter,
                            data.get_axis_num('time'), 
                            data)

        print(f'New data has shape {output.shape}')
        # Creating an exmpty dataset with no time dimension
        ds = xr.zeros_like(data.isel(time=0).squeeze())
        
        # Adding in the first dimension to the dataset
        ds.name = 'first_stable'
        
        # Output is an array of arrays. Adding the first elemtn
        ds += output[0]
        
        # Converting to dataset so other data vars can be added.
        ds = ds.to_dataset()
        
        
        # The dims data should have
        dims = np.array(data.dims) 
        
        # Data has been reduced along time dimension. Don't want time.
        dims_sans_time = dims[dims != 'time']
        
        # Adding all other variables
        ds['average_length'] = (dims_sans_time, output[1])
        ds['number_periods'] = (dims_sans_time, output[2])
        ds['total_time_stable'] = (dims_sans_time, output[3])
        ds['percent_time_stable'] = (dims_sans_time, output[4])
        
        
        # Adding long names
        ds.first_stable.attrs = {"long_name": "** First Year Stable", 'units':'year'}
        ds.average_length.attrs = {"long_name": "** Average Length of Stable Periods", 'units':'year'}
        ds.number_periods.attrs = {"long_name": "** Number of Different Stable Periods", 'units':''}
        ds.percent_time_stable.attrs = {"long_name": "** Percent of Time Stable", 'units':'%'}

        return ds.squeeze()


@xr.register_dataset_accessor('clima_ds')
class ClimatologyFunctionDataSet:
    '''All the above accessors are all for data arrays. This will apply the above methods
    to data sets'''
    def __init__(self, xarray_obj):
        self._obj = xarray_obj    
        
    def space_mean(self):
        data = self._obj
        data_vars = list(data.data_vars)
    
        return xr.merge([data[dvar].clima.space_mean() for dvar in data_vars])
    
    def anomalies(self, historical_ds: xr.Dataset) -> xr.Dataset:
        
        ds = self._obj
        
        # The data vars in each of the datasets
        data_vars = list(ds.data_vars)
        hist_vars = list(historical_ds.data_vars)
        
        # Looping through all data_vars and calculating the anomlies
        to_merge = []
        for dvar in data_vars:
            print(f'{dvar}, ', end='')
            # Var not found in historical.
            if dvar not in hist_vars:
                print(f'{dvar} is not in historiocal dataset - anomalies cannot be calculated')
            else:
                # Extracing the single model.
                da = ds[dvar]
                historical_da = historical_ds[dvar]
                
                # Calculating anomalies
                start = historical_da.time.dt.year.values[0]
                end = historical_da.time.dt.year.values[-1]
                anoma_da = da.clima.anomalies(start = start, end = end, historical = historical_da)

                to_merge.append(anoma_da)
            
        return xr.merge(to_merge, compat='override')
    
    
    def sn_multiwindow(self, historical_ds: xr.Dataset, start_window = 20, end_window = 40, step_window = 5
                      ,logginglevel='ERROR'):
        
        
        '''Loops through all of the data vars in an xarray dataset.'''
        
        
        ds = self._obj
        
        # The data vars in each of the datasets
        data_vars = list(ds.data_vars)
        hist_vars = list(historical_ds.data_vars)

        stable_sn_dict = {}
        unstable_sn_dict = {}

        for dvar in data_vars:
            print(f'\n{dvar}')

            # Making sure it doesn't fail
            try:
                da = ds[dvar].dropna(dim='time')
                da_hist = historical_ds[dvar].dropna(dim='time')
                unstable_sn_da , stable_sn_da  = sn.sn_multi_window(
                                        da, 
                                        da_hist, 
                                        start_window = start_window,
                                        end_window=end_window, step_window=step_window,
                                        logginglevel=logginglevel)

                # Storing the values as a dictionary and not concating for now.
                stable_sn_dict[dvar] = stable_sn_da
                unstable_sn_dict[dvar] = unstable_sn_da

            # If there is a value error, document this and move on.
            except:
                print(f'!!!!!!!!!!!!!!!!!!!!!!\n\n\n{dvar} has error \n {da} \n {da_hist}\n\n\n!!!!!!!!!!!!!!!!!!!!!!')
                    
        stable_sn_ds = xce.xr_dict_to_xr_dataset(stable_sn_dict)
        unstable_sn_ds = xce.xr_dict_to_xr_dataset(unstable_sn_dict)
                         
        return stable_sn_ds, unstable_sn_ds



# +
# @xr.register_dataarray_accessor('clima')
# class ClimatologyFunction:
    
#     def __init__(self, xarray_obj):
#         self._obj = xarray_obj
        
#     def climatology(self, start = 1850, end = 1901):
#         '''
#         CLIMATOLOGY
#         Getting just the years for climatology. This should be for each pixel, the mean temperature
#         from 1850 to 1900.

#         Parameters
#         ----------
#         hist: xarray dataset with dimension time
#         start: float/int of the start year.
#         end: float/ind of the end year

#         Returns:
#         climatologyL xarray dataset with the mean of the time dimension for just the years from 
#         start to end. Still contains all other dimensions (e.g. lat and lon) if passed in.

#         '''
#         data = self._obj
#         climatology = data.where(data.time.dt.year.isin(np.arange(start,end)), drop = True)\
#                             .mean(dim = 'time')

#         return climatology
    
    
    
#     # TODO: Need kwargs for this.
#     def anomalies(self, start = 1850, end = 1901, **kwargs): # hist

#         data = self._obj
        
        
#         if 'historical' in kwargs:
#             print('Using historical dataset')
#             climatology = kwargs['historical'].clima.climatology(start = start, end = end)
            
#         else:
#             climatology = data.clima.climatology(start = start, end = end)

#         data_anom = (data - climatology).chunk({'time':8})
    
#         if 'debug' in kwargs:
#             return data_anom, climatology

#         return data_anom

#     def space_mean(self):
#         '''
#         When calculating the space mean, the mean needs to be weighted by latitude.

#         Parameters
#         ----------
#         data: xr.Dataset with both lat and lon dimension

#         Returns
#         -------
#         xr.Dataset that has has the weighted space mean applied.

#         '''

#         data = self._obj

#         # Lat weights
#         weights = np.cos(np.deg2rad(data.lat))
#         weights.name = 'weights'

#         # Calculating the weighted mean.
#         data_wmean = data.weighted(weights).mean(dim = ['lat','lon'])
        
#         return data_wmean
