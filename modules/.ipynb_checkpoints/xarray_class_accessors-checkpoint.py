# +
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
import xarray_extender as xce
import signal_to_noise as sn

import stats

import statsmodels.api as sm 
lowess = sm.nonparametric.lowess


import utils
logger = utils.get_notebook_logger()


from numpy.typing import ArrayLike


@xr.register_dataarray_accessor('clima')
class ClimatologyFunction:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def climatology(self, start = None, end = None,logginglevel='ERROR'):
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
        utils.change_logging_level(logginglevel)
        
        logger.info('Calculating Climatology')
        
        data = self._obj
         
        if start and end:
            logger.debug('Using custom start and end points')
            data = data.where(data.time.dt.year.isin(np.arange(start,end)), drop = True)
            
        climatology = data.mean(dim = 'time')

        return climatology
    
    
    
    def anomalies(self, historical=None,
                  start:int = None, end:int = None,logginglevel='ERROR'):
        utils.change_logging_level(logginglevel)
                    
        logger.info('Calculating anomalies')

        data = self._obj
        
        
        if isinstance(historical, xr.DataArray):
            logger.debug('Using historical dataset')
            climatology = historical.clima.climatology(start = start, end = end)
            
        else:
            logger.debug('Anomallies from self mean')
            climatology = data.clima.climatology(start = start, end = end)
    
        logger.debug('Subtracting the clilamtology from data')
        data_anom = (data - climatology)
    

        return data_anom

    def space_mean(self,logginglevel='ERROR'):
        '''
        When calculating the space mean, the mean needs to be weighted by latitude.

        Parameters
        ----------
        data: xr.Dataset with both lat and lon dimension

        Returns
        -------
        xr.Dataset that has has the weighted space mean applied.

        '''
        utils.change_logging_level(logginglevel)

        logger.info('Calculating the weighted mean for lat and lon. ')

        data = self._obj

        # Lat weights
        weights = np.cos(np.deg2rad(data.lat))
        weights.name = 'weights'

        # Calculating the weighted mean.
        data_wmean = data.weighted(weights).mean(dim = ['lat','lon'])
        
        return data_wmean
    

@xr.register_dataarray_accessor('detrend')
class DetrendMethods:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def trend_fit(self, logginglevel='INFO', *args, **kwargs):
        logger.debug(f'{args=}')
        logger.debug(f'{kwargs}')
        return stats.trend_fit(self._obj, *args, **kwargs)
    
    def detrend_data(self, logginglevel='INFO', *args, **kwargs):
        logger.debug(f'{args=}')
        logger.debug(f'{kwargs}')
        return self._obj - self.trend_fit(self._obj, *args, **kwargs)


# def __mult_func(arr, arr2):
#     return arr * arr2

# def __grid_gradient(arr, axis):
#     xs = np.arange(arr.shape[axis])

#     xs_mult_arr = np.apply_along_axis(__mult_func, axis=axis, arr=arr, arr2=xs)
#     denominator = np.mean(xs) **2 - np.mean(xs**2)

#     t1 = np.nanmean(xs) * np.nanmean(arr, axis=axis)
#     t2 = np.nanmean(xs_mult_arr, axis=axis)
#     numerator = (t1-t2)
#     result = numerator/denominator
#     return result        
 
    
@xr.register_dataarray_accessor('sn')
class SignalToNoise:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    

    @staticmethod
    def __grid_gradient(arr: ArrayLike, axis: int, xs:ArrayLike=None, mean_xs=None, denominator=None):
        def __mult_func(arr:ArrayLike, arr2:ArrayLike):
            return arr * arr2
        
        # xs is need only when it is not provided AND
        # what is calculated from it is not provided (denominator and mean_xs)
        if xs is None: xs = np.arange(arr.shape[axis])
        if denominator is None: denominator = np.mean(xs) **2 - np.mean(xs**2)
        if mean_xs is None: mean_xs = np.nanmean(xs)
        if isinstance(axis, tuple): axis = axis[0]
        xs_mult_arr = np.apply_along_axis(__mult_func, axis=axis, arr=arr, arr2=xs)

        t1 = mean_xs * np.nanmean(arr, axis=axis)
        t2 = np.nanmean(xs_mult_arr, axis=axis)
        numerator = (t1-t2)
        result = numerator/denominator
        return result    
  

    def adjust_time_from_rolling(self, window:int, logginglevel='ERROR'):
        # Get the dataset from the object
        data = self._obj
        return  sn.adjust_time_from_rolling(data, window, logginglevel)

    
    def rolling_signal(self, window:int = 20, min_periods:int = 0, center=True,
                       method:str='gradient', logginglevel='ERROR') -> xr.DataArray:
        '''
        Previosuly signal_grad
        '''
        
        utils.change_logging_level(logginglevel)

        logger.info("Calculting the rolling signal")
        
        data = self._obj
        
        # If no min_periods, then min_periods is just roll_period.
        if ~min_periods:
            min_periods = window

        logger.debug(f'{window=}, {min_periods=}\ndata=\n{data}')

        logger.info(f'Signal method calculation using {method=}')
        if method == 'gradient':
            # Rolling gradient * window
            xs = np.arange(window)
            mean_xs = np.nanmean(xs)
            denominator = np.mean(xs) **2 - np.mean(xs**2)
            mean_xs = np.nanmean(xs)
            signal_da = data.rolling(time=window, min_periods=min_periods, center=center).reduce(
                self.__grid_gradient, xs=xs, mean_xs=mean_xs, denominator=denominator) * window
    
            # The rolling is applied in the start. Thus, the datasets need to be moved forewards or backwards
            # depending on where we want it to start from
            if center == True:
                signal_da = signal_da.sn.adjust_time_from_rolling(window=window, logginglevel=logginglevel)
            else:
                signal_da = signal_da.dropna(dim='time')

        elif method == 'periods':
            signal_da = sn.calculate_rolling_period_diff(data, window)
        else:
            raise TypeError(f'method must be one of [gradient, periods]. value entered {method=}')
    
        signal_da.name = 'signal'
        
        signal_da = signal_da.expand_dims('window').assign_coords(window=('window', [window]))

        return signal_da
    
    
    def calculate_rolling_noise(self, window = 61, min_periods = 0,center=True,logginglevel='ERROR') -> xr.DataArray:
        
        utils.change_logging_level(logginglevel)

        logger.info("Calculting the rolling noise")

        data = self._obj
        
        # If no min_periods, then min_periods is just roll_period.
        if ~min_periods:
            min_periods = window
        
        # Rolling standard deviation
        noise_da = \
           data.rolling(time = window, min_periods = min_periods, center = True).std()

        if center == True:
            noise_da = noise_da.sn.adjust_time_from_rolling(window=window, logginglevel=logginglevel)
        else:
            noise_da = noise_da.dropna(dim='time')
        # noise_da = noise_da.sn.adjust_time_from_rolling(window=window, position=position, logginglevel=logginglevel) 
        
        noise_da.name = 'noise'
        
        noise_da = noise_da.expand_dims('window').assign_coords(window=('window', [window]))
        
        return noise_da
    


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
    
    def anomalies(self, historical_ds: xr.Dataset,logginglevel='ERROR') -> xr.Dataset:
        
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
                
                anoma_da = da.clima.anomalies(historical_da)

                to_merge.append(anoma_da)
            
        return xr.merge(to_merge, compat='override')
    

#     def calculate_rolling_signal(self, window:int = 61, min_periods:int = 0, logginglevel='ERROR') -> xr.DataArray:
#         '''
#         Previosuly signal_grad
#         '''
        
#         utils.change_logging_level(logginglevel)

#         logger.info("Calculting the rolling signal")
        
#         data = self._obj
        
#         # If no min_periods, then min_periods is just roll_period.
#         if ~min_periods:
#             min_periods = window

#         logger.debug(f'{window=}, {min_periods=}\ndata=\n{data}')
#         # Rolling gradient * window
#         signal_da = data.rolling(time = window, min_periods = min_periods, center = True)\
#             .reduce(self._apply_along_helper, func1d = self.trend_line) * window
    
#         signal_da = signal_da.sn.adjust_time_from_rolling(window = window, logginglevel=logginglevel)
    
#         signal_da.name = 'signal'
        
#         signal_da = signal_da.expand_dims('window').assign_coords(window=('window', [window]))

#         return signal_da
    
    
#     @staticmethod
#     def trend_line(x, use = [0][0]):
#         '''
#         Parameters
#         ----------
#         x: the y values of our trend
#         use: 
#         [0][0] will just return the gradient
#         [0,1] will return the gradient and y-intercept.
#         Previosly: _grid_trend
#         '''
#         if all(~np.isfinite(x)):
#             return np.nan

#         t = np.arange(len(x))

#         # Getting the gradient of a linear interpolation
#         idx = np.isfinite(x) #checking where the nans.
#         x = x[idx]
#         t = t[idx]

#         if len(x) < 3:
#             return np.nan

#         poly = np.polyfit(t,x,1)

#         return poly[use]
    
#     @staticmethod
#     def _apply_along_helper(arr, axis, func1d,logginglevel='ERROR'):
#         '''
#         Parameters
#         -------
#         arr : an array
#         axis: the axix to apply the grid_noise function along


#         Example
#         --------
#         >>> ipsl_anom_smean.rolling(time = ROLL_PERIOD, min_periods = MIN_PERIODS, center = True)\
#         >>>    .reduce(apply_along_helper, grid_noise_detrend)
#         '''

#         # If axis is 1D then might become int. Otherwise is array.
#         # TODO: Should this be 0 axis though???
#         axis = axis if isinstance(axis, int) else axis[0]

#         # func1ds, axis, arr 
#         return np.apply_along_axis(func1d, axis, arr)       
    

# @xr.register_dataset_accessor('sn_ds')
# class SignalToNoiseDS:
#     '''All the above accessors are all for data arrays. This will apply the above methods for singal to noise
#     to data sets'''
#     def __init__(self, xarray_obj):
#         self._obj = xarray_obj    
        
#     def sn_multiwindow(self, historical_ds: xr.Dataset, logginglevel='ERROR', **kwargs):
#         '''Loops through all of the data vars in an xarray dataset.'''
        
#         utils.change_logging_level(logginglevel)

#         ds = self._obj
        
#         stable_sn_dict = {}
#         unstable_sn_dict = {}

#         for dvar in list(ds.data_vars):
#             logger.error(f'\n===={dvar}\n')
#             try:
#                 unstable_sn_ds , stable_sn_ds  = sn.sn_multi_window(
#                     ds[dvar].dropna(dim='time'),
#                     historical_ds[dvar].dropna(dim='time'),
#                     logginglevel='ERROR', **kwargs)
               
#                 stable_sn_dict[dvar] = stable_sn_ds['signal_to_noise']
#                 unstable_sn_dict[dvar] = unstable_sn_ds['signal_to_noise']

#             except:
#                 logger.error(f'!!!!!!!!!!!!!!!!!!!!!!\n\n\n{dvar} has error \n {da} \n {da_hist}\n\n\n!!!!!!!!!!!!!!!!!!!!!!')
             
#         stable_sn_ds = xce.xr_dict_to_xr_dataset(stable_sn_dict)
#         unstable_sn_ds = xce.xr_dict_to_xr_dataset(unstable_sn_dict)
                         
#         return stable_sn_ds, unstable_sn_ds


# @xr.register_dataarray_accessor('correct_data')
# class CorrectData:
    
#     def __init__(self, xarray_obj):
#         self._obj = xarray_obj
        
#     def apply_corrections(self, freq='M'):
#         data = self._obj
        
#         if freq == 'M':
#             print('Testing months in each year...')
#             data = self._test_correct_months_in_years(data)
#         return data

#     @staticmethod     
#     def _test_correct_months_in_years(data):
        
#         print(f'Inital time length: {len(data.time.values)}')
#         year, count = np.unique(data.time.dt.year.values, return_counts=True)
        
#         # If the first year has less than 12 months.
#         if count[0] < 12:
#             data = data.where(~data.time.dt.year.isin(year[0]), drop = True)
#             print(f'- First year removed: {count[0]} month(s) in year {year[0]}')
            
#         # If the last year has less than 12 months.
#         if count[-1] < 12:
#             data = data.where(~data.time.dt.year.isin(year[-1]), drop = True)
#             print(f'- Last year removed:  {count[-1]} month(s) in year {year[-1]}')
        
#         # If there is a year that has failed, the whole time needs to be redone.
#         if np.unique(count[1:-1])[0] != 12:
#             fixed_time = xr.cftime_range(start=data.time.values[0],
#                                         periods=len(data.time.values),
#                                         freq='1M')
#             data['time'] = fixed_time
#             print('- Incorrect year detected and time overridden')
       
#         print('\nData Correction complete - all years now have 12 months')
#         print(f'Final time length: {len(data.time.values)}')
            
#         return data
