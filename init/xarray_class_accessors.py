import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
lowess = sm.nonparametric.lowess


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

#         data_resampled = data.resample(time = 'Y').mean()
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
    def loess_filter(y: np.array, step_size = 10):

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

        #yhat is the loess version of y - this is the final product.
        yhat = lowess(y, x, frac  = frac)

        return yhat[:,1]




    def loess_grid(self,
                      step_size = 60, 
                      min_periods = 0) -> xr.DataArray:
        
        '''Applies the loess filter static method to an array through time'''
        
        data = self._obj
       
        # Loess filter
        loess = np.apply_along_axis(self.loess_filter, data.get_axis_num('time'), data.values, step_size = step_size)

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
        condition = data
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
    
    
    
#     def consec_counter(self, threshold, condition = ):
        
        
    

