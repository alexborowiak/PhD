import pytest
import numpy as np
import sys, os
import cftime
import xarray as xr

sys.path.append('/home/563/ab2313/Documents/PhD')
import constants
sys.path.append(constants.MODULE_DIR)
import xarray_class_accessors as xca
import signal_to_noise as sn
import stats

ACCEPTED_NOISE_PERCENTAGE_ERROR = 10
ACCEPTED_SN_PERCENTAGE_ERROR = 10

def percent_diff(da, val: float):
    return np.abs((da-val)*100/da)

@pytest.fixture
def generate_mock_dataset():
    def _generate_mock_dataset(input_mock_values) -> xr.DataArray:
        t0 = cftime.datetime(1, 1, 1, 0, 0, 0, 0, calendar='gregorian')

        lat = np.linspace(-25, -10, 10)
        lon = np.linspace(110, 135, 20)

        time = xr.cftime_range(start=t0, periods=len(input_mock_values), freq='1Y')

        mock_values = np.array([np.array([input_mock_values for _ in lat]) for _ in lon])

        mock_da = xr.Dataset({'tas':(('lon', 'lat', 'time'), mock_values)},
                             {'lat':lat, 'lon':lon,'time':time}).to_array(name='tas').squeeze()

        return mock_da
    return _generate_mock_dataset

@pytest.mark.usefixtures('generate_mock_dataset')
@pytest.mark.parametrize('gradient, window',
                         [(1, 31), (2, 101), (3, 101), (50, 61), (-10, 11), (0, 21)])
def test_rolling_signal_calculation(generate_mock_dataset, gradient, window):
    
    number_points = 500

    input_mock_values = gradient * np.arange(0, number_points)
    mock_da = generate_mock_dataset(input_mock_values)
    mock_sn = mock_da.sn.rolling_signal(window=window)
    
    print(mock_sn.isel(lat=0, lon=0).values)
    assert np.all(mock_sn.values.round(0) == float(window) * gradient)
    assert np.all(mock_sn.window.values == float(window))
    assert len(mock_sn.time.values) == (number_points-window+1)
    
    
    
@pytest.mark.usefixtures('generate_mock_dataset')
@pytest.mark.parametrize('window, std, mean', [(101, 0.5, 10), (31, 1, 0), (11, 1.7, 2)])
def test_rolling_noise_calculation(generate_mock_dataset, window, std, mean):
    number_points=500
    input_mock_values = np.random.normal(mean, std, number_points)
    actual_std = np.std(input_mock_values)
    mock_da = generate_mock_dataset(input_mock_values)
    print(mock_da)
    mock_noise = mock_da.sn.calculate_rolling_noise(window=window)
    
    print(mock_noise)
    mean_diff = np.nanmean(percent_diff(mock_noise.mean(dim='time').values, actual_std))
    print(f'{mean_diff=}')
    assert  mean_diff< ACCEPTED_NOISE_PERCENTAGE_ERROR
    assert len(mock_noise.time.values) == (number_points-window+1)
    
    

@pytest.mark.usefixtures('generate_mock_dataset')
@pytest.mark.parametrize('gradient, order',
                         [(1, 1), (2, 1), (3, 3), (1, 3), (-10, 2)])
def test_detrending(generate_mock_dataset, gradient, order):
    number_points = 100
    input_mock_values = gradient * np.arange(0, number_points) ** order
    mock_da = generate_mock_dataset(input_mock_values)
    mock_trend = stats.trend_fit(mock_da, method='polynomial', order=order)
    detrended = mock_da - mock_trend
    assert np.all(detrended.values.round(0) == 0)
    assert len(detrended.time.values) == number_points


    
@pytest.mark.usefixtures('generate_mock_dataset')
@pytest.mark.parametrize('window, gradient, std1, std2',
                         [(31, 1, .1, 1), (101, 3, .5, .2)])
def test_rolling_signal_to_noise_calculation(generate_mock_dataset, window, gradient, std1, std2):
    mean = 0
    number_points = 500
    
    input_mock_values_2 = gradient * np.arange(0, number_points, dtype=np.float64)

    mid_point = int(number_points/2)
    
    random1 = np.random.normal(mean, std1, len(input_mock_values_2[:mid_point]))
    random2 = np.random.normal(mean, std2, len(input_mock_values_2[mid_point:]))

    input_mock_values_2[:mid_point] = input_mock_values_2[:mid_point] + random1
    input_mock_values_2[mid_point:] = input_mock_values_2[mid_point:] + random2

    mock_da_2 = generate_mock_dataset(input_mock_values_2)
    
    mock_sn = sn.signal_to_noise(da=mock_da_2, window=window, detrend_kwargs={'method':'polynomial', 'order':1}, 
                            return_all=True)
    
    expcted_signal = window * gradient
    expected_noise1 = np.std(random1)
    expected_noise2 = np.std(random2)
    expected_signal_to_noise1 = expcted_signal/expected_noise1
    expected_signal_to_noise2 = expcted_signal/expected_noise2
    
    assert mock_sn.signal.mean().round(0).values == expcted_signal
    
    first_half = slice(0, mid_point-window)
    second_half = slice(mid_point+window, -window-10)
    
    assert percent_diff(
        mock_sn.isel(time=second_half).noise.mean().values ,expected_noise2) < ACCEPTED_SN_PERCENTAGE_ERROR
    assert percent_diff(
        mock_sn.isel(time=first_half).noise.mean().values ,expected_noise1) < ACCEPTED_SN_PERCENTAGE_ERROR
    assert percent_diff(
        mock_sn.signal_to_noise.isel(time=first_half).mean().values, expected_signal_to_noise1) < ACCEPTED_SN_PERCENTAGE_ERROR
    assert percent_diff(
        mock_sn.signal_to_noise.isel(time=second_half).mean().values, expected_signal_to_noise2) < ACCEPTED_SN_PERCENTAGE_ERROR
    assert len(mock_sn.signal_to_noise.dropna(dim='time').time.values) == (number_points-window+1)
    assert len(mock_sn.signal.dropna(dim='time').time.values) == (number_points-window+1)
    