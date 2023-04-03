from enum import Enum


class detrendingMethods(str, Enum):
    POLYNOMIAL = 'polynomial'
    LOWESS = 'lowess'

class ExperimentTypes(str, Enum):
    CONTROL = 'control'
    ABRUPT4X = 'abrupt4x'
    SIGNAL_TO_NOISE = 'signal_to_noise'
    SIGNAL_TO_NOISE_STABLE = 'signal_to_noise_stable'
    SIGNAL_TO_NOISE_DECREASING = 'signal_to_noise_decreasing'
    SIGNAL_TO_NOISE_INCREASING = 'signal_to_nosie_increasing'
    
    
class LongRunMIPError(Exception):
    pass


class LocationsLatLon(Enum):
    '''Latittude and longitude of different locations around the world'''
    EPACIFIC = (2.7 ,-85)
    NATLANTIC = (2.2, -8)
    ECHINA_SEA = (26, 128)
    MELBOURNE = (-38, 145)
    LONDON = (51, 0.12)
    NEW_YORK = (40.7, 74)
    BEIJING = (40, 116) 
    BUENOS_AIRES = (35, 58)
    KINSHASA = (4.4, 15.2)
    ARCTIC = (76, 100)
    JAKARTA = (-6.2088, 106.84)
    CENTRAL_PACIFIC = (0, -140)
    SOUTHERN_OCEAN = (-45, 60)