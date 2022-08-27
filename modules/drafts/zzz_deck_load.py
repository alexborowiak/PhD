import numpy as np
import pandas as pd
import xarray as xr
import os




class AnomType:
    def __init__(self, data, info = None):
        self.data = data
        self.info = info
        
    def space_mean(self):
        ### WEIGHTS
        # The weights first need to be calculated.
        data = self.data
        
        weights = np.cos(np.deg2rad(data.lat))
        weights.name= 'weights'
        data_wmean = data.weighted(weights).mean(dim = ['lat','lon'])
        data_wmean.name = data.name

        return data_wmean
    
    def __repr__(self):
        print(f'Weighted Space mean of {self.info}')
        print(f'Time frame: {self.data.time.dt.year.values[0]} - {self.data.time.dt.year.values[-1]}')
        return ''

class Deck:
    def __init__(self, institution, model, ensemble,variable):
        self.data = None
        self.get_full_deck_data(institution, model, ensemble,variable)
        self.description = f'DECK data for {institution} {model} \n**** ENS: {ensemble} \n**** VAR: {variable}'
    
    def get_full_deck_data(self, institution, model, ensemble,variable):
        roots = []
        experiments = ['1pctCO2','abrupt-4xCO2','historical','piControl']
        available_experminents  = os.listdir(f'/g/data/oi10/replicas/CMIP6/CMIP/{institution}/{model}/')
        
        if len(available_experminents) < 4:
            raise ValueError(f'Full deck is not there: {len(available_experminents)}' +\
                             f'\n {available_experminents}')
        
        for experiment in experiments:
            ROOT = self.__get_deck_root(institution, model, experiment, ensemble,variable)
            roots.append(ROOT)
            
        self.data = self.__concat_deck_data(
                self.__get_scenario_data(roots[0]),
                self.__get_scenario_data(roots[1]),
                self.__get_scenario_data(roots[2]),
                self.__get_scenario_data(roots[3]),
                experiments)[variable]


    def __get_deck_root(self,institution, model, experiment, ensemble,variable):
        ROOT = f'/g/data/oi10/replicas/CMIP6/CMIP/{institution}/{model}/{experiment}/{ensemble}/' +\
        f'Amon/{variable}'
    
        try:
            SUB_BIT = '/gn/'
            version = os.listdir(ROOT + SUB_BIT)[0]
        except:
            SUB_BIT = '/gr/'
            version = os.listdir(ROOT + SUB_BIT)[0]
            
        ROOT += SUB_BIT

        ROOT += version

        return ROOT  
    
    def __concat_deck_data(self, d1,d2,d3,d4,experiments):
        return xr.concat([d1,d2,d3,d4], pd.Index(experiments, name = 'scenario'))
        
    
    def __get_scenario_data(self,ROOT):
        # Certain scerious contain either one data set (run to 2100) or two data sets (runs to 2300).
        # If there is only data set we can open this plainly.

        files = os.listdir(ROOT)
        if len(files) > 1:
            df  = xr.open_mfdataset(ROOT + '/*.nc', use_cftime = True, 
                                    chunks = {'lat':100,'lon':100})
        # If there are two, we can employ a mutilfile merge to get them into the one dataset.
        else:
            df = xr.open_dataset(ROOT + '/{}'.format(files[0]), use_cftime = True, 
                                 chunks = {'lat':100,'lon':100})

        return df
    
    def __repr__(self):
        print(self.description)
        print('\n --------------- \n')
        print('Summary of: self.data',self.data.coords, sep = '\n')
        print(f'Var: {self.data.name}')
        
        return ''
        

        
    def climatology(self):
            # Climatology calculation.
        # The method goes as follows 
        hist = self.data.sel(scenario = 'historical')
        
        ### CLIMATOLOGY

        # Getting just the years for climatology. This should be for each pixel, the mean temperature
        # from 1850 to 1900. 
        climatology = hist.where(hist.time.dt.year.isin(np.arange(1850,1901)), drop = True)\
                            .mean(dim = 'time')

        return climatology

    
    def anomalies(self):
        
        data = self.data
        
        climatology = self.climatology()
        
        data_resampled = data.resample(time = 'Y').mean()
        data_anom = (data_resampled - climatology).chunk({'time':8})


        return AnomType(data_anom, self.description)
    
    
