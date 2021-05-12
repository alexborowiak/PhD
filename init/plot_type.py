import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec



def plot_line_with_annotation(data, time,ax, label = '', deltax = 0, deltay = 0):
    
    fullname_dict = {'piControl':'piControl','historical':'Historical',
                     'abrupt-4xCO2':'Abrupt-4xCO2','1pctCO2':'1%CO2'}
    
    ax.plot(time, data.values, label = fullname_dict[label])
  
    lines = plt.gca().lines
    line_color = lines[len(lines) - 1].get_color()
  
    temp = data.values
    time = data.time.values
    
    x = time[np.isfinite(temp)][-1]
    y = temp[np.isfinite(temp)][-1]
    
    ax.annotate(fullname_dict[label], xy = (x + pd.to_timedelta(f'{deltax}Y'),y + deltay), color = line_color)
