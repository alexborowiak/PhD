import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec


# This document contains different plot types used in the init folder.

# plot_line_with_annotation:
    # This is used to plot a single line that has an annotation at the end point. 
    # This is very general.
    
# temp_anom_and_signal_plot
    # The temperature anomalies on first y axis and signal/noise on the twiny.
    # This plot needs to be chasnged so that it is just the styling for each subplot
    




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

    
def temp_anom_and_signal_plot(anom_dataset, sn_dataset, ROLL_PERIOD = 61):
    
    fig = plt.figure(figsize = (10,20))
    gs = gridspec.GridSpec(4,1)

    for plot_num,scenario in enumerate(anom_dataset.scenario.values):
    
        ax = fig.add_subplot(gs[plot_num])

        anom_data = anom_dataset.sel(scenario = scenario)
        sn_data = sn_dataset.sel(scenario = scenario)

        anom_data = anom_data.where(np.isfinite(anom_data), drop = True)
        sn_data = sn_data.where(np.isfinite(sn_data), drop = True)


        ax.plot(anom_data.time.values, anom_data.values, label = 'Temperature Anomalies', alpha = 0.5)
        c1 = plt.gca().lines[0].get_color()
        ax.tick_params(axis = 'y', labelcolor = c1)
        ax.set_ylabel('Temperature\nAnomaly', size = 12, color = c1, rotation = 0, labelpad = 45, va = 'center')

        ax.plot([anom_data.time.values[0], anom_data.time.values[0] + pd.to_timedelta(f'{ROLL_PERIOD }Y')],
           [ax.get_yticks()[0] + 0.1,ax.get_yticks()[0] + 0.1], color = 'purple', linestyle = ':', alpha = 0.5)

        ax2 = ax.twinx()

        ax2.plot(sn_data.time.values, sn_data.values, color = 'red', label = 'S/N', alpha = 0.5)
        c2 = plt.gca().lines[0].get_color()
        ax2.tick_params(axis = 'y', labelcolor = c2)
        ax2.set_ylabel('Signal/Noise', size = 12, color = c2, rotation = 0, labelpad = 45, va = 'center')

        ax2.set_ylim([np.min(sn_data.values) - .1, np.max(sn_data.values + .1)])

        ax2.plot([anom_data.time.values[0],anom_data.time.values[-1]], [1,1], color = 'k', linestyle = '--', alpha = .2,
                zorder = -1);


    
    ax.set_title(scenario)
    
    return fig,ax