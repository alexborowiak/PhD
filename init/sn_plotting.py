import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import ticker as mticker

def temperature_vs_sn_plot(ax, sn, temp, temp_highlight=None, sn_highlight=None):
    '''
    Plot the temperature and signal_to_noise (could also be just
    signal or just noise)
    
    Parameters
    ----------
    All datasets in this plot are xr.DataArrays.
    *_highlight are optiononal parameters that can be added to highligh certain
    sections of the plot
    
    '''    
    ax.plot(sn.time.values,sn.values, label = 'S/N', c = 'tomato')
    if isinstance(sn_highlight, xr.DataArray):
        ax.plot(sn_highlight.time.values,sn_highlight.values, label = 'S/N', c = 'darkred')


    c0 = plt.gca().lines[0].get_color()
    ax.tick_params(axis = 'y', labelcolor = c0)
    ax.set_ylabel('Signal\n----------\nNoise', fontsize = 12, color = c0, rotation = 0, labelpad = 55);

    leg = ax.legend(ncol = 1, fontsize = 12, bbox_to_anchor = [-0.05, 1])
    leg.set_title('Noise Type')
    leg.get_title().set_fontsize('12')

    ax.yaxis.set_major_locator(mticker.MultipleLocator(.5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(.25))

    # Second y-axis: the temperature anomalies.
    ax2 = ax.twinx()

    ax2.plot(temp.time.values, temp.values,
             color = 'green', alpha = 0.4, label  = 'Unstable')
    
    if isinstance(temp_highlight, xr.DataArray):
        ax2.plot(temp_highlight.time.values, temp_highlight.values,
                 color = 'darkgreen', alpha = 0.8, label  = 'Stable')

    c1 = plt.gca().lines[0].get_color()
    ax2.tick_params(axis = 'y', labelcolor = c1)
    ax2.spines['right'].set_color(c1)
    ax2.spines['left'].set_color(c0)
    ax2.set_ylabel('Tempearture\nAnomaly', fontsize = 12, color = c1, rotation = 0, labelpad = 55);
    
    leg2 = ax2.legend(ncol = 1, fontsize = 12, bbox_to_anchor = [1.17, 1])
    leg2.set_title('Temperature\nAnomaly')
    leg2.get_title().set_fontsize('12')
    
    
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(.5))
    ax2.yaxis.set_minor_locator(mticker.MultipleLocator(.25))

    
    ax.set_xlabel('Year', fontsize = 12);
    ax.set_xlim(temp.time.values[0], temp.time.values[-1])
    ax2.set_xlim(temp.time.values[0], temp.time.values[-1])