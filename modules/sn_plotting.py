import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from typing import Union
from matplotlib import ticker as mticker
import matplotlib.gridspec as gridspec
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from typing import Dict
import exceptions

import sys,logging
logging.basicConfig(format="%(message)s", filemode='w', stream = sys.stdout)
logger = logging.getLogger()


def temperature_vs_sn_plot(ax,
                           sn:xr.DataArray=None,
                           temp:xr.DataArray=None,
                           temp_highlight:xr.DataArray=None,
                           sn_highlight:xr.DataArray=None,
                          bounds:Dict[str, float] = None):
    '''
    Plot the temperature and signal_to_noise (could also be just
    signal or just noise)
    
    Parameters
    ----------
    ax: matplotlib axis
    sn, temp: temp_highlight=None, sn_highlight=None: xr.DataArray
    
    Returns
    --------
    
    [ax, ax2]: matplotlib axis
    
    All datasets in this plot are xr.DataArrays.
    *_highlight are optiononal parameters that can be added to highligh certain
    sections of the plot
    
    '''    
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    ax.plot(sn.time.values,sn.values, label = 'Unstable', c = 'tomato', linestyle='--')
    if isinstance(sn_highlight, xr.DataArray):
        ax.plot(sn_highlight.time.values,sn_highlight.values, label = 'Stable', c = 'darkred')


    c0 = plt.gca().lines[0].get_color()
    ax.tick_params(axis = 'y', labelcolor = c0)
    ax.set_ylabel('Signal to Noise', fontsize = 16, color = c0, rotation = 0, labelpad = 55);
    
    
        
    if isinstance(bounds, dict):
        for key, value in bounds.items():
            ax.plot([temp.time.values[0], temp.time.values[-1]], [value, value], 
                   color=c0, linestyle=':', alpha=0.8)
    
    leg = ax.legend(ncol = 1, fontsize = 15, bbox_to_anchor = [-0.03, 1])
    leg.set_title('S/N')
    leg.get_title().set_fontsize('15')

    ax.yaxis.set_major_locator(mticker.MultipleLocator(.5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(.25))

    # Second y-axis: the temperature anomalies.
    ax2 = ax.twinx()

    ax2.plot(temp.time.values, temp.values,
             color = 'green', alpha = 0.4, label  = 'Unstable', linestyle='--')
    
    if isinstance(temp_highlight, xr.DataArray):
        ax2.plot(temp_highlight.time.values, temp_highlight.values,
                 color = 'darkgreen', alpha = 0.8, label  = 'Stable')

    c1 = plt.gca().lines[0].get_color()

    ax2.spines['right'].set_color(c1)
    ax2.spines['left'].set_color(c0)
    ax2.set_ylabel('Tempearture\nAnomaly'+ r' ($^{\circ}$C)', fontsize = 16,
                   color = c1, rotation = 0, labelpad = 55);
    
    leg2 = ax2.legend(ncol = 1, fontsize = 15, bbox_to_anchor = [-0.03, 0.8]) # 1.19 for rhs
    leg2.set_title('Temperature\nAnomaly')
    leg2.get_title().set_fontsize('15')
    
    
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(.5))
    ax2.yaxis.set_minor_locator(mticker.MultipleLocator(.25))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax2.tick_params(axis = 'y', labelsize=14, labelcolor = c1)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    
    
    ax.set_xlabel('Year', fontsize = 16);
    ax.set_xlim(temp.time.values[0], temp.time.values[-1])
    ax2.set_xlim(temp.time.values[0], temp.time.values[-1])

    
    return [ax, ax2]


def sn_plot_kwargs(kwargs):
    # The default plot kwargs
    plot_kwargs = dict(height = 15, width = 7, hspace=0.3, vmin = -8, vmax = 8, step = 2, 
                      cmap = 'RdBu_r', line_color = 'limegreen', line_alpha = 0.5, 
                      cbar_label = 'S/N',cbartick_offset = 0,title='', label_size = 12, extend='both', 
                      xlowerlim = None, xupperlim = None)
    
    
    # Merging the dicionaries
    plot_kwargs = {**plot_kwargs, **kwargs}
    
    
    if 'filter_max' not in plot_kwargs.keys():
        plot_kwargs['filter_max'] = False

    
    plot_kwargs['levels'] = np.arange(plot_kwargs['vmin'], # Min
                       plot_kwargs['vmax'] + plot_kwargs['step'], # Max
                       plot_kwargs['step']) # Step
    
    plot_kwargs['cmap'] = plt.cm.get_cmap(plot_kwargs['cmap'], len(plot_kwargs['levels']) + 1)

        
    if 'cbar_xticklabels' in kwargs.keys():
        plot_kwargs['cbar_xticklabels'] = kwargs['cbar_xticklabels']
    else:
        plot_kwargs['cbar_xticklabels'] = plot_kwargs['levels']

    
    if 'cbar_ticks' in kwargs.keys():
        cbar_ticks = kwargs['cbar_ticks']
    elif 'cbartick_offset' in kwargs.keys(): # TODO -  I don't think this works
        cbar_ticks =  np.arange(plot_kwargs['vmin']+ plot_kwargs['cbartick_offset'],
                                plot_kwargs['vmax']+ plot_kwargs['cbartick_offset'],
                                plot_kwargs['step'])
        # When this happens we usually want to cut off the last value
        plot_kwargs['cbar_xticklabels'] = plot_kwargs['cbar_xticklabels'][0:len(cbar_ticks)]

    else:  
        cbar_ticks = plot_kwargs['levels']
                                
                                
    plot_kwargs['cbar_ticks'] = cbar_ticks[:len(plot_kwargs['cbar_xticklabels'])]
    


    
    print(plot_kwargs)
    print('\n')
    return plot_kwargs


def sn_multi_window_in_time(unstable_sn_multi_window_da: xr.DataArray, 
                            stable_sn_multi_window_da: xr.DataArray,
                            abrupt_anom_smean: Union[xr.DataArray, xr.Dataset],
                            logginglevel='INFO',
                            **kwargs):
    
    '''
    Plot with window on LHS and temperature anomlay on RHS
    
    Parameters
    ----------
    unstable_sn_multi_window_da: xr.DataArray - 2D array of dims time and window
    stable_sn_multi_window_da: xr.DataArray - 2D array of dims time and window
    abrupt_anom_smean: xr.DataArray - 1D array with time dimension
    
    
    
    Returns
    --------
    fig, ax1, ax2, ax3, cbar
    
    Default values
    --------------
    height = 15, width = 7, hspace=0.3, vmin = -8, vmax = 8, step = 2, 
    cmap = 'RdBu_r', line_color = 'limegreen', line_alpha = 0.5, 
    cbar_label = 'S/N', cbartick_offset = 0, title='', label_size = 12, extend='both', 
    xlowerlim = None, xupperlim = None
    '''
    eval(f'logging.getLogger().setLevel(logging.{logginglevel})')
    # Upadting the plot kwargs
    plot_kwargs = sn_plot_kwargs(kwargs)
    

    fig = plt.figure(figsize = (plot_kwargs['height'], plot_kwargs['width']))
    gs = gridspec.GridSpec(2,1, height_ratios = [1, 0.1], hspace = plot_kwargs['hspace'])
    

    ### Window
    ax1 = fig.add_subplot(gs[0])

    
    # The plot doesn't accept values equal to the upper bound. E.g if upper bounds is
    # 100, and there is a value of 100, then this won't be plotted. Thus, any values less
    # than the upper bound are kept, but if equal to the upper bounds, a nominal amount
    # .01 is subtracted. [PHD-5]
    if plot_kwargs['filter_max'] == True:
        unstable_sn_multi_window_da =\
                    unstable_sn_multi_window_da.where((
            unstable_sn_multi_window_da < plot_kwargs['vmax']), plot_kwargs['vmax'] - .01)
    
    
    cs = unstable_sn_multi_window_da.plot(levels= plot_kwargs['levels'], cmap = plot_kwargs['cmap'], 
                                          extend=plot_kwargs['extend'], add_colorbar=False)

    stable_sn_multi_window_da.plot(cmap='gist_gray', extend=plot_kwargs['extend'],
                         alpha = 0.15, add_colorbar=False)

    ax1.set_ylabel('Window length (years)', size = plot_kwargs['label_size'])

    ### Colorbar
    ax3 = fig.add_subplot(gs[1])
    cbar = fig.colorbar(cs, cax = ax3, extend=plot_kwargs['extend'], orientation='horizontal')
    cbar.set_label(plot_kwargs['cbar_label'], size =plot_kwargs['label_size'])
    cbar.set_ticks(plot_kwargs['cbar_ticks'])
    cbar.ax.set_xticklabels(plot_kwargs['cbar_xticklabels'])
    logger.debug(f'cbar x-tick labels = {plot_kwargs["cbar_xticklabels"]}')

    ### Temperature Anomaly
    ax2 = ax1.twinx()
    
    if isinstance(abrupt_anom_smean, xr.DataArray):
        abrupt_anom_smean = abrupt_anom_smean.to_dataset(name='tas')
    
    # The variables to loop through and plot
    data_vars = list(abrupt_anom_smean.data_vars)
    
    
    # Usually use a red cmap, so making sure the lines are not red.
    no_red_colors = (plot_kwargs['line_color'], 'darkblue', 'green',
                     'yellow', 'purple', 'black', 'brown','darkgreen' , 'lightblue', 'greenyellow')
    # If there is only one line being used, I want to use a color of my choosing.
    # for the line and thea ax2 spines
    if len(data_vars) == 1:
        ax2.spines['right'].set_color(plot_kwargs['line_color'])
        ax2.tick_params(axis='y', colors=plot_kwargs['line_color'])

    print(data_vars)
    for i, dvar in enumerate(data_vars):
        print(str(i) + ' ', end='')
        da = abrupt_anom_smean[dvar]
        ax2.plot(da.time.dt.year.values, da.values,
                 alpha= plot_kwargs['line_alpha'], zorder=1000, label=dvar, linewidth = 2, 
                c = no_red_colors[i])


    
    # Only add a legend if there are multiple data vars
    if len(data_vars) > 1:
        if 'cbar_ncols' in plot_kwargs:
            ncol = plot_kwargs['cbar_ncols']
        else:
            ncol = len(data_vars)
        ax2.legend(ncol=ncol)

    ### General
    ax2.set_ylabel(r'Global Mean Temperature Anomaly ($^\circ$C)', size =12);

    ax1.set_xlabel('Time (years)', size =plot_kwargs['label_size'])
    ax1.xaxis.set_minor_locator(mticker.MultipleLocator(50))
    ax1.set_title(plot_kwargs['title'])

    ax1.set_xlim(plot_kwargs['xlowerlim'], plot_kwargs['xupperlim'])
    
    return (fig, ax1, ax2, ax3, cbar)


def format_plot(fig, ax):
    '''
    Small function for formatting map plots
    Reseson
    ------
    Usef in 07_exploring_consecutive_metrics_all_models_(nb_none)
    '''
    ax.coastlines(alpha=0.7)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.right_labels = False
    gl.top_labels = False


def format_colorbar(gs, pcolor, labelsize=10):
    '''
    Creates a colorbar that takes up all columns in the 0th row.
    The tick labels are percent
    
    Reason
    ------
    In 07_exploring_consecutive_metrics_all_models_(nb_none) a colorbar 
    of this type is used repeatedly. 
    '''
    cax = plt.subplot(gs[0,:])

    cbar = plt.colorbar(pcolor, cax=cax, orientation='horizontal')
    xticks = cbar.ax.get_xticks()
    cbar.ax.set_xticks(xticks)
    cbar.ax.set_xticklabels([str(int(xt)) +'%' for xt in xticks]);
    cbar.ax.tick_params(labelsize=labelsize)
    
    return cbar


def plot_all_period_maps(ds, periods, suptitle = 'Percent of Years That are Stable', cmap = 'RdBu', col_increase = 1,
                         row_increase = 2, 
                        y=0.89):
    '''
    Creates a plot of all the different periods 
    '''
    import utils
    
    data_vars = list(ds.data_vars)
    
    # Rows is the number of period, columns is the length of the data vars
    nrows, ncols = (len(periods) * row_increase, len(data_vars) * col_increase)
    
    
    fig = plt.figure(figsize = (8 * ncols, 5 * nrows))
    fig.suptitle(suptitle, y=y, fontsize=15)

    gs = gridspec.GridSpec(nrows + 1, ncols, height_ratios = [.2] + [1] * nrows, hspace=0.4, wspace=0)
    
    plot_num = ncols
    for period in periods:
        for dvar in data_vars:
            ax = fig.add_subplot(gs[plot_num], projection=ccrs.PlateCarree())
            da = ds[dvar].sel(period=period)
            pcolor = da.plot(
                ax=ax, vmin=0, vmax=100, cmap=cmap, extend='neither', add_colorbar=False)

            format_plot(fig, ax)

            formatted_period = utils.convert_period_string(period)
            ax.set_title(f'{formatted_period} {dvar.capitalize()}', fontsize=12);
            plot_num += 1
    
    cbar = format_colorbar(gs, pcolor)
    
    return fig, gs, cbar
