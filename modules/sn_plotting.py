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
from constants import MODEL_PARAMS
from pprint import pprint, pformat
import sys,logging
import utils
logging.basicConfig(format="%(message)s", filemode='w', stream = sys.stdout)
logger = logging.getLogger()
sys.path.append('../')
import constants
# experiment_colors = {'tas_global': 'red', 'tas_land_global': 'lightcoral',
#                      'pr_global': 'brown', 'pr_land_global': 'peru', 
#                     'sic_sea_global': 'blue', 'sic_sea_northern_hemisphere': 'darkblue',
#                        'sic_sea_southern_hemisphere': 'cornflowerblue', 'tos_sea_global': 'orange'}

experiment_colors = {'tas_global': 'red', 'tas_land_global': 'lightcoral',
                     'pr_global': 'green', 'pr_land_global': 'yellowgreen', 
                     'tos_sea_global': 'blue'}


def highlight_plot(ax, ds, ds_highlight=None, legend_on:bool =True, yaxis_right:bool=False, label=None,
                  color='tomato', highlight_color='darkred', bbox_to_anchor = [-0.03, 1]):
    '''Plots a line a dash line with the option of another solid line being plotted over the top'''
    if yaxis_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        color = 'green'
        highlight_color = 'darkgreen'
        bbox_to_anchor = [-0.03, 0.8] # 1.19 for rhs
        

    ax.plot(ds.time.values, ds.values,
             color = color, alpha = 0.4, label  = 'Unstable', linestyle='--')
    
    if isinstance(ds_highlight, xr.DataArray):
        ax.plot(ds_highlight.time.values, ds_highlight.values,
                 color = highlight_color, alpha = 0.8, label  = 'Stable')
    else:
        legend_on = False # Turn legend off if only one line
    c1 = plt.gca().lines[0].get_color()
    ax.set_ylabel(label, fontsize = 18,
                   color = c1, rotation = 0, labelpad = 55);
    
    if legend_on:
        leg = ax.legend(ncol=1, fontsize=15, bbox_to_anchor=bbox_to_anchor, frameon=True)
        leg.set_title(label)
        leg.get_title().set_fontsize('15')
        
    major_ticks, minor_ticks = utils.get_tick_locator(ds.values)
    
    
    ax.yaxis.set_major_locator(mticker.MultipleLocator(major_ticks))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(minor_ticks))
    ax.tick_params(axis = 'y', labelsize=14, labelcolor = c1)
    ax.tick_params(axis='x', labelsize=14)
    
    ax.set_xlim(ds.time.values[0], ds.time.values[-1])
    ax.set_xlabel('Time (Years)', fontsize=18)


def two_line_highlight_plot(left_ds:xr.DataArray=None, 
                            right_ds:xr.DataArray=None,
                            left_highlight_ds:xr.DataArray=None,
                            right_highlight_ds:xr.DataArray=None, 
                            left_label = None, right_label=None,
                            bounds:Dict[str, float] = None):
    plt.style.use('seaborn-darkgrid')

    fig = plt.figure(figsize=  (15,10))
    ax1 = fig.add_subplot(111)
    
    if isinstance(left_ds, xr.DataArray):
        highlight_plot(ax1, left_ds, ds_highlight = left_highlight_ds,
                       label=left_label)
        
    if isinstance(right_ds, xr.DataArray):
        ax2 = ax1.twinx()
        highlight_plot(ax2, right_ds, ds_highlight = right_highlight_ds,
                       yaxis_right=True, label=right_label)
    else:
        ax2=None
        
    if isinstance(bounds, dict):
        for key, value in bounds.items():
            ax1.plot([left_ds.time.values[0], left_ds.time.values[-1]], [value, value], 
                   color='tomato', linestyle=':', alpha=0.8)
        
    return fig, ax1, ax2




def temperature_vs_sn_plot(ax,
                           sn:xr.DataArray=None,
                           temp:xr.DataArray=None,
                           temp_highlight:xr.DataArray=None,
                           sn_highlight:xr.DataArray=None,
                          bounds:Dict[str, float] = None):
    print('!!!!!!! Warning: This is a legacy function and is no longer supported.')
    print('Please use two_line_highlight_plot is sn_plotting')
    plt.style.use('seaborn-darkgrid')

    if isinstance(sn, xr.DataArray):
        highlight_plot(ax, sn, ds_highlight = sn_highlight,
                       label='Signal\to\nNoise')
    
    ax2 = ax.twinx()
    highlight_plot(ax2, temp, ds_highlight = temp_highlight,
                   yaxis_right=True, label='GMST\nAnomaly'+ r' ($^{\circ}$C)')
    
    if isinstance(bounds, dict):
        for key, value in bounds.items():
            ax.plot([sn.time.values[0], sn.time.values[-1]], [value, value], 
                   color='tomato', linestyle=':', alpha=0.8)
    
    return ax, ax2



def sn_plot_kwargs(kwargs, logginglevel='ERROR'):
    
    utils.change_logging_level(logginglevel)
    # The default plot kwargs
    plot_kwargs = dict(height = 15, width = 7, hspace=0.3, vmin = -8, vmax = 8, step = 2, 
                      cmap = 'RdBu_r', line_color = 'limegreen', line_alpha = 0.65, 
                       ax2_ylabel = 'Anomaly',
                      cbar_label = 'Signal-to-Noise',cbartick_offset = 0,title='', label_size = 12, extend='both', 
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
    
    
    if 'ax2_ylabel' in kwargs.keys():
        cbar_ticks = kwargs['ax2_ylabel']
   
    
    logger.info(pformat(plot_kwargs))
    return plot_kwargs


def sn_multi_window_in_time(unstable_sn_multi_window_da: xr.DataArray, 
                            stable_sn_multi_window_da: xr.DataArray,
                            abrupt_anom_smean: Union[xr.DataArray, xr.Dataset],
                            stable_point_ds:xr.Dataset = None,
                            logginglevel='ERROR',
                            font_scale = 1,
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
    utils.change_logging_level(logginglevel)
    # Upadting the plot kwargs
    plot_kwargs = sn_plot_kwargs(kwargs, logginglevel)
    
    
    # TODO: DRYER way of doing this.
    unstable_sn_multi_window_da = unstable_sn_multi_window_da.isel(
        time=slice(plot_kwargs['xlowerlim'], plot_kwargs['xupperlim']))
    unstable_sn_multi_window_da['time'] = unstable_sn_multi_window_da.time.dt.year.values
    
    stable_sn_multi_window_da = stable_sn_multi_window_da.isel(
        time=slice(plot_kwargs['xlowerlim'], plot_kwargs['xupperlim']))
    stable_sn_multi_window_da['time'] = stable_sn_multi_window_da.time.dt.year.values

    
    abrupt_anom_smean = abrupt_anom_smean.isel(
        time=slice(plot_kwargs['xlowerlim'], plot_kwargs['xupperlim']))
    abrupt_anom_smean['time'] = abrupt_anom_smean.time.dt.year.values
    

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
    
    
    xr.where(np.isfinite(stable_sn_multi_window_da), .4, 0).plot(ax=ax1,
                    cmap='gist_gray_r', alpha=0.3,vmax=1, vmin=0, add_colorbar=False)
    
    cs = unstable_sn_multi_window_da.plot(ax=ax1, levels=plot_kwargs['levels'], cmap = plot_kwargs['cmap'], 
                                          extend=plot_kwargs['extend'], add_colorbar=False)
    if stable_point_ds:
        stable_point_ds.time.plot(y='window',  ax=ax1, color='k')

    ax1.set_ylabel('Window length (years)', size = plot_kwargs['label_size'])

    ### Colorbar
    ax3 = fig.add_subplot(gs[1])
    cbar = fig.colorbar(cs, cax = ax3, extend=plot_kwargs['extend'], orientation='horizontal')
    cbar.set_label(plot_kwargs['cbar_label'], size =24 * font_scale)#plot_kwargs['label_size'])
    cbar.set_ticks(plot_kwargs['cbar_ticks'])
    cbar.ax.set_xticklabels(plot_kwargs['cbar_xticklabels'])
    cbar.ax.tick_params(labelsize=20 * font_scale)
    logger.debug(f'cbar x-tick labels = {plot_kwargs["cbar_xticklabels"]}')

    ### Temperature Anomaly
    ax2 = ax1.twinx()
    
    if isinstance(abrupt_anom_smean, xr.DataArray):
        abrupt_anom_smean = abrupt_anom_smean.to_dataset(name='variable')
    
    # The variables to loop through and plot
    data_vars = list(abrupt_anom_smean.data_vars)
    
    
    # Usually use a red cmap, so making sure the lines are not red.
    no_red_colors = (plot_kwargs['line_color'], 'darkblue', 'green',
                     'yellow', 'mediumpurple', 'black','lightgreen' , 'lightblue', 'greenyellow')
    # If there is only one line being used, I want to use a color of my choosing.
    # for the line and thea ax2 spines
    if len(data_vars) == 1:
        ax2.spines['right'].set_color(plot_kwargs['line_color'])
        ax2.tick_params(axis='y', colors=plot_kwargs['line_color'])
    
    time = abrupt_anom_smean.time.values # .dt.year
    logger.info(f'{data_vars=}')
    models = [model for model in list(MODEL_PARAMS) if model in data_vars]
    
    if len(models) == 0:
        logger.error(f'No matching models found {data_vars=}')
        models = data_vars

    for i, dvar in enumerate(models):
        logger.debug(f'{i} {dvar}, ')
        
        if dvar in list(MODEL_PARAMS):
            c = MODEL_PARAMS[dvar]['color']
        else:
            c = no_red_colors[i]
            
        da = abrupt_anom_smean[dvar]
        label=dvar
        
        if dvar in list(MODEL_PARAMS):
            ECS = MODEL_PARAMS[dvar]['ECS']
            label += f' ({ECS}K)' 
            
        ax2.plot(time, da.values,
                 alpha= plot_kwargs['line_alpha'], zorder=1000, label=label, linewidth = 2,  
                c = c)

    if len(data_vars) > 1:
        leg = ax2.legend(ncol=1, fontsize = 20 * font_scale, bbox_to_anchor=[1.07,1])
        leg.set_title('Model')
        leg.get_title().set_fontsize(str(int('24')  * font_scale))

    ax2.set_ylabel(plot_kwargs['ax2_ylabel'], rotation=0, size =24 * font_scale, labelpad=100 * font_scale);
    ax1.set_ylabel('Window Length\n(Years)', fontsize=24 * font_scale, rotation=0, labelpad=100 * font_scale)
    ax1.set_xlim(plot_kwargs['xlowerlim'], plot_kwargs['xupperlim'])
    ax1.set_xlabel('Time (years)', size=24 * font_scale)#plot_kwargs['label_size'])
    ax2.set_xlabel('Time (years)', size=24 * font_scale)#plot_kwargs['label_size'])

    ax1.tick_params(axis='y', labelsize=20 * font_scale)
    ax2.tick_params(axis='y', labelsize=20 * font_scale)
    ax1.tick_params(axis='x', labelsize=20 * font_scale)

    ax1.set_title('')
    ax2.set_title('')
    
    fig.suptitle(plot_kwargs['title'], fontsize=18, y=0.92)
  
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
    gl.left_labels = False
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



def plot_year_of_stability(ds: xr.Dataset, varible_to_loop: str, title:str=None):
    '''
    Plots the year of stability for different window lenght. This can be 
    for any variable that is stored as a coordinte. 
    
    ds: xr.Dataset
        
    
    '''
    
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for variable in ds[varible_to_loop].values:
        color = experiment_colors[variable]
        label = variable.replace('_', ' - ').replace('sea -', '')
        label = label.replace('- global', '')

        da = ds.sel(variable=variable).time.plot(ax=ax,y='window', label=label,
                                                linewidth=1.5, color=color, alpha=0.8)
    if title is None:
        model = str(ds.model.values)
        ECS = f' (ECS={constants.MODEL_PARAMS[model]["ECS"]}K)'
        title = f'{model} Year of Stabilisation {ECS}'
        
    ax.set_title(title, fontsize=25)


#     ax.legend(fontsize=25)
    leg = ax.legend(ncol=1, frameon=True, facecolor='white', fontsize=18) # , bbox_to_anchor=[1, 0.857]
    leg.set_title('Variable')
    leg.get_title().set_fontsize('18')
    ax.set_xlim(-1, np.max(ds.time.values))
    ax.set_ylim(np.min(ds.window.values), np.max(ds.window.values))
    ax.set_xlabel('Year of Stabilisation', fontsize=18)
    ax.set_ylabel('Window Length (years)', fontsize=18)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
#     ax.set_xlabel('Year of Stabilisation', fontsize=18)
#     ax.set_ylabel('Window lengths (years)', fontsize=18)
    
    return fig, ax

def local_stabilisation_average_year_and_uncertainty_plot(ds, plot_dict, suptitle=None, cmap='Reds'):
    '''
    ds: xr.Dataset
        Coords: window, lat, lat
        Data vars: median_value, uncertainty
    plot_dict:
        dictionary of values for plot
    '''
    windows = ds.window.values

    fig = plt.figure(figsize=(8.3 * len(windows), 12))
    
    gs = gridspec.GridSpec(3, len(windows), height_ratios=[0.2, 1,1])

    axes = []
    plots = []

    if suptitle:
        fig.suptitle(suptitle, fontsize=25)

    y_axis_kwargs = dict(xy=(-0.05, 0.5), ha='center', va='center', xycoords='axes fraction', 
                       rotation=90, size=18)

    for plot_num, window in enumerate(windows):    

        ax = fig.add_subplot(gs[1, plot_num], projection=ccrs.PlateCarree())
        da = ds.sel(window=window).median_value
        plot = da.plot(ax=ax, cmap=cmap, add_colorbar=False, levels=plot_dict[window]['levels'])

        ax.coastlines()
        ax.set_title(f'{window} Year Window', fontsize=18)
        format_plot(fig, ax)

        if not plot_num:
            ax.annotate('Mean', **y_axis_kwargs)

        axes.append(ax)
        plots.append(plot)

    for plot_num, window in enumerate(windows):

        ax = fig.add_subplot(gs[2, plot_num], projection=ccrs.PlateCarree())
        da = ds.sel(window=window).uncertainty
        plot = da.plot(ax=ax, cmap=cmap, add_colorbar=False)
        ax.coastlines()
        format_plot(fig, ax)

        if not plot_num:
            ax.annotate('Uncertainty', **y_axis_kwargs)

        ax.set_title('')
        axes.append(ax)
        plots.append(plot)


    for plot_num, plot in enumerate(plots[:len(windows)]):
        cax = plt.subplot(gs[0, plot_num])
        cbar = plt.colorbar(plot, cax=cax, orientation='horizontal')
        cbar.ax.set_title('Year of Stabilisation', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
    return fig

def plot_all_model_multi_window_maps(ds, variable, plot_dict, cmap='Reds', extend='max'):

    print(plot_dict)
    
    windows = ds.window.values
    models = ds.model.values
    num_cols = len(windows)
    num_rows = len(models)

    fig = plt.figure(figsize=(6*num_cols, 4.*num_rows))
    gs = gridspec.GridSpec(num_rows+1, num_cols, height_ratios = [0.2] + [1] * num_rows,
                           hspace=0.2, wspace=0.2)

    fig.suptitle(f'{constants.VARIABLE_INFO[variable]["longname"]} Year of Stabilisation', 
                fontsize=25, y=.91)

    axes = []
    plots = []

    y_axis_kwargs = dict(xy=(-0.05, 0.5), ha='center', va='center', xycoords='axes fraction', 
                       rotation=90, size=18)

    for row, model in enumerate(models):
        for col, window in enumerate(windows):
            ax = fig.add_subplot(gs[row+1, col], projection=ccrs.PlateCarree())
            da = ds.time.sel(window=window, model=model)
            plot = da.plot(ax=ax, cmap=cmap, levels = plot_dict[window]['levels'], 
                           add_colorbar=False, extend=extend)
            ax.set_title('')
            if not col:
                ax.annotate(f'{model}', **y_axis_kwargs)
            ax.coastlines()
            format_plot(fig, ax)
            axes.append(ax)
            plots.append(plot)

    for window,ax in zip(windows, axes[:len(windows)]):
        ax.set_title(f'{window} Year Window', fontsize=18)

    for plot_num, plot in enumerate(plots[:len(windows)]):
        cax = plt.subplot(gs[0, plot_num])
        cbar = plt.colorbar(plot, cax=cax, orientation='horizontal')
        cbar.ax.set_title('Year of Stabilisation', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        
    return fig, axes, plots
# def temperature_vs_sn_plot(ax,
#                            sn:xr.DataArray=None,
#                            temp:xr.DataArray=None,
#                            temp_highlight:xr.DataArray=None,
#                            sn_highlight:xr.DataArray=None,
#                           bounds:Dict[str, float] = None):
#     '''
#     Plot the temperature and signal_to_noise (could also be just
#     signal or just noise)
    
#     Parameters
#     ----------
#     ax: matplotlib axis
#     sn, temp: temp_highlight=None, sn_highlight=None: xr.DataArray
    
#     Returns
#     --------
    
#     [ax, ax2]: matplotlib axis
    
#     All datasets in this plot are xr.DataArrays.
#     *_highlight are optiononal parameters that can be added to highligh certain
#     sections of the plot
    
#     '''    
    
# #     mpl.rcParams.update(mpl.rcParamsDefault)
#     plt.style.use('seaborn-darkgrid')

#     ax.plot(sn.time.values,sn.values, label = 'Unstable', c = 'tomato', linestyle='--')
#     if isinstance(sn_highlight, xr.DataArray):
#         ax.plot(sn_highlight.time.values,sn_highlight.values, label = 'Stable', c = 'darkred')


#     c0 = plt.gca().lines[0].get_color()
#     ax.tick_params(axis = 'y', labelcolor = c0)
#     ax.set_ylabel('Signal to Noise', fontsize = 16, color = c0, rotation = 0, labelpad = 55);
    
    
        
#     if isinstance(bounds, dict):
#         for key, value in bounds.items():
#             ax.plot([temp.time.values[0], temp.time.values[-1]], [value, value], 
#                    color=c0, linestyle=':', alpha=0.8)
    
#     leg = ax.legend(ncol = 1, fontsize = 15, bbox_to_anchor = [-0.03, 1])
#     leg.set_title('S/N')
#     leg.get_title().set_fontsize('15')

#     ax.yaxis.set_major_locator(mticker.MultipleLocator(.5))
#     ax.yaxis.set_minor_locator(mticker.MultipleLocator(.25))

#     # Second y-axis: the temperature anomalies.
#     ax2 = ax.twinx()

#     ax2.plot(temp.time.values, temp.values,
#              color = 'green', alpha = 0.4, label  = 'Unstable', linestyle='--')
    
#     if isinstance(temp_highlight, xr.DataArray):
#         ax2.plot(temp_highlight.time.values, temp_highlight.values,
#                  color = 'darkgreen', alpha = 0.8, label  = 'Stable')

#     c1 = plt.gca().lines[0].get_color()

#     ax2.spines['right'].set_color(c1)
#     ax2.spines['left'].set_color(c0)
#     ax2.set_ylabel('Tempearture\nAnomaly'+ r' ($^{\circ}$C)', fontsize = 16,
#                    color = c1, rotation = 0, labelpad = 55);
    
#     leg2 = ax2.legend(ncol = 1, fontsize = 15, bbox_to_anchor = [-0.03, 0.8]) # 1.19 for rhs
#     leg2.set_title('Temperature\nAnomaly')
#     leg2.get_title().set_fontsize('15')
    
    
#     ax2.yaxis.set_major_locator(mticker.MultipleLocator(.5))
#     ax2.yaxis.set_minor_locator(mticker.MultipleLocator(.25))
#     ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
#     ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
#     ax2.tick_params(axis = 'y', labelsize=14, labelcolor = c1)
#     ax2.set_xlim(temp.time.values[0], temp.time.values[-1])

#     ax.tick_params(axis='y', labelsize=14)
#     ax.tick_params(axis='x', labelsize=14)
    
    
#     ax.set_xlabel('Year', fontsize = 16);
#     ax.set_xlim(temp.time.values[0], temp.time.values[-1])
    
#     return ax, ax2