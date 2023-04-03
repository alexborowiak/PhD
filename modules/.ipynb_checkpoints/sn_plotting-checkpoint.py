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
# from constants import MODEL_PARAMS
from pprint import pprint, pformat
import sys,logging
import utils
logging.basicConfig(format="%(message)s", filemode='w', stream = sys.stdout)
logger = logging.getLogger()
sys.path.append('../')
import constants
import plotting_functions

# Usually use a red cmap, so making sure the lines are not red.
NO_RED_COLORS = ('darkblue', 'green','yellow', 'mediumpurple', 'black',
                 'lightgreen','lightblue', 'greenyellow')
MODEL_PROFILES = {'longrunmip': constants.LONGRUNMIP_MODEL_PARAMS, 'zecmip': constants.ZECMIP_MODEL_PARAMS}

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
    plot_kwargs = dict(height=12, width=22, hspace=0.3, vmin=-8, vmax=8, step=2, 
                       cmap = 'RdBu_r', line_color = 'limegreen', line_alpha=0.65, 
                       ax2_ylabel = 'Anomaly', cbar_label = 'Signal-to-Noise', cbartick_offset=0,
                       title='', label_size=12, extend='both', xlowerlim=None, xupperlim=None,  filter_max=True,)
    
    
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


def plot_all_coord_lines(da: xr.DataArray, coord='model', exp_type=None,
                         fig=None, ax:plt.Axes=None, figsize:tuple=(15,7),
                         font_scale=1, consensus=True, xlabel=None, ylabel=None,
                         bbox_to_anchor=(1.02,1),
                        **kwargs):
    '''
    Plots all of the values in time for a coordinate. E.g. will plot all of the models values
    in time for the global average or for a given grid cell.
    '''
    
    fig = plt.figure(figsize=figsize) if not fig else fig
    ax = fig.add_subplot(111) if not ax else ax
    
    coord_values = list(da[coord].values)
    time = da.time.values
    logger.info(f'{coord_values=}')
    if exp_type:
        MODEL_PARAMS = MODEL_PROFILES[exp_type]
        coord_values = [model for model in list(MODEL_PARAMS) if model in coord_values]
    
    for i, coord_value in enumerate(coord_values):
        logger.debug(f'{i} {coord_value}, ')
       
        if exp_type:
            c = MODEL_PARAMS[coord_value]['color']
        else:
            c = NO_RED_COLORS[i]

        
        label=coord_value
        if exp_type:
            if coord_value in list(MODEL_PARAMS):
                ECS = MODEL_PARAMS[coord_value]['ECS']
                label += f' ({ECS}K)' 
            
        ax.plot(time, da.loc[{coord:coord_value}].values,
                alpha=kwargs['line_alpha'] if 'line_alpha' in kwargs else 1,
                zorder=1000, label=label, linewidth=2,  
                c=c)
    if consensus: ax.plot(time, da.mean(dim=coord).values,
                alpha=kwargs['line_alpha'] if 'line_alpha' in kwargs else 1,
                zorder=1000, label='Mean', linewidth=2,  
                c='black')
    
    if len(coord_values) > 1:
        leg = ax.legend(ncol=4, bbox_to_anchor=bbox_to_anchor,
                        fontsize=constants.PlotConfig.legend_text_size*font_scale)
        leg.set_title('Model')
        leg.get_title().set_fontsize(constants.PlotConfig.legend_title_size*font_scale)
        
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def sn_multi_window_in_time(unstable_da:xr.DataArray,  stable_da:xr.DataArray,
                            temp_da: Union[xr.DataArray, xr.Dataset], stable_point_ds:xr.Dataset=None,
                            fig=None, gs=None, ax1=None, max_color_lim=None, 
                            logginglevel='ERROR', exp_type=None, font_scale=1.5,
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
    mpl.rcParams.update(mpl.rcParamsDefault)
    utils.change_logging_level(logginglevel)
    
    plot_kwargs = sn_plot_kwargs(kwargs, logginglevel)
      
    # ---> Updating x lims
    xlims = dict(time=slice(plot_kwargs['xlowerlim'], plot_kwargs['xupperlim']))
    unstable_da = unstable_da.isel(**xlims)
    stable_da = stable_da.isel(**xlims)
    temp_da = temp_da.isel(**xlims)
    
    if max_color_lim:
        unstable_da = unstable_da.isel(time=slice(None, max_color_lim))

    unstable_da['time'] = unstable_da.time.dt.year.values
    stable_da['time'] = stable_da.time.dt.year.values
    temp_da['time'] = temp_da.time.dt.year.values
    
    # ---> Creating plot
    fig = plt.figure(figsize=(plot_kwargs['width'], plot_kwargs['height'])) if not fig else fig
    gs = gridspec.GridSpec(2,1, height_ratios=[1, 0.1], hspace=plot_kwargs['hspace']) if not gs else gs
    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()

    # ---> Plotting colors
    if plot_kwargs['filter_max'] == True:
        unstable_da = unstable_da.where((unstable_da < plot_kwargs['vmax']), plot_kwargs['vmax'] - .01)
    
    stable_da = xr.where(np.isfinite(stable_da), .4, 0) # Only want grey
    stable_da.plot(ax=ax1, cmap='gist_gray_r', alpha=0.3,vmax=1, vmin=0, add_colorbar=False)
    
    cs = unstable_da.plot(ax=ax1, levels=plot_kwargs['levels'], cmap=plot_kwargs['cmap'], 
                          extend=plot_kwargs['extend'], add_colorbar=False)
    
    # ---> Stable Year 
    if stable_point_ds: stable_point_ds.time.plot(y='window',  ax=ax1, color='k')

    # ---> Temperature Anomaly
    plot_all_coord_lines(da=temp_da, ax=ax2, fig=fig, exp_type=exp_type, font_scale=font_scale, bbox_to_anchor=(1, 1.3))
                         #bbox_to_anchor=(1.07,1.1))
    
    
    # ---> Colorbar
    cax = fig.add_subplot(gs[1])
    cbar = fig.colorbar(cs, cax=cax, extend=plot_kwargs['extend'], orientation='horizontal')
    cbar.set_label(plot_kwargs['cbar_label'], size=constants.PlotConfig.cmap_title_size*font_scale)
    cbar.set_ticks(plot_kwargs['cbar_ticks'])
    cbar.ax.set_xticklabels(plot_kwargs['cbar_xticklabels'])
    cbar.ax.tick_params(labelsize=constants.PlotConfig.legend_text_size*font_scale)
    logger.debug(f'cbar x-tick labels = {plot_kwargs["cbar_xticklabels"]}')
    

    
    # ---> Axes formatting
    ax1.set_xlim(plot_kwargs['xlowerlim'], plot_kwargs['xupperlim'])
    xlabel = 'Time (Years)' if 'xlabel' not in plot_kwargs else plot_kwargs['xlabel']
    plotting_functions.format_axis(ax1, xlabel=xlabel, ylabel='Window Length\n(Years)', font_scale=font_scale)
    plotting_functions.format_axis(ax2, xlabel=xlabel, ylabel=plot_kwargs['ax2_ylabel'], font_scale=font_scale)
    
    ax1.set_title('')
    ax2.set_title('')
    
    fig.suptitle(plot_kwargs['title'], fontsize=18, y=0.92)
  
    return (fig, [ax1, ax2, cax], cbar)


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


def format_colorbar(gs, pcolor, labelsize=10, cax=None, tick_symbol='%'):
    '''
    Creates a colorbar that takes up all columns in the 0th row.
    The tick labels are percent
    
    Reason
    ------
    In 07_exploring_consecutive_metrics_all_models_(nb_none) a colorbar 
    of this type is used repeatedly. 
    '''
    cax = plt.subplot(gs[0,:]) if not cax else cax

    cbar = plt.colorbar(pcolor, cax=cax, orientation='horizontal')
    xticks = cbar.ax.get_xticks()
    cbar.ax.set_xticks(xticks)
    if tick_sybmol: cbar.ax.set_xticklabels([str(int(xt)) + tick_symbol for xt in xticks]);
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

def plot_all_model_multi_window_maps(ds, variable:int, plot_dict:Dict, cmap='Reds', extend='max'):

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




def plot_stable_year_all_models(ds, fig=None, ax=None, linestyle='dashed', exp_type=None, add_legend=True, 
                               legend_loc='right', ncol=1, bbox_to_anchor=None, font_scale=1):
    ''''
    Plotting the year of stabilisation at each window for different models
    '''
    plt.style.use('seaborn-darkgrid')
    if not fig: fig = plt.figure(figsize=(10, 8))
    if not ax: ax = fig.add_subplot(111)
    
    if exp_type in MODEL_PROFILES:
        information_profile = MODEL_PROFILES[exp_type]
        models = np.intersect1d(ds.model.values, list(information_profile))
    else:
        colors = constants.RANDOM_COLOR_LIST
        models = ds.model.values
      
    
    ds.median(dim='model').time.plot(ax=ax,y='window', label='Median', color='k', linewidth=1.5, 
                                    linestyle='solid') 
    
    for num, model in enumerate(models):
        if not exp_type:
            color = colors[num]
            label=model
        else:
            color = information_profile[model]['color']
            ECS = information_profile[model]['ECS']
            label = f'{model} ({ECS=}K)'
            
        da = ds.sel(model=model).time.plot(ax=ax,y='window', linewidth=1.5, alpha=0.8,
                                           color=color, label=label, linestyle=linestyle)
        
    ylims = np.take(ds.window.values, [0,-1])
    xlims = [np.min(ds.time.values)-5, np.max(ds.time.values)]
    if isinstance(ncol, str):
        if ncol == 'coords': ncol = len(models)
    if add_legend:
        if isinstance(bbox_to_anchor, tuple): bbox_to_anchor=bbox_to_anchor
        else:
            if legend_loc == 'right': bbox_to_anchor=(1, 0.857)
            elif legend_loc == 'top_ofset': bbox_to_anchor=(1.5, 1.1)
        leg = ax.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, frameon=True, facecolor='white',
                        fontsize=constants.PlotConfig.legend_text_size*font_scale)
        leg.set_title('Model')
        leg.get_title().set_fontsize(constants.PlotConfig.legend_title_size*font_scale)
    plotting_functions.format_axis(ax, xlabel='Year of Stabilisation', ylabel='Window Length\n(Years)',
                                   font_scale=font_scale, labelpad=76)
    ax.set_title('')
    
    return fig, ax

def plot_median_stable_year(ds1, ds2, fig=None, ax=None):
    ''''
    Plotting the median year of stabilisation at each window for two 
    different datasets.
    '''
    plt.style.use('seaborn-darkgrid')
    if not fig: fig = plt.figure(figsize=(10, 8))
    if not ax: ax = fig.add_subplot(111)

        
    ds1.median(dim='model').time.plot(ax=ax,y='window', label='rolling', color='k', linewidth=1.5, 
                                    linestyle='solid') 
            
    ds2.median(dim='model').time.plot(ax=ax,y='window', label='static', color='k', linewidth=1.5, 
                                    linestyle='dashed') 

    ylims = np.take(ds1.window.values, [0,-1])
    xlims = [np.min(ds1.time.values)-5, np.max(ds1.time.values)]
    leg = ax.legend(ncol=1, bbox_to_anchor=[1, 0.857], frameon=True, facecolor='white', 
                   fontsize=14)
    leg.set_title('Noise Type')
    leg.get_title().set_fontsize('16')
    ax.set_xlabel('Year of Stabilisation', fontsize=16)
    ax.set_ylabel('Window Length (years)', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title('')
    return fig, ax