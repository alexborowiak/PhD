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
from typing import Dict, List
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
from matplotlib.patches import Rectangle
from numpy.typing import ArrayLike

# Usually use a red cmap, so making sure the lines are not red.
NO_RED_COLORS = ('k', 'green','yellow', 'mediumpurple', 'black',
                 'lightgreen','lightblue', 'greenyellow')

MODEL_PROFILES = {'longrunmip': constants.LONGRUNMIP_MODEL_PARAMS, 'zecmip': constants.ZECMIP_MODEL_PARAMS}

# experiment_colors = {'tas_global': 'red', 'tas_land_global': 'lightcoral',
#                      'pr_global': 'brown', 'pr_land_global': 'peru', 
#                     'sic_sea_global': 'blue', 'sic_sea_northern_hemisphere': 'darkblue',
#                        'sic_sea_southern_hemisphere': 'cornflowerblue', 'tos_sea_global': 'orange'}

experiment_colors = {'tas_global': 'red', 'tas_land_global': 'lightcoral',
                     'pr_global': 'green', 'pr_land_global': 'yellowgreen', 
                     'tos_sea_global': 'blue'}



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
    else: ax2=None
        
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



plot_kwargs = dict(height=12, width=22, hspace=0.3, #vmin=-8, vmax=8, step=2, 
                   cmap = 'RdBu_r', line_color = 'limegreen', line_alpha=0.65, 
                   ax2_ylabel = 'Anomaly', cbar_label = 'Signal-to-Noise', cbartick_offset=0,
                   axes_title='', 
                   title='', label_size=12, extend='both', xlowerlim=None, xupperlim=None,  filter_max=True,)



def plot_all_coord_lines(da: xr.DataArray, coord='model', exp_type=None,
                         fig=None, ax:plt.Axes=None, figsize:tuple=(15,7),
                         font_scale=1, consensus=True, xlabel=None, ylabel=None, labelpad=60,
                         bbox_to_anchor=(1.02,1), ncol=4, add_legend=True, xlim=None, ylim=None, title=None, 
                        **kwargs):
    '''
    Plots all of the values in time for a coordinate. E.g. will plot all of the models values
    in time for the global average or for a given grid cell.
    '''
    
    fig = plt.figure(figsize=figsize) if not fig else fig
    ax = fig.add_subplot(111) if not ax else ax
    
    coord_values = da[coord].values.flatten() # Flatten in-case 0D array
    
    time = da.time.values
    if exp_type:
        MODEL_PARAMS = MODEL_PROFILES[exp_type]
        coord_values = [model for model in list(MODEL_PARAMS) if model in coord_values]
    
    for i, coord_value in enumerate(coord_values):
        logger.debug(f'{i} {coord_value}, ')
       
        if exp_type: kwags_to_use = dict(c = MODEL_PARAMS[coord_value]['color'])
        else: kwags_to_use = dict(c = NO_RED_COLORS[i])

        label=coord_value
        if exp_type:
            if coord_value in list(MODEL_PARAMS):
                ECS = MODEL_PARAMS[coord_value]['ECS']
                label += f' ({ECS}K)' 
        da_to_plot = da.loc[{coord:coord_value}].values if len(coord_values) > 1 else da.values
    
        ax.plot(time, da_to_plot,
                alpha=kwargs['line_alpha'] if 'line_alpha' in kwargs else 1,
                zorder=1000, label=label, linewidth=3, **kwags_to_use)
        
         
    if consensus and len(coord_values) > 1: ax.plot(time, da.mean(dim=coord).values,
                alpha=kwargs['line_alpha'] if 'line_alpha' in kwargs else 1,
                zorder=1000, label='Mean', linewidth=3,  
                c='black')
    
    if isinstance(xlim, tuple): ax.set_xlim(da.time.values[xlim[0]], da.time.values[xlim[-1]])
    if isinstance(ylim, tuple): ax.set_ylim(ylim)
    if len(coord_values) > 1 and add_legend:
        leg = ax.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor,
                        fontsize=constants.PlotConfig.legend_text_size*font_scale)
        leg.set_title(coord.capitalize())
        leg.get_title().set_fontsize(constants.PlotConfig.legend_title_size*font_scale)
        
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, labelpad=labelpad)
    return fig, ax



def create_levels(vmax:float, vmin:float=None, step:float=1)->np.ndarray:
    '''
    Ensures that all instances of creating levels using vmax + step as the max.
    '''
    vmin = -vmax if vmin is None else vmin
    return np.arange(vmin, vmax + step, step)

def create_colorbar(plot, cax, levels, ticks='', cbar_title='', 
                    labelpad=30, title_rotation=0, rotation=0, orientation='horizontal', 
                    extend='neither', tick_offset=None, cut_ticks=1, shrink=1, round_level=2,
                    font_scale=1):
    '''
    plot: the plot that th cbar is refering to.
    caxes: the colorbar axes.
    levels: the levels on the plot
    '''
    
    cbar = plt.colorbar(plot, cax=cax, orientation=orientation, extend=extend, shrink=shrink)
    
    tick_locations = levels; tick_labels = levels
    if tick_offset == 'center':
        tick_locations = levels[:-1] + np.diff(levels)/2
        tick_labels = tick_labels[:-1]
    if cut_ticks > 1:
        tick_locations = tick_locations[::cut_ticks]
        tick_labels = tick_labels[::cut_ticks]
    
    cbar.set_ticks(tick_locations)    
    tick_labels = np.round(tick_labels, round_level)
        
    if orientation == 'horizontal':
        cbar.ax.set_xticklabels(tick_labels, fontsize=constants.PlotConfig.legend_text_size*font_scale, rotation=rotation)
        cbar.ax.set_title(cbar_title, size=constants.PlotConfig.cmap_title_size*font_scale)
    else:
        cbar.ax.set_yticklabels(tick_labels, fontsize=constants.PlotConfig.legend_text_size*font_scale, 
                                rotation=rotation)
        cbar.ax.set_ylabel(cbar_title, size=constants.PlotConfig.label_size*cmap_title_size, 
                           rotation=title_rotation, labelpad=labelpad)
        
    return cbar


def format_colorbar(pcolor, gs=None, cax:plt.Axes=None, tick_symbol:str='%'):
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


def plot_heatmap(da:xr.DataArray, fig:plt.figure=None, gs=None, ax:plt.Axes=None, cax:plt.Axes=None,
                 figsize:tuple=None, cmap='Blues', extend='neither', max_color_lim:int=None,
                 levels:list=None, vmin=None, vmax=None, step=None, 
                 xlims:tuple=None, font_scale=1,
                 cbar_tile:str='', tick_labels=None, add_colorbar=True, cbar_label=None,
                 tick_offset=None, cut_ticks=1, patch=False, hspace=0,
                 title:str=None, axes_title:str=None, labelpad=100, rotation=0,
                 ylabel='Window Length\n(Years)', xlabel='Time (Years)', return_all=True, **kwargs):
    '''
    Plots a heatmatp of ds. Lots of options for entering different arguements
    '''
    
    figsize = figsize if figsize is not None else (plot_kwargs['width'], plot_kwargs['height'])
    fig = fig if fig is not None else plt.figure(figsize=figsize)
    gs = (gs if gs is not None else gridspec.GridSpec(2,1, height_ratios=[1, 0.1],
                                                      hspace=plot_kwargs['hspace']+hspace))
    ax = ax if ax is not None else fig.add_subplot(gs[0])
    
    
    if xlims is not None: da = da.isel(time=slice(*xlims))
    if not np.issubdtype(da.time.dtype, np.int64): da['time'] = da.time.dt.year.values
    if max_color_lim: da = da.isel(time=slice(None, max_color_lim))
    
    
    if levels is not None:
        colormap_kwargs = dict(levels=levels)
    elif vmax is not None and step is not None:
        levels = create_levels(vmin=vmin, vmax=vmax, step=step)
        colormap_kwargs = dict(levels=levels)
    else:
        colormap_kwargs = dict(robust=True)
    
    cs = da.plot(ax=ax, cmap=cmap, extend=extend, add_colorbar=False, **colormap_kwargs)
    
    # ---> Labelling
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel=ylabel,labelpad=labelpad, font_scale=font_scale,
                                  rotation=rotation)
    fig.suptitle(title, fontsize=constants.PlotConfig.title_size*font_scale, y=0.92)
    ax.set_title(axes_title, fontsize=constants.PlotConfig.title_size*font_scale)
    if xlims is not None: ax.set_xlim(xlims)
 
    # ---> Artist
    if patch: ax.add_artist(Rectangle((max_color_lim, 0), xlims[-1]-max_color_lim,
                                      200, color='grey', alpha=0.2, zorder=-1000))
    
    # ---> Colorbar
    if add_colorbar:
        cax = cax if cax is not None else fig.add_subplot(gs[1])
        cbar = create_colorbar(
            cs, cax=cax, levels=levels, extend=extend, orientation='horizontal',
            font_scale=font_scale, cbar_title=cbar_label, tick_offset=tick_offset,
            cut_ticks=cut_ticks)
    if return_all:
        return fig, gs, ax, cax


def sn_multi_window_in_time(da:xr.DataArray, exp_type:str=None,
                            temp_da:Union[xr.DataArray, xr.Dataset]=None,
                            stable_point_ds:xr.Dataset=None,
                            fig:plt.figure=None, gs=None, ax:plt.Axes=None, cax:plt.Axes=None,
                            figsize:tuple=None, cmap='Blues', extend='neither', max_color_lim:int=None,
                            levels:list=None, vmin=None, vmax=None, step=None, 
                            xlims:tuple=(None,None), font_scale=1.5,
                            cbar_tile:str='', tick_labels=None, add_colorbar=True, cbar_label=None,
                            tick_offset=None, cut_ticks=1, patch=False, hspace=0,
                            title:str=None, axes_title:str=None,rotation=0,
                            ylabel='Window Length\n(Years)', xlabel='Time (Years)',
                            ax2_ylabel = 'Anomaly', add_legend=True, labelpad=100, 
                            bbox_to_anchor=(1, 1.3), stable_year_kwargs=dict(),
                            logginglevel='ERROR', return_all=True):
    '''
    
    '''
    mpl.rcParams.update(mpl.rcParamsDefault)
    utils.change_logging_level(logginglevel)    
    
     
    # ---> Creating plot
    fig = plt.figure(figsize=(plot_kwargs['width'], plot_kwargs['height'])) if fig is None else fig
    gs = gridspec.GridSpec(2,1, height_ratios=[1, 0.1], hspace=plot_kwargs['hspace']) if gs is None else gs
    ax = fig.add_subplot(gs[0]) if ax is None else ax
    
    # ---> Stable Year 
    stable_year_kwargs = {'color':'k', 'linestyle':':', 'linewidth':1.5, **stable_year_kwargs}
    if stable_point_ds: stable_point_ds.time.plot(y='window', ax=ax, **stable_year_kwargs)

    # ---> Plotting colors
    fig, gs, ax, cax = plot_heatmap(da=da, fig=fig, gs=gs, ax=ax, cax=cax,
                 figsize=figsize, cmap=cmap, extend=extend, max_color_lim=max_color_lim,
                 levels=levels, vmin=vmin, vmax=vmax, step=step, 
                 xlims=xlims, font_scale=font_scale,
                 cbar_tile=cbar_tile, tick_labels=tick_labels, add_colorbar=add_colorbar, cbar_label=cbar_label,
                 tick_offset=tick_offset, cut_ticks=cut_ticks, patch=patch, hspace=hspace,
                 title=title, axes_title=axes_title, rotatiogn=rotation,
                 ylabel=ylabel, xlabel=xlabel, labelpad=labelpad)

    # ---> Temperature Anomaly
    if isinstance(temp_da, xr.DataArray):
        ax2 = ax.twinx()
        temp_da = temp_da.isel(time=slice(*xlims))
        if not np.issubdtype(temp_da.time.dtype, np.int64): temp_da['time'] = temp_da.time.dt.year.values
        plot_all_coord_lines(da=temp_da, ax=ax2, fig=fig, exp_type=exp_type, add_legend=add_legend,
                             font_scale=font_scale, bbox_to_anchor=bbox_to_anchor)
        plotting_functions.format_axis(ax2, xlabel=xlabel, ylabel=ax2_ylabel,
                                       font_scale=font_scale, labelpad=labelpad, rotation=rotation)
        ax2.set_title(None)

    if return_all:
        return (fig, [ax, ax2, cax])


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
    
    cbar = format_colorbar(pcolor, gs=gs)
    
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
    leg = ax.legend(ncol=1, frameon=True, facecolor='white', fontsize=18) # , bbox_to_anchor=[1, 0.857]
    leg.set_title('Variable')
    leg.get_title().set_fontsize('18')
    ax.set_xlim(-1, np.max(ds.time.values))
    ax.set_ylim(np.min(ds.window.values), np.max(ds.window.values))
    ax.set_xlabel('Year of Stabilisation', fontsize=18)
    ax.set_ylabel('Window Length (years)', fontsize=18)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)

    
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

def __create_plot_dict(ds:xr.DataArray, dim:str, percentile:float=99, vmin=None, num_steps:int=11, debug=False) -> Dict:
    '''
    Creates a dictionary of all of the levels on dimension
    '''
    dims = list(ds.dims)
    dims_bar_dim = [d for d in dims if d != dim]
    robust_max = ds.reduce(np.nanpercentile,q=percentile, dim=dims_bar_dim).values
    if debug: print(f'{robust_max=}')
    dim_values = ds[dim].values
    
    plot_dict = {}
    for dv, vm in zip(dim_values, robust_max):
        step_size_raw = vm/num_steps
        step_size_actual = np.ceil(step_size_raw)
        vm_actual = step_size_actual*num_steps
        vmin = -vm_actual if vmin is None else vmin
        if debug: print(f'{step_size_raw=} {step_size_actual=} {vmin=} {vm_actual=} {step_size_actual=}')
        plot_dict[dv] = {'levels': np.arange(vmin, vm_actual+step_size_actual, step_size_actual)}
    
    
    return plot_dict


def __make_up_coords(ds):
    made_up_name = '_'
    ds =  ds.expand_dims(made_up_name).assign_coords(_=(made_up_name, ['_']))
    return '_', ds

def map_plot_all_for_coords(ds:xr.DataArray, variable:int, column_coord:str=None, row_coord:str=None,
                            column_title_tag:str='', y=0.89, row_labels:List[str]=None, vmin=None, 
                            plot_dict:Dict=None, cmap='Reds', extend='max', hspace=0.2, one_colorbar=False):

    '''
    Map plot for two coords
    Updated: 27-th March 2023
    '''
    if isinstance(ds, xr.Dataset): ds = ds.to_array().squeeze()
    
    # Yes, this is weird. Looping over rows and cols later. If not colr
    # or row lets just make one up so this all works.
    if column_coord is None: column_coord, ds = __make_up_coords(ds)
    if row_coord is None: row_coord, ds = __make_up_coords(ds)
        
    
    column_coord_values = ds[column_coord].values; row_coord_values = ds[row_coord].values
    row_labels = row_coord_values if row_labels is None else row_labels
    num_cols = len(column_coord_values); num_rows = len(row_coord_values)
    plot_dict = __create_plot_dict(ds, column_coord, vmin=vmin) if not plot_dict else plot_dict

    fig = plt.figure(figsize=(6*num_cols, 4*num_rows))
    gs = gridspec.GridSpec(num_rows+1, num_cols, height_ratios=[0.2]+[1]*num_rows,
                           hspace=hspace, wspace=0.2)

    fig.suptitle(f'{constants.VARIABLE_INFO[variable]["longname"]} Year of Stabilisation', 
                fontsize=25, y=y)

    axes = []; plots = []

    y_axis_kwargs = dict(xy=(-0.05, 0.5), ha='center', va='center', xycoords='axes fraction', 
                         rotation=90, size=18)
    
    for row, rcv in enumerate(row_coord_values):
        for col, ccv in enumerate(column_coord_values):
            ax = fig.add_subplot(gs[row+1, col], projection=ccrs.PlateCarree())
            da = ds.loc[{column_coord:ccv, row_coord:rcv}]
            plot = da.plot(ax=ax, cmap=cmap, levels = plot_dict[ccv]['levels'], 
                           add_colorbar=False, extend=extend)
            
            if not col: ax.annotate(f'{row_labels[row]}', **y_axis_kwargs)
            format_plot(fig, ax); ax.coastlines(); ax.set_title(None)
            axes.append(ax); plots.append(plot)

    for column_coord,ax in zip(column_coord_values, axes[:len(column_coord_values)]):
        ax.set_title(f'{column_coord} {column_title_tag}', fontsize=18)

    for plot_num, plot in enumerate(plots[:len(column_coord_values)]):            
        cax = plt.subplot(gs[0, plot_num] if not one_colorbar else gs[0, :])
        cbar = plt.colorbar(plot, cax=cax, orientation='horizontal')
        cbar.ax.set_title('Year of Stabilisation', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        if one_colorbar: break
            
    for num, ax_ in enumerate(axes):
        plotting_functions.add_figure_label(ax_, f'{chr(97+num)})')
        
    return fig, axes, plots



def map_plot_all_for_coords_2(da:xr.DataArray, dim:str, levels:ArrayLike, ncols=3,
                              fig=None, axes:plt.Axes=None, gs=None,cax=None, add_colorbar=True,
                              projection=ccrs.Robinson, cmap='RdBu_r', extend='both', font_scale=1.4,
                              cbar_title=None, return_all=True, add_label=True, debug=False):
    '''
    Plots rows x colums of the data for the dim coordinate. This can take any projeciton and
    also can take new figure or generate a figure.
    
    seperate_cbar: bool
        A seperate cbar per columns. Only valid currenty for a single row
    '''
    
    projection = projection(central_longitude=int(np.mean(da.lon.values)))
    
    dim_vals =da[dim].values
    nrows = int(np.ceil(len(dim_vals)/ncols))
    fig = plt.figure(figsize=(18*nrows, 10*ncols)) if fig is None else fig
    gs = gridspec.GridSpec(nrows+1, ncols, height_ratios=[1]*nrows + [0.15]) if gs is None else gs
    num_plots = ncols*nrows; num_plots = num_plots-1 if num_plots>2 else num_plots
    axes = ([fig.add_subplot(gs[i], projection=projection) for i in range(0,num_plots)]
            if axes is None else axes)
    colobar_completed = False
    # If levels[0] is not list or array it will be a int/float, then this is just one levels.
    matching_levels_for_all = isinstance(levels[0], (int, float, np.int64, np.float32, np.float64))
    for num,dv in enumerate(dim_vals):
        levels_to_use = levels if matching_levels_for_all else levels[num]
        if debug: print(levels_to_use)
        c = axes[num].contourf(da.lon.values, da.lat.values, da.loc[{dim:dv}].values,
                      transform=ccrs.PlateCarree(), cmap=cmap, levels=levels_to_use, extend=extend)
        axes[num].coastlines()
        axes[num].set_title(dv, fontsize=constants.PlotConfig.title_size*font_scale)
        if add_label: plotting_functions.add_figure_label(axes[num], f'{chr(97+num)})', font_scale=font_scale)
        if add_colorbar and not colobar_completed:
            gs_to_use = gs[nrows, :] if matching_levels_for_all else gs[nrows, num] # One cax, or multiple
            print(gs_to_use)
            cax_to_use = plt.subplot(gs_to_use)if cax is None else cax[num]
            create_colorbar(c, cax=cax_to_use, levels=levels_to_use, extend=extend, orientation='horizontal',
                            font_scale=font_scale, cbar_title=cbar_title)
            colobar_completed = True if matching_levels_for_all else False

    if return_all: return fig, gs, axes

def plot_stable_year_all_models(ds, fig=None, ax=None, linestyle='solid', exp_type=None, add_legend=True, 
                               legend_loc='right', ncol=1, bbox_to_anchor=None, font_scale=1, labelpad=60,
                               xlabel='Year of Stabilisation', xlabelpad=20):
    '''
    Plot the year of stabilisation at each window for different models.

    Args:
        window_stabilization_data (xarray.Dataset): Data containing the year of stabilisation at each window for different models.
        fig (matplotlib.figure.Figure, optional): Figure to plot on. If None, create a new figure. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, create a new axes. Defaults to None.
        linestyle (str, optional): Linestyle of the plotted lines. Defaults to 'solid'.
        exp_type (str, optional): Experiment type. Defaults to None.
        add_legend (bool, optional): Whether to add a legend. Defaults to True.
        legend_loc (str, optional): Location of the legend. Defaults to 'right'.
        ncol (int or str, optional): Number of columns for the legend. If 'coords', use the number of models. Defaults to 1.
        bbox_to_anchor (tuple, optional): Bounding box for the legend. Defaults to None.
        font_scale (float, optional): Scaling factor for the font size. Defaults to 1.
        labelpad (int, optional): Padding for the axis labels. Defaults to 60.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Year of Stabilisation'.
        xlabelpad (int, optional): Padding for the x-axis label. Defaults to 20.

    Returns:
        matplotlib.figure.Figure: The figure.
        matplotlib.axes.Axes: The axes.
    '''
    plt.style.use('seaborn-darkgrid')
    if not fig: fig = plt.figure(figsize=(10, 8))
    if not ax: ax = fig.add_subplot(111)
    if exp_type:
        information_profile = MODEL_PROFILES[exp_type]
        models = [model for model in list(information_profile) if model in ds.model.values]
    else:
        colors = constants.RANDOM_COLOR_LIST; models = ds.model.values
      
    ds.median(dim='model').time.plot(ax=ax,y='window', label='Median', color='k', linewidth=4, linestyle='solid') 
    
    for num, model in enumerate(models):
        if not exp_type: kwargs_to_use = dict(color = colors[num], label=model)
        else:
            ECS = information_profile[model]['ECS']
            kwargs_to_use = dict(color=information_profile[model]['color'], label=f'{model} ({ECS=}K)', linestyle=information_profile[model]['linestyle'])
            
        da = ds.sel(model=model).time.plot(ax=ax,y='window', linewidth=3, alpha=0.8, **kwargs_to_use)
        
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
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel='Window Length\n(Years)',
                                   font_scale=font_scale, xlabelpad=xlabelpad, labelpad=labelpad)
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