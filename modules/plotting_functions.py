import numpy as np
from typing import List

import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.append('../')

import utils

logger = utils.get_notebook_logger()
import constants
from constants import PlotConfig

def create_levels(vmax:float, vmin:float=None, step:float=1)->np.ndarray:
    '''
    Ensures that all instances of creating levels using vmax + step as the max.
    '''
    vmin = -vmax if vmin is None else vmin
    return np.arange(vmin, vmax + step, step)

def add_figure_label(ax: plt.Axes, label:str, font_scale:int=1, x:float=0.01, y:float=1.05):
    ax.annotate(label, xy = (x,y), xycoords = 'axes fraction', size=PlotConfig.label_size*font_scale)

def format_axis(ax: plt.Axes, title:str=None, xlabel:str=None, ylabel:str=None, invisible_spines=None, 
               font_scale=1, rotation=0, labelpad=100, xlabelpad=10, grid:bool=True):
    '''Formatting with no top and right axis spines and correct tick size.'''
    if xlabel: ax.set_xlabel(xlabel, fontsize=PlotConfig.label_size*font_scale, ha='center', va='center',
                            labelpad=xlabelpad)
    if ylabel: ax.set_ylabel(ylabel, rotation=rotation, labelpad=labelpad*font_scale,
                             fontsize=PlotConfig.label_size*font_scale, ha='center', va='center')
    if title: ax.set_title(title, fontsize=PlotConfig.title_size*font_scale)
    ax.tick_params(axis='x', labelsize=PlotConfig.tick_size*font_scale)
    ax.tick_params(axis='y', labelsize=PlotConfig.tick_size*font_scale)
    if invisible_spines: [ax.spines[spine].set_visible(False) for spine in invisible_spines]
    if grid: ax.grid(True, alpha=0.5, c='grey', linestyle='--')
    return ax


def match_ticks(ax1, ax2, master_axis='left'):
    """
    Match the tick values and limits of two matplotlib axes.

    This function is used to synchronize the y-axis ticks and limits of two
    subplots in a Matplotlib figure.

    Parameters:
    ax1 (matplotlib.axes._subplots.AxesSubplot): The first subplot.
    ax2 (matplotlib.axes._subplots.AxesSubplot): The second subplot.
    master_axis (str): The axis to be used as the master for synchronization.
        Can be 'left' or 'right'.

    Returns:
    None

    Raises:
    TypeError: If master_axis is not 'left' or 'right'.
    """

    # Get the tick values of both axes
    ax1_yticks = ax1.get_yticks()
    ax2_yticks = ax2.get_yticks()

    # Set the y-axis limits of both axes to the first and last ticks
    ax1.set_ylim(np.take(ax1_yticks, [0, -1]))
    ax2.set_ylim(np.take(ax2_yticks, [0, -1]))

    if master_axis == 'left':
        # If left is the master, synchronize right axis with left
        ax2_yticks = np.linspace(*np.take(ax2_yticks, [0, -1]), len(ax1_yticks))
        ax2.set_yticks(ax2_yticks)
        ax2.set_ylim(np.take(ax2_yticks, [0, -1]))

    elif master_axis == 'right':
        # If right is the master, synchronize left axis with right
        ax1_yticks = np.linspace(*np.take(ax1_yticks, [0, -1]), len(ax2_yticks))
        ax1.set_yticks(ax1_yticks)
        ax1.set_ylim(np.take(ax1_yticks, [0, -1]))

    else:
        raise TypeError(f'master_axis must be either "right" or "left". Entered: {master_axis}')

    return ax1, ax2

def clip_axis_ends(ax):
    """
    Clip the first and last y-axis tick labels on a Matplotlib Axes object.

    Parameters:
    ax (matplotlib.axes._subplots.Axes): The Axes object for which you want to modify the y-axis tick labels.

    Returns:
    None

    This function retrieves the current y-axis tick labels from the provided Axes object and removes the text from
    the first and last tick labels, effectively clipping the ends of the y-axis.

    Example:
    import matplotlib.pyplot as plt

    # Create a simple plot
    plt.plot([1, 2, 3, 4, 5], [10, 20, 25, 30, 35])

    # Get the current Axes object
    ax = plt.gca()

    # Call clip_axis_ends to clip the y-axis tick labels
    clip_axis_ends(ax)

    # Display the modified plot
    plt.show()
    """
    # Get the current y-axis tick labels
    labels = [label.get_text() for label in ax.get_yticklabels()]

    # Clip the first and last y-axis tick labels by setting them to empty strings
    labels[0] = ''
    labels[-1] = ''

    # Set the modified y-axis tick labels back to the Axes object
    ax.set_yticklabels(labels)

def fig_formatter(height_ratios: List[float] , width_ratios: List[float],  hspace:float = 0.4, wspace:float = 0.2):
    
    height = np.sum(height_ratios)
    width = np.sum(width_ratios)
    num_rows = len(height_ratios)
    num_cols = len(width_ratios)
    
    fig  = plt.figure(figsize = (10*width, 5*height)) 
    gs = gridspec.GridSpec(num_rows ,num_cols, hspace=hspace, 
                           wspace=wspace, height_ratios=height_ratios, width_ratios=width_ratios)
    return fig, gs



def create_discrete_cmap(cmap, number_divisions:int=None, levels=None, vmax=None, vmin=None, step=1,
                         add_white:bool=False, white_loc='start', clip_ends:int=0):
    '''
    Creates a discrete color map of cmap with number_divisions
    '''
    
    if levels is not None: number_divisions = len(levels)
    elif vmax is not None: number_divisions = len(create_levels(vmax, vmin, step))
                
    color_array = plt.cm.get_cmap(cmap, number_divisions+clip_ends)(np.arange(number_divisions+clip_ends)) 

    if add_white:
        if white_loc == 'start':
            white = [1,1,1,1]
            color_array[0] = white
        elif white_loc == 'middle':
            upper_mid = np.ceil(len(color_array)/2)
            lower_mid = np.floor(len(color_array)/2)

            white = [1,1,1,1]

            color_array[int(upper_mid)] = white
            color_array[int(lower_mid)] = white

            # This must also be set to white. Not quite sure of the reasoning behind this. 
            color_array[int(lower_mid) - 1] = white
        
    cmap = mpl.colors.ListedColormap(color_array)
    
    return cmap


def create_colorbar(plot, cax, levels, tick_offset=None, cut_ticks=1, round_level=2,
                    font_scale=1, cbar_title='', orientation='horizontal', logginglevel='ERROR', **kwargs):
    """
    Create and customize a colorbar for a given plot.

    Parameters:
        plot: matplotlib plot object
            The plot that the colorbar is associated with.
        cax: matplotlib axes object
            The colorbar axes.
        levels: array-like
            The levels used in the plot.
        tick_offset: str, optional
            Offset method for ticks ('center' or None). Default is None.
        cut_ticks: int, optional
            Frequency of ticks to cut for better visualization. Default is 1.
        round_level: int, optional
            Number of decimal places to round tick labels to. Default is 2.
        font_scale: float, optional
            Scaling factor for font size. Default is 1.
        cbar_title: str, optional
            Title for the colorbar. Default is an empty string.
        orientation: str, optional
            Orientation of the colorbar ('horizontal' or 'vertical'). Default is 'horizontal'.
        **kwargs:
            Additional keyword arguments for colorbar customization.

    Returns:
        cbar: matplotlib colorbar object
            The customized colorbar.
    """
    
    utils.change_logginglevel(logginglevel)
    logger.info(utils.function_name())
    logger.info(f'**{__file__}')
    logger.debug(locals())
    # Create the colorbar with specified orientation and other keyword arguments
    cbar = plt.colorbar(plot, cax=cax, orientation=orientation, **kwargs) # , ticks=levels
    # cbar.ax.tick_params(axis='both', which='both', labelsize=tick_size)
    
    # Calculate tick locations and labels based on tick_offset and cut_ticks settings
    tick_locations = levels
    tick_labels = levels
    logger.debug(f'{tick_labels=}\n{tick_locations=}')
    if tick_offset == 'center':
        tick_locations = levels[:-1] + np.diff(levels) / 2
        tick_labels = tick_labels[:-1]
    
    if cut_ticks > 1:
        tick_locations = tick_locations[::cut_ticks]
        tick_labels = tick_labels[::cut_ticks]
    
    logger.debug(f'{tick_labels=}\n{tick_locations=}')
    
    # Set tick locations and labels on the colorbar
    cbar.set_ticks(tick_locations)
    tick_labels = np.round(tick_labels, round_level)
    
    logger.info(f'{tick_labels=}')
    
    # Customize colorbar based on orientation
    if orientation == 'horizontal':
        cbar.ax.xaxis.set_ticks(tick_locations, minor=True)
        cbar.ax.set_xticklabels(tick_labels, fontsize=14*font_scale)
        cbar.ax.set_title(cbar_title, size=18 * font_scale)

    else:
        cbar.ax.set_yticks(tick_locations)
        cbar.ax.set_yticklabels(tick_labels, fontsize=10*font_scale, rotation=90)
        cbar.ax.set_ylabel(cbar_title, size=12 * font_scale, rotation=0, labelpad=30)
        
    return cbar



