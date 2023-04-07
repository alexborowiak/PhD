import numpy as np
from typing import List

import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.append('../')

import constants
from constants import PlotConfig

def create_levels(vmax:float, vmin:float=None, step:float=1)->np.ndarray:
    '''
    Ensures that all instances of creating levels using vmax + step as the max.
    '''
    vmin = -vmax if vmin is None else vmin
    return np.arange(vmin, vmax + step, step)

def add_figure_label(ax: plt.Axes, label:str, font_scale:int=1):
    ax.annotate(label, xy = (0.01,1.05), xycoords = 'axes fraction', size=PlotConfig.label_size*font_scale)

def format_axis(ax: plt.Axes, title:str=None, xlabel:str=None, ylabel:str=None, invisible_spines=None, 
               font_scale=1, rotation=0, labelpad=100, xlabelpad=10):
    '''Formatting with no top and right axis spines and correct tick size.'''
    if xlabel: ax.set_xlabel(xlabel, rotation=rotation, fontsize=PlotConfig.label_size*font_scale, ha='center', va='center',
                            labelpad=xlabelpad)
    if ylabel: ax.set_ylabel(ylabel, rotation=rotation, labelpad=labelpad*font_scale,
                             fontsize=PlotConfig.label_size*font_scale, ha='center', va='center')
    if title: ax.set_title(title, fontsize=PlotConfig.title_size*font_scale)
    ax.tick_params(axis='x', labelsize=PlotConfig.tick_size*font_scale)
    ax.tick_params(axis='y', labelsize=PlotConfig.tick_size*font_scale)
    if invisible_spines: [ax.spines[spine].set_visible(False) for spine in invisible_spines]
    return ax
    
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
    
    if levels is not None:
        number_divisions = len(levels)
    elif vmax is not None:
        number_divisions = len(create_levels(vmax, vmin, step))
                
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


