
import os
import sys
from importlib import reload
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.join(os.getcwd(), 'Documents', 'PhD'))
import constants
sys.path.append(constants.MODULE_DIR)
import listXarray as lxr
from listXarray import listXarray

import utils
logger = utils.get_notebook_logger()


def plot_histogram(da: listXarray, bins: Union[List[float], np.ndarray],
                   step: int, zec_vals: listXarray = None, bar_label: str = '', 
                   line_label: str = None, bar_color: str = 'blue', line_color: str = 'red', ylabel_right:bool=False, 
                   fig: Optional[plt.Figure] = None, axes: Optional[List[plt.Axes]] = None, xlim:float=None, ylim:float=0.3,
                   add_legend: bool = False, xlabel:str=None, title_loc: str = 'regular', label_ensemble:bool=False, return_fig_axes=False,
                  logginglevel='ERROR') -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot histograms for the given DataArray and models.
    ...
    """
    utils.change_logginglevel(logginglevel)

    logger.debug(f'locals\n{locals()}')
    
    
    models = da.model.values if isinstance(da, (xr.Dataset, xr.DataArray)) else da.refkeys

    if fig is None and axes is None:
        logger.info(' - Creating new figure')
        fig = plt.figure(figsize=(15, 8))
        ncols = 3
        nrows = int(np.ceil(len(models) / ncols))
        gs = gridspec.GridSpec(nrows, ncols, hspace=0.45)
        axes = [fig.add_subplot(gs[i]) for i in range(nrows * ncols)]

    for num, (ax, model) in enumerate(zip(axes, models)):
        vals = da.sel(model=model) if isinstance(da, (xr.Dataset, xr.DataArray)) else da[model]
        logger.debug(f' - {bins=}')
        logger.debug(vals)
        logger.debug(f' - {type(vals)}')
        hist, edges = np.histogram(vals.values, bins=bins)
        
        ax.bar(edges[:-1], hist / len(vals.time.values), alpha=0.5, width=step, label=bar_label, color=bar_color)
        
        if zec_vals is not None:
            logger.debug(f'zec_vals type is {type(zec_vals)}')
            if isinstance(zec_vals, listXarray):
                logger.info('Type is listXarray')
                ensembles = zec_vals[model].ensemble.values
                values = zec_vals[model].values
                #label = line_label if ensemble == zec_vals[model].ensemble.values[0] else None
            elif isinstance(zec_vals, pd.DataFrame):
                logger.info('Type is Pandas')
                index_values = [ind for ind in zec_vals.index.values if model in ind]
                values = zec_vals.loc[index_values].values
                ensembles = [ind.split('_')[-1] for ind in index_values]
                
            for ensemble, zec_val in zip(ensembles, values):
                if line_label is not None: label=line_label
                else: label=ensemble
                    
                ax.axvline(zec_val, 0, 1, color=line_color, label=label, linewidth=2)
                if label_ensemble:
                    ax.annotate(ensemble, xy=(zec_val+0.01 if zec_val>0 else zec_val-0.01, 0.19),
                                size=6, va='top', ha='center', color=line_color, clip_on=False, rotation=90)

        ax.set_title(model if title_loc == 'regular' else '')
        ax.annotate(model, xy=(0.02, 0.05), fontsize=14, xycoords='axes fraction' if title_loc != 'regular' else 'figure fraction')
        
        
        if xlabel: ax.set_xlabel(xlabel, fontsize=14)
        if num == np.ceil(len(models)/2)-1:
            ax.set_ylabel('Relative Frequency', fontsize=14)
            # Remove the default y-axis label on the left
            if ylabel_right: ax.yaxis.set_label_coords(1.15, 0.5)
        ax.set_ylim(0, ylim)
        ax.set_yticks(np.arange(0.05, ylim, 0.05))
        if not xlim:
            logger.info('Auto xlim creation')
            xlim = np.max([np.max(np.concatenate([ds.values for ds in da.to_list()])), np.max(np.abs(bins))]) * 1.2
        logger.info(f'{xlim=}')
        [ax.set_xlim(-xlim, xlim) for ax in axes]
        ax.axvline(color="grey", alpha=0.2, zorder=-100, linestyle='--')
        
        if add_legend and ax is axes[0]: ax.legend(loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.2, color='grey')

    if return_fig_axes: return fig, axes