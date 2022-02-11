import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from matplotlib import ticker as mticker
import matplotlib.gridspec as gridspec

def temperature_vs_sn_plot(ax, sn, temp, temp_highlight=None, sn_highlight=None):
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
    ax.plot(sn.time.values,sn.values, label = 'Unstable', c = 'tomato')
    if isinstance(sn_highlight, xr.DataArray):
        ax.plot(sn_highlight.time.values,sn_highlight.values, label = 'Stable', c = 'darkred')


    c0 = plt.gca().lines[0].get_color()
    ax.tick_params(axis = 'y', labelcolor = c0)
    ax.set_ylabel(r'$\dfrac{Signal}{Noise}$', fontsize = 12, color = c0, rotation = 0, labelpad = 55);

    leg = ax.legend(ncol = 1, fontsize = 12, bbox_to_anchor = [-0.05, 1])
    leg.set_title('S/N')
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
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

    
    ax.set_xlabel('Year', fontsize = 12);
    ax.set_xlim(temp.time.values[0], temp.time.values[-1])
    ax2.set_xlim(temp.time.values[0], temp.time.values[-1])
    
    return [ax, ax2]


def sn_plot_kwargs(kwargs):
    # The default plot kwargs
    plot_kwargs = dict(height = 15, width = 7, hspace=0.3, vmin = -8, vmax = 8, step = 2, 
                      cmap = 'RdBu_r', line_color = 'limegreen', line_alpha = 0.5, 
                      cbar_label = 'S/N',cbartick_offset = 0,title='', label_size = 12, extend='both', 
                      xlowerlim = None, xupperlim = None)
    
    # Upading any of the kwargs with custom values
    for key, value in kwargs.items():
        plot_kwargs[key] = value

    
    
    plot_kwargs['levels'] = np.arange(plot_kwargs['vmin'], # Min
                       plot_kwargs['vmax'] + plot_kwargs['step'], # Max
                       plot_kwargs['step']) # Step
    
    plot_kwargs['cmap'] = plt.cm.get_cmap(plot_kwargs['cmap'], len(plot_kwargs['levels']) + 1)


        
    if 'cbar_xticklabels' in kwargs.keys():
        plot_kwargs['cbar_xticklabels'] = kwargs['cbar_xticklabels']
    else:
        plot_kwargs['cbar_xticklabels'] = plot_kwargs['levels']
        
    
    if 'cbar_ticks' in kwargs.keys():
        cbar_ticks = kwargs['cbar_ticks'] + plot_kwargs['cbartick_offset']
    else:
        cbar_ticks = plot_kwargs['levels']+ plot_kwargs['cbartick_offset']
    plot_kwargs['cbar_ticks'] = cbar_ticks[:len(plot_kwargs['cbar_xticklabels'])]
    
    print(plot_kwargs)
    return plot_kwargs


def sn_multi_window_in_time(unstable_sn_multi_window_da: xr.DataArray, 
                            stable_sn_multi_window_da: xr.DataArray,
                            abrupt_anom_smean: xr.DataArray, **kwargs):
    
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
    
    # Upadting the plot kwargs
    plot_kwargs = sn_plot_kwargs(kwargs)
    

    fig = plt.figure(figsize = (plot_kwargs['height'], plot_kwargs['width']))
    gs = gridspec.GridSpec(2,1, height_ratios = [1, 0.1], hspace = plot_kwargs['hspace'])
    

    ### Window
    ax1 = fig.add_subplot(gs[0])

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

    ### Temperature Anomaly
    ax2 = ax1.twinx()
    c1 = plot_kwargs['line_color']
    ax2.plot(abrupt_anom_smean.time.dt.year.values, abrupt_anom_smean.values, color=c1,
             alpha= plot_kwargs['line_alpha'], zorder=1000)

    ax2.spines['right'].set_color(c1)
    ax2.tick_params(axis='y', colors=c1)
    ax2.set_ylabel(r'Global Mean Temperature Anomaly ($^\circ$C)', color=c1, size =12);


    ### General
    ax1.set_xlabel('Time (years)', size =plot_kwargs['label_size'])
    ax1.xaxis.set_minor_locator(mticker.MultipleLocator(50))
    ax1.set_title(plot_kwargs['title'])

    ax1.set_xlim(plot_kwargs['xlowerlim'], plot_kwargs['xupperlim'])
    return (fig, ax1, ax2, ax3, cbar)
