

# This function takes in an xarray file that is of dimension lat, lon and time. This plot will animate it 
# whilst using the colormap RdBu

def set_animimate_plot(d1, name = 'file'):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    import matplotlib.animation as animation
    from matplotlib import animation, rc
    from IPython.display import HTML

    x,y = np.meshgrid(d1.lon.values, d1.lat.values)
    z = d1.isel(time = 0).values

    fig = plt.figure(figsize=(16, 8),facecolor='white')
    gs = gridspec.GridSpec(1, 1)

    #########################
    ax = plt.subplot(gs[0])
    quad1 = ax.pcolormesh(x,y,z,shading='gouraud',cmap = 'RdBu')
    cb = fig.colorbar(quad1,ax=ax)

    def animate(time):
        z = d1.isel(time = time).values
        quad1.set_array(z.ravel())
        ax.set_title(str(time))
        return quad1

    gs.tight_layout(fig)

    anim = animation.FuncAnimation(fig,animate,frames=len(d1.time.values),
                                   interval=200,blit=False,repeat=True)
    
    anim.save(name + '.gif', writer='imagemagick')
    plt.close(fig)
    
    
def unset_animate_plot(d1,time_frac = 1, cmap = 'RdBu', name = 'file'):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    import matplotlib.animation as animation
    from matplotlib import animation, rc
    from IPython.display import HTML
    
    # These are the frames that are going to actually get used. This enables every second, third ect.
    # time to be used rather than just using all of the time steps which can be too much. 
    used_frames = np.arange(0, len(d1.time.values), time_frac)
    datetime = d1.time.values

    x,y = np.meshgrid(d1.lon.values, d1.lat.values)
    
    # If the figure (and the axis) is not being specified, then
    # we need to create out own figure.
#     if not fig:
    z = d1.isel(time = 0).values
    fig = plt.figure(figsize = (16, 8),facecolor = 'white')
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])
    quad1 = ax.pcolormesh(x,y,z,shading = 'gouraud',cmap = cmap)
    cb = fig.colorbar(quad1, ax = ax)

    def animate(frame):
        time = used_frames[frame]
        time_val = datetime[time]
        z = d1.isel(time = time).values
        quad1.set_array(z.ravel())
        ax.set_title('{}\nFrame {} Time {}'.format(name,str(frame),time_val), fontsize = 15)
        return quad1
    
    anim = animation.FuncAnimation(fig,animate,frames = len(used_frames),
                                   interval=200,blit=False,repeat=True)
    
    anim.save(name + '.gif', writer = 'imagemagick')
    plt.close(fig)