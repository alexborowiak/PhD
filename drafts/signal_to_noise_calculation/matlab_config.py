import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt


def matlab_config():
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)


    params = {'axes.labelsize' : 'x-large',
             'legend.fontsize': 'x-large',
              'axes.titlesize': 'x-large',
             }

    pylab.rcParams.update(params)