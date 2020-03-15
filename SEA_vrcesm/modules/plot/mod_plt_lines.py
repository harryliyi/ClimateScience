#import libraries
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
plt.switch_backend('agg')


def plot_lines(x, plot_data, colors, line_types, legends, xlabel, ylabel, title, fname, **kwargs):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if 'ylog' in kwargs:
        ylog = kwargs['ylog']
    else:
        ylog = False

    if 'xlog' in kwargs:
        xlog = kwargs['xlog']
        xticks = kwargs['xticks']
    else:
        xlog = False

    if 'multix' in kwargs:
        multix = kwargs['multix']
    else:
        multix = False

    for idx in range(len(plot_data)):
        if multix:
            plt.plot(x[idx], plot_data[idx], color=colors[idx], marker='o', markersize=1,
                     linestyle=line_types[idx], linewidth=1.5, label=legends[idx])
        else:
            plt.plot(x, plot_data[idx], color=colors[idx], marker='o', markersize=1,
                     linestyle=line_types[idx], linewidth=1.5, label=legends[idx])
        if 'yerr' in kwargs:
            plot_yerr = kwargs['yerr']
            if multix:
                plt.errorbar(x[idx], plot_data[idx], yerr=plot_yerr[idx], fmt='o',
                             markersize=1, elinewidth=1, ecolor='black')
            else:
                plt.errorbar(x, plot_data[idx], yerr=plot_yerr[idx], fmt='o',
                             markersize=1, elinewidth=1, ecolor='black')

    plt.legend(handlelength=4, fontsize=5)
    if ('xticks' in kwargs)and('xticknames' in kwargs):
        plt.xticks(kwargs['xticks'], kwargs['xticknames'], fontsize=6)
    elif ('xticks' in kwargs):
        plt.xticks(kwargs['xticks'], fontsize=6)
    else:
        plt.xticks(fontsize=6)
    if ('yticks' in kwargs)and('yticknames' in kwargs):
        plt.yticks(kwargs['yticks'], kwargs['yticknames'], fontsize=6)
    elif ('yticks' in kwargs):
        plt.yticks(kwargs['yticks'], fontsize=6)
    else:
        plt.yticks(fontsize=6)

    if ylog:
        ax.set_yscale('log')

    if xlog:
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=5)
        ax.get_yaxis().set_major_formatter(tick.ScalarFormatter())

    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)

    plt.savefig(fname, bbox_inches='tight')

    plt.close(fig)
