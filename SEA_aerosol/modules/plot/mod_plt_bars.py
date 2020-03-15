#import libraries
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_bars(plot_data, plot_err, colors, xticks, xlabel, ylabel, title, fname):

    bar_width = 0.8
    opacity = 0.8
    index = np.arange(len(plot_data))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(index, plot_data, bar_width, alpha=opacity, color=colors)
    plt.errorbar(index, plot_data, yerr=plot_err, elinewidth=0.5,
                 ecolor='black', fmt='none', alpha=opacity)

    plt.xticks(index, xticks, fontsize=5, rotation=60)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
