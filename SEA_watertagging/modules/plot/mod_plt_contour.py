
# import libraries
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds, latbounds, varname, var_unit, title, fname, opt=0, **kwargs):

    arranges = {1: [1, 1], 2: [2, 1], 3: [3, 1], 4: [2, 2], 5: [3, 2],
                6: [2, 4], 8: [2, 4], 9: [3, 3], 10: [3, 4], 12: [3, 4], 24: [8, 3]}
    nfigs = len(plot_data)
    # print(nfigs)

    if nfigs not in arranges:
        print('plot_2Dcontour: Error! Too many Sub-figures, the maximum number is 9!')
        return -1

    if opt == 1:
        if 'sig_test' not in kwargs:
            print('plot_2Dcontour: Warning! sig_test is missing, significance level is skipped!')
            opt = 1
        else:
            plot_test = kwargs['sig_test']

    plt.clf()
    figsize = (8, 6)
    if nfigs == 24:
        figsize = (10, 16)
    # if nfigs == 12:
    #     figsize = (10, 9)
    fig = plt.figure(figsize=figsize)
    # axes = axes.flatten()
    # print(arranges[nfigs][0],arranges[nfigs][1])
    # irow = 0
    # icol = 0

    for idx in range(len(plot_data)):

        ax = fig.add_subplot(arranges[nfigs][0], arranges[nfigs][1], idx+1)
        ax.set_title(legends[idx], fontsize=5, pad=5)

        # create basemap
        map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                      llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
        map.drawcoastlines(linewidth=0.3)
        map.drawcountries()

        # draw lat/lon lines
        parallels = np.arange(latbounds[0], latbounds[1], 20)
        meridians = np.arange(lonbounds[0], lonbounds[1], 20)
        map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
        map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)

        lons = plot_lons[idx]
        lats = plot_lats[idx]

        mlons, mlats = np.meshgrid(lons, lats)
        x, y = map(mlons, mlats)

        # plot the contour
        cs = map.contourf(x, y, plot_data[idx], clevs, cmap=colormap, alpha=0.9, extend="both")

        # plot the significance level, if needed
        if (opt == 1):
            levels = [0., 2.01, plot_test[idx].max()]
            csm = ax.contourf(x, y, plot_test[idx], levels=levels,
                              colors='none', hatches=['', '....'], alpha=0)

        # set x/y tick label size
        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelsize=5)

        if 'xlabel' in kwargs:
            xlabel = kwargs['kwxlabelargs']

            if (len(plot_data)-idx) <= 2:
                if type(xlabel) == str:
                    ax.set_xlabel(xlabel, fontsize=5)
                else:
                    ax.set_xlabel(xlabel[int(idx % (arranges[nfigs][1]))], fontsize=5)

        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
            if (idx % (arranges[nfigs][1]) == 0):
                if type(ylabel) == str:
                    ax.set_ylabel(ylabel, fontsize=5, labelpad=0.7)
                else:
                    ax.set_ylabel(ylabel[int(idx/(arranges[nfigs][1]))], fontsize=5, labelpad=13.)

    # add colorbar
    if (arranges[nfigs][0] > arranges[nfigs][1]):
        fig.subplots_adjust(right=0.7, wspace=0.2, hspace=0.2)
        cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=4)
        cbar.set_label(varname+' ['+var_unit+']', fontsize=5, labelpad=0.7)
    else:
        fig.subplots_adjust(bottom=0.23, wspace=0.2, hspace=0.2)
        cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
        cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=4)
        cbar.set_label(varname+' ['+var_unit+']', fontsize=5, labelpad=-0.7)

    # add title
    if opt == 1:
        plt.savefig(fname+'_wtsig.png', bbox_inches='tight', dpi=600)
    else:
        plt.savefig(fname+'.png', bbox_inches='tight', dpi=600)

    plt.suptitle(title, fontsize=7, y=0.95)

    # save figure
    if opt == 1:
        plt.savefig(fname+'_wtsig.pdf', bbox_inches='tight', dpi=600)
    else:
        plt.savefig(fname+'.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)

# plot 2D contour with vectors


def plot_2Dvectorcontour(plot_data, plot_u, plot_v, plot_lons, plot_lats, colormap, clevs, legends, lonbounds, latbounds, varname, var_unit, vector_unit, vector_length, title, fname, opt=0, **kwargs):

    arranges = {1: [1, 1], 2: [2, 1], 3: [3, 1], 4: [2, 2], 6: [3, 2], 8: [2, 4], 9: [3, 3]}
    nfigs = len(plot_data)
    # print(nfigs)

    if nfigs not in arranges:
        print('plot_2Dcontour: Error! Too many Sub-figures, the maximum number is 9!')
        return -1

    if opt == 1:
        if 'sig_test' not in kwargs:
            print('plot_2Dcontour: Warning! sig_test is missing, significance level is skipped!')
            opt = 0
        else:
            plot_test = kwargs['sig_test']

        if 'sig_thres' not in kwargs:
            print('plot_2Dcontour: Warning! sig_thres is not given, default value is 2.01 (for t-test)!')
            sig_thres = 2.01
        else:
            sig_thres = kwargs['sig_thres']

    plt.clf()
    fig = plt.figure()
    # axes = axes.flatten()
    # print(arranges[nfigs][0],arranges[nfigs][1])
    # irow = 0
    # icol = 0

    for idx in range(len(plot_data)):

        ax = fig.add_subplot(arranges[nfigs][0], arranges[nfigs][1], idx+1)
        ax.set_title(legends[idx], fontsize=5, pad=2)

        # create basemap
        map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                      llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
        map.drawcoastlines(linewidth=0.3)
        map.drawcountries()

        # draw lat/lon lines
        parallels = np.arange(latbounds[0], latbounds[1], 20)
        meridians = np.arange(lonbounds[0], lonbounds[1], 20)
        map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
        map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)

        lons = plot_lons[idx]
        lats = plot_lats[idx]

        mlons, mlats = np.meshgrid(lons, lats)
        x, y = map(mlons, mlats)

        # plot the contour
        cs = map.contourf(x, y, plot_data[idx], clevs, cmap=colormap, alpha=0.9, extend="both")
        # cq = map.quiver(x, y, plot_u[idx], plot_v[idx])
        vector_gap = int(4./(lats[1]-lats[0]))
        # vector_length = 3. * np.mean(np.sqrt(np.power(plot_u[idx], 2)+np.power(plot_v[idx], 2)))
        cq = map.quiver(x[::vector_gap, ::vector_gap], y[::vector_gap, ::vector_gap], plot_u[idx][::vector_gap,
                                                                                                  ::vector_gap], plot_v[idx][::vector_gap, ::vector_gap], scale=vector_length * 20, scale_units='width')
        # qk = ax.quiverkey(cq, 0.9, 0.9, vector_length, str(vector_length)+' '+vector_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})

        # plot the significance level, if needed
        if (opt == 1):
            temptest = plot_test[idx]
            temptest = temptest[~np.isnan(temptest)]
            levels = [0., sig_thres, temptest.max()]
            csm = ax.contourf(x, y, plot_test[idx], levels=levels,
                              colors='none', hatches=['', '....'], alpha=0)

        # set x/y tick label size
        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelsize=5)

    # add colorbar
    if (arranges[nfigs][0] > arranges[nfigs][1]):
        fig.subplots_adjust(right=0.7, wspace=0.2, hspace=0.2)
        cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=4)
        cbar.set_label(varname+' ['+var_unit+']', fontsize=5, labelpad=0.7)
    else:
        fig.subplots_adjust(bottom=0.2, wspace=0.15, hspace=0.1)
        cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
        cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=4)
        cbar.set_label(varname+' ['+var_unit+']', fontsize=5, labelpad=-0.5)

    # add vector legend
    vector_length = round(vector_length, 2)
    qk = plt.quiverkey(cq, 0.7, 0.9, vector_length, str(vector_length)+' '+vector_unit,
                       labelpos='E', coordinates='figure', fontproperties={'size': 5})

    # add title
    plt.suptitle(title, fontsize=7, y=0.95)

    # save figure
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
