# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import netCDF4 as nc
import datetime as datetime
from mpl_toolkits.basemap import Basemap
import pandas as pd
plt.switch_backend('agg')


# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = "/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/"


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/sst/'

# set up variable names and file name
varname = 'SST_cpl'
var_longname = 'Sea Surface Temperature'
varstr = 'sst'
var_unit = r'$^{\circ}C$'


# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-30, 30]
lonbounds = [30, 300]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'mon'

# define pressure level
plevel = 500

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# define ENSO yyears
years_elweak = [1980, 1983, 1987, 1988, 1992, 1995, 1998, 2003, 2005]
years_elmod = [1983, 1987, 1988, 1992, 1998, 2003]
years_laweak = [1984, 1985, 1989, 1996, 1999, 2000, 2001]
years_lamod = [1989, 1999, 2000]

years_elweakpre = [iyear-1 for iyear in years_elweak]
years_laweakpre = [iyear-1 for iyear in years_laweak]

############################################################################
# define functions
############################################################################


def cal_diff(var1, var2, std1, std2, n1, n2):
    res = var1-var2
    SE = np.sqrt((std1**2/n1) + (std2**2/n2))
    res_sig = res/SE

    return res, res_sig


def plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs1, clevs2, legends, lonbounds, latbounds, varname, var_unit, title, fname, opt=0, **kwargs):

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
        if idx < len(plot_data)-1:
            cs1 = map.contourf(x, y, plot_data[idx], clevs1, cmap=colormap, alpha=0.9, extend="both")
        else:
            cs2 = map.contourf(x, y, plot_data[idx], clevs2, cmap=colormap, alpha=0.9, extend="both")

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
    fig.subplots_adjust(right=0.83, wspace=0.2, hspace=0.2)
    cbar_ax1 = fig.add_axes([0.85, 0.5, 0.01, 0.3])
    cbar1 = fig.colorbar(cs1, cax=cbar_ax1, orientation='vertical')
    cbar1.ax.tick_params(labelsize=4)
    cbar1.set_label(varname+' ['+var_unit+']', fontsize=5, labelpad=0.7)

    cbar_ax2 = fig.add_axes([0.85, 0.13, 0.01, 0.2])
    cbar2 = fig.colorbar(cs2, cax=cbar_ax2, orientation='vertical')
    cbar2.ax.tick_params(labelsize=4)
    cbar2.set_label(varname+' ['+var_unit+']', fontsize=5, labelpad=0.7)

    # add title
    plt.savefig(fname+'.png', bbox_inches='tight', dpi=600)
    plt.suptitle(title, fontsize=7, y=0.95)

    # save figure
    plt.savefig(fname+'.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)


############################################################################
# read data
############################################################################

# read vrcesm

print('Reading CESM data...')

fdir = '/scratch/d/dylan/harryli/gpcdata/cesm1.2/inputdata/atm/cam/sst/'
fname = 'sst_HadOIBl_bc_1.9x2.5_1850_2008_c100127.nc'

dir_lndfrc = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
flndfrc = 'USGS-gtopo30_1.9x2.5_remap_c050602.nc'

inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

dataset = nc.Dataset(fdir+fname)
time_var = dataset.variables['time']
cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
dtime = []
for istr in cftime:
    temp = istr.strftime('%Y-%m-%d %H:%M:%S')
    dtime.append(temp)
dtime = pd.to_datetime(dtime)

select_dtime = (dtime >= inidate) & (dtime < enddate) & (
    ~((dtime.month == 2) & (dtime.day == 29)))
time = dtime[select_dtime]

lats = dataset.variables['lat'][:]
lons = dataset.variables['lon'][:]

lat_1 = np.argmin(np.abs(lats - latbounds[0]))
lat_2 = np.argmin(np.abs(lats - latbounds[1]))
lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

var = dataset.variables[varname][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
lats = lats[lat_1: lat_2 + 1]
lons = lons[lon_1: lon_2 + 1]

dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc)
lndfrc = dataset_lndfrc.variables['LANDFRAC'][lat_1: lat_2 + 1, lon_1: lon_2 + 1]
for idx in range(len(time)):
    var[idx, :, :] = np.ma.masked_where(lndfrc > 0.5, var[idx, :, :])

var[var.mask] = np.nan


print(time)
print(var.shape)

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(lats - reg_lats[0]))
model_latu = np.argmin(np.abs(lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(lons - reg_lons[1]))


############################################################################
# calculate the monthly averaged values
############################################################################

for idx in range(12):
    select_el = np.in1d(time.year, years_elweak) & (time.month == months[idx])
    select_la = np.in1d(time.year, years_laweak) & (time.month == months[idx])
    time_temp = time[select_el]

    var_el = var[select_el, :, :]
    var_la = var[select_la, :, :]

    var_el_mean = np.mean(var_el, axis=0)
    var_la_mean = np.mean(var_la, axis=0)
    var_el_std = np.std(var_el, axis=0)
    var_la_std = np.std(var_la, axis=0)

    var_diff, var_sig = cal_diff(var_el_mean, var_la_mean, var_el_std, var_la_std, len(years_elweak), len(years_laweak))
    var_elsig = var_sig
    var_lasig = var_sig
    var_elsig[:] = 0
    var_lasig[:] = 0

    plot_data = [var_el_mean, var_la_mean, var_diff]
    plot_lons = [lons, lons, lons]
    plot_lats = [lats, lats, lats]
    plot_test = [var_elsig, var_lasig, var_sig]

    legends = ['Under El Nino events', 'Under La Nina events', 'Differences (El Nino - La Nina)']

    clevs1 = np.arange(15, 37, 1)
    clevs2 = np.arange(-2, 2.4, 0.4)
    colormap = cm.RdBu_r

    title = ' CESM monthly averaged '+var_longname+' in ENSO years: '+monnames[idx]
    fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_diff_'+str(idx+1)

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs1, clevs2, legends, lonbounds,
                   latbounds, var_longname, var_unit, title, outdir+fname, opt=0)

for idx in range(12):
    select_el = np.in1d(time.year, years_elweakpre) & (time.month == months[idx])
    select_la = np.in1d(time.year, years_laweakpre) & (time.month == months[idx])
    time_temp = time[select_el]
    print(time_temp)

    var_el = var[select_el, :, :]
    var_la = var[select_la, :, :]

    var_el_mean = np.mean(var_el, axis=0)
    var_la_mean = np.mean(var_la, axis=0)
    var_el_std = np.std(var_el, axis=0)
    var_la_std = np.std(var_la, axis=0)

    var_diff, var_sig = cal_diff(var_el_mean, var_la_mean, var_el_std, var_la_std, len(years_elweak), len(years_laweak))
    var_elsig = var_sig
    var_lasig = var_sig
    var_elsig[:] = 0
    var_lasig[:] = 0

    plot_data = [var_el_mean, var_la_mean, var_diff]
    plot_lons = [lons, lons, lons]
    plot_lats = [lats, lats, lats]
    plot_test = [var_elsig, var_lasig, var_sig]

    legends = ['Under El Nino events', 'Under La Nina events', 'Differences (El Nino - La Nina)']

    clevs1 = np.arange(15, 37, 1)
    clevs2 = np.arange(-2, 2.4, 0.4)
    colormap = cm.RdBu_r

    title = ' CESM monthly averaged '+var_longname+' in ENSO years: (-1) '+monnames[idx]
    fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_diff_(-1)'+str(idx+1)

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs1, clevs2, legends, lonbounds,
                   latbounds, var_longname, var_unit, title, outdir+fname, opt=0)
