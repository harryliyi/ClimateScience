# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_obs_ERAinterim import readobs_ERAinterim, readobs_ERAinterim_wind_3D

import matplotlib.cm as cm
import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = '/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/'


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/climatology/circulation/era_interim/'

# set up variable names and file name
variname = 'U'
varjname = 'V'
varfname = "850hPa_wind"
varstr = "850hPa wind"
var_unit = 'm/s'

# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-40, 60]
lonbounds = [40, 180]

# latbounds = [-80, 80]
# lonbounds = [0, 360]

# latbounds = [ -40 , 40 ]
# lonbounds = [ 10 , 160 ]

# define top layer
ptop = 200
plevel = 850
plevs = [200, 300, 400, 500, 700, 850, 925]

# contants
oro_water = 997
g = 9.8
r_earth = 6371000

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

# set data frequency
frequency = 'mon'

# month series
months = np.arange(1, 13, 1)
monthts = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'DJF', 'MAM', 'JJA', 'SON']
monfnames = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', 'DJF', 'MAM', 'JJA', 'SON']


################################################################################################
# S1-read climatological data
################################################################################################
# read vrcesm
print('Reading ERA-interim data...')

# read PS
varname = 'sp'

varfname = 'surface_pressure'
model_ps, model_time,  model_lats, model_lons = readobs_ERAinterim(
    varname, iniyear, endyear, varfname, latbounds, lonbounds)

print(model_time)

model_ps = model_ps/100

varname = 'u'
varfname = 'wind'
model_u, model_time, model_levs, model_lats, model_lons = readobs_ERAinterim_wind_3D(
    varname, iniyear, endyear, varfname, latbounds, lonbounds)

varname = 'v'
varfname = 'wind'
model_v, model_time, model_levs, model_lats, model_lons = readobs_ERAinterim_wind_3D(
    varname, iniyear, endyear, varfname, latbounds, lonbounds)

print(model_time)
print(model_levs)
print(model_lats)
print(model_lons)

print('finished reading...')

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))
model_lev = np.argmin(np.abs(model_levs - plevel))
lattest = np.argmin(np.abs(model_lats - 45))
lontest = np.argmin(np.abs(model_lons - 160))

levtop = np.abs(model_levs - ptop).argmin()
nlevs = len(model_levs)

print(model_lats[model_latl])
print(model_lats[model_latu])
print(model_lons[model_lonl])
print(model_lons[model_lonr])

for idx in range(len(model_levs)):
    temp_lev = model_levs[idx]
    temp = model_u[:, idx, :, :]
    temp[model_ps < temp_lev] = np.nan
    model_u[:, idx, :, :] = temp
    temp = model_v[:, idx, :, :]
    temp[model_ps < temp_lev] = np.nan
    model_v[:, idx, :, :] = temp

print(model_ps[0, lattest, lontest])
print(model_u[0, :, lattest, lontest])
print(model_v[0, :, lattest, lontest])

for idx in range(len(monthts)):
    select_time = np.in1d(model_time.month, monthts[idx])
    time_temp = model_time[select_time]
    print('current month is '+monnames[idx])
    # print(time_temp)

    windu = np.nanmean(model_u[select_time, model_lev, :, :], axis=0)
    windv = np.nanmean(model_v[select_time, model_lev, :, :], axis=0)

    # plot data
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('  ', fontsize=8, pad=1.2)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines()
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6, linewidth=0.1)
    # x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
    mlons, mlats = np.meshgrid(model_lons, model_lats)
    x, y = map(mlons, mlats)
    # print(mlons)
    # print(lons[lonli:lonui+1])
    # print(x)
    clevs = np.arange(-12., 13., 1.)
    # clevs = np.arange(-36., 38., 2.)
    cs = map.contourf(x, y, windu, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
    cq = map.quiver(x[::5, ::5], y[::5, ::5], windu[::5, ::5], windv[::5, ::5])
    qk = ax.quiverkey(cq, 0.8, 1.03, 20, '20 '+var_unit, labelpos='E', coordinates='axes', fontproperties={'size': 7})

    # add colorbar.
    fig.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(var_unit, fontsize=8, labelpad=0.7)
    # add title
    title = 'ERA-interim monthly averaged '+str(plevel)+'hPa winds in '+monnames[idx]
    fname = 'erainterim_'+str(plevel)+'hPa_winds_SEA_monthly_mean_contour_'+monfnames[idx]
    plt.savefig(outdir+fname+'.png', bbox_inches='tight', dpi=600)
    plt.suptitle(title, fontsize=10)
    plt.savefig(outdir+fname+'.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)
