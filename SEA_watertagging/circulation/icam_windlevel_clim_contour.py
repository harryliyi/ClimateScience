# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_cesm import readcesm, readcesm_3D

import matplotlib.cm as cm
import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = '/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/'


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/climatology/circulation/icam_py_contour/'

# set up variable names and file name
variname = 'U'
varjname = 'V'
varfname = "850hPa_wind"
varstr = "850hPa wind"
var_unit = 'm/s'
var_res = "fv19"

# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-40, 60]
lonbounds = [40, 180]

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
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


################################################################################################
# S1-read climatological data
################################################################################################
# read vrcesm
print('Reading CESM data...')

# read PS
varname = 'PS'

resolution = 'fv19'
varfname = 'PS'
model_ps, model_time,  model_lats, model_lons = readcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)


model_ps = model_ps/100

varname = 'U'
varfname = 'U'
model_u, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

varname = 'V'
varfname = 'V'
model_v, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)


print('finished reading...')

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))
model_lev = np.argmin(np.abs(model_levs - plevel))
lattest = np.argmin(np.abs(model_lats - 5))
lontest = np.argmin(np.abs(model_lons - 98))

levtop = np.abs(model_levs - ptop).argmin()
nlevs = len(model_levs)
print(model_lats[model_latl])
print(model_lats[model_latu])
print(model_lons[model_lonl])
print(model_lons[model_lonr])

for idx in range(12):
    select_time = (model_time.month == months[idx])
    time_temp = model_time[select_time]
    print('current month is '+monnames[idx])

    windu = np.mean(model_u[select_time, model_lev, :, :], axis=0)
    windv = np.mean(model_v[select_time, model_lev, :, :], axis=0)

    # plot data
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines()
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
    # x, y = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
    mlons, mlats = np.meshgrid(model_lons, model_lats)
    x, y = map(mlons, mlats)
    # print(mlons)
    # print(lons[lonli:lonui+1])
    # print(x)
    clevs = np.arange(-12., 13., 1.)
    cs = map.contourf(x, y, windu, clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
    cq = map.quiver(x[::2, ::2], y[::2, ::2], windu[::2, ::2], windv[::2, ::2], scale=2., scale_units='xy')
    qk = ax.quiverkey(cq, 0.9, 0.9, 20, '20 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 8})

    # add colorbar.
    cbar = map.colorbar(cs, location='bottom', pad="5%")
    cbar.set_label(var_unit)
    # add title
    title = ' icam monthly averaged 850hPa winds in '+monnames[idx]
    fname = 'icam5_850hPa_winds_SEA_monthly_mean_contour_'+str(idx+1)
    plt.suptitle(title, fontsize=12)
    plt.savefig(outdir+fname+'.pdf')
