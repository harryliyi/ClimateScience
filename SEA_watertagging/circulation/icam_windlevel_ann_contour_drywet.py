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
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/circulation/drywet/'

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
yearts = np.arange(iniyear, endyear+1)
nyears = len(yearts)

# define the contour plot region
latbounds = [-20, 40]
lonbounds = [60, 150]

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
reg_lats = [10, 20]
reg_lons = [100, 110]

# define Wet-Dry yyears
monsoon_period_str = 'May'
if monsoon_period_str == 'JJA':
    monsoon_period = [6, 7, 8]
    years_wet = [1987, 1993, 2002]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'JJAS':
    monsoon_period = [6, 7, 8, 9]
    years_wet = [1987, 1993, 2002]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'MJJAS':
    monsoon_period = [5, 6, 7, 8, 9]
    years_wet = [1987, 1993, 2000]
    years_dry = [1988, 1992, 1995, 1998, 1999, 2005]
if monsoon_period_str == 'JAS':
    monsoon_period = [7, 8, 9]
    years_wet = [1987, 1991, 1993, 2000]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'AS':
    monsoon_period = [8, 9]
    years_wet = [1987, 1991, 1996, 2000]
    years_dry = [1980, 1990, 1995, 1998, 2005]
if monsoon_period_str == 'Sep':
    monsoon_period = [9]
    years_wet = [1981, 1983, 1987, 1996, 2000]
    years_dry = [1980, 1990, 1995, 2005]
if monsoon_period_str == 'Oct':
    monsoon_period = [10]
    years_wet = [1981, 1986, 1989, 2001]
    years_dry = [1990, 1992, 1993, 1995, 2004]
if monsoon_period_str == 'May':
    monsoon_period = [5]
    years_wet = [1981, 1982, 1985, 1990, 2000]
    years_dry = [1983, 1987, 1988, 1992, 1998]
if monsoon_period_str == 'AM':
    monsoon_period = [4, 5]
    years_wet = [1981, 1985, 1996, 2000]
    years_dry = [1983, 1987, 1992, 1998, 2005]
if monsoon_period_str == 'MJ':
    monsoon_period = [5, 6]
    years_wet = [1981, 1985, 1997, 2001]
    years_dry = [1987, 1988, 1992, 2005]
outdir = outdir + monsoon_period_str+'/'

nyears_wet = len(years_wet)
nyears_dry = len(years_dry)

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

model_u = model_u[:, model_lev, :, :]
model_v = model_v[:, model_lev, :, :]

model_u[model_ps < plevel] = 0.
model_v[model_ps < plevel] = 0.

model_u_seas = np.zeros((nyears, len(model_lats), len(model_lons)))
model_v_seas = np.zeros((nyears, len(model_lats), len(model_lons)))
for iyear in range(nyears):
    model_u_seas[iyear, :, :] = np.mean(model_u[12*iyear+monsoon_period[0]-1: 12 *
                                                iyear+monsoon_period[-1], :, :], axis=0)
    model_v_seas[iyear, :, :] = np.mean(model_v[12*iyear+monsoon_period[0]-1: 12 *
                                                iyear+monsoon_period[-1], :, :], axis=0)

wet_index = np.in1d(yearts, years_wet)
dry_index = np.in1d(yearts, years_dry)
wet_time = yearts[wet_index]
dry_time = yearts[dry_index]

model_u_wet = np.mean(model_u_seas[wet_index, :, :], axis=0)
model_u_dry = np.mean(model_u_seas[dry_index, :, :], axis=0)
model_v_wet = np.mean(model_v_seas[wet_index, :, :], axis=0)
model_v_dry = np.mean(model_v_seas[dry_index, :, :], axis=0)

plot_udata = [model_u_wet, model_u_dry, model_u_wet-model_u_dry]
plot_vdata = [model_v_wet, model_v_dry, model_v_wet-model_v_dry]
plot_lats = [model_lats, model_lats, model_lats]
plot_lons = [model_lons, model_lons, model_lons]

legends = ['Wet', 'Dry', 'Wet-Dry']

title = ' icam seasonal averaged '+str(plevel)+'hPa winds in '+monsoon_period_str
fname = 'icam5_'+str(plevel)+'hPa_winds_SEA_monthly_mean_contour_'+monsoon_period_str

print('current season is '+monsoon_period_str)

# plot data
plt.clf()
fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ss in range(3):
    axes[ss].set_title(legends[ss], fontsize=5, pad=-0.3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l', ax=axes[ss])
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)

    x, y = np.meshgrid(model_lons, model_lats)

    if ss < 2:
        clevs = np.arange(-16., 17., 2.)
        cs = map.contourf(x, y, plot_udata[ss], clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
        cq = map.quiver(x[::1, ::1], y[::1, ::1], plot_udata[ss][::1, ::1],
                        plot_vdata[ss][::1, ::1], scale=2.5, scale_units='xy')
        if ss == 0:
            qk = axes[ss].quiverkey(cq, 0.75, 0.9, 20, '20 '+var_unit, labelpos='E',
                                    coordinates='figure', fontproperties={'size': 4})
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[::3]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-12., 13., 1.)
        cs = map.contourf(x, y, plot_udata[ss], clevs, cmap=cm.PuOr, alpha=0.9, extend="both")
        cq = map.quiver(x[::1, ::1], y[::1, ::1], plot_udata[ss][::1, ::1],
                        plot_vdata[ss][::1, ::1], scale=2.5, scale_units='xy')
        # qk = axes[ss].quiverkey(cq, 0.7, 1.05, 20, '20 '+var_unit, labelpos='E', coordinates='figure', fontproperties={'size': 5})
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[::4]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=3)
    cbar.ax.set_title('m/s', fontsize=4)

plt.savefig(outdir+fname+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=5, y=0.95)
plt.savefig(outdir+fname+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
