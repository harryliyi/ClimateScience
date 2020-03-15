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
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/omega/drywet/'

# set up variable names and file name
varname = 'OMEGA'
var_longname = r"$\omega 500$"
varstr = "500hPa_omega"
var_unit = r'$\times 10^{-3} Pa/s$'

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
plevel = 500
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

############################################################################
# define functions
############################################################################


def cal_diff(var1, var2, std1, std2, n1, n2):
    res = var1-var2
    SE = np.sqrt((std1**2/n1) + (std2**2/n2))
    res_sig = res/SE
    res_sig = np.abs(res_sig)

    return res, res_sig


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

varname = 'OMEGA'
varfname = 'OMEGA'
model_var, model_time, model_levs, model_lats, model_lons = readcesm_3D(
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

model_var = model_var[:, model_lev, :, :] * 1000

model_var[model_ps < plevel] = 0.

model_var_seas = np.zeros((nyears, len(model_lats), len(model_lons)))
for iyear in range(nyears):
    model_var_seas[iyear, :, :] = np.mean(model_var[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)

wet_index = np.in1d(yearts, years_wet)
dry_index = np.in1d(yearts, years_dry)
wet_time = yearts[wet_index]
dry_time = yearts[dry_index]

model_var_wet_mean = np.mean(model_var_seas[wet_index, :, :], axis=0)
model_var_dry_mean = np.mean(model_var_seas[dry_index, :, :], axis=0)

model_var_wet_std = np.std(model_var_seas[wet_index, :, :], axis=0)
model_var_dry_std = np.std(model_var_seas[dry_index, :, :], axis=0)

var_diff, var_sig = cal_diff(model_var_wet_mean, model_var_dry_mean, model_var_wet_std, model_var_dry_std, len(years_wet), len(years_dry))

model_sig = np.zeros((len(model_lats), len(model_lons)))
model_sig[:, :] = 0.5

plot_data = [model_var_wet_mean, model_var_dry_mean, var_diff]
plot_lats = [model_lats, model_lats, model_lats]
plot_lons = [model_lons, model_lons, model_lons]
plot_test = [model_sig, model_sig, var_sig]

legends = ['Wet', 'Dry', 'Wet-Dry']

title = ' icam seasonal averaged '+str(plevel)+'hPa OMEGA in '+monsoon_period_str
fname = 'icam5_'+str(plevel)+'hPa_omega_SEA_monthly_mean_contour_'+monsoon_period_str

print('current season is '+monsoon_period_str)

# plot for wet-dry contours
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
        clevs = np.arange(-80, 80.1, 20)
        cs = map.contourf(x, y, plot_data[ss], clevs, cmap=cm.PuOr_r, alpha=0.9, extend="both")
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[:]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-80, 80.1, 20)
        cs = map.contourf(x, y, plot_data[ss], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[:]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=3)
    cbar.ax.set_title(r'$\times 10^{-3} Pa/s$', fontsize=4, loc='left')

plt.savefig(outdir+fname+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=7, y=0.95)
plt.savefig(outdir+fname+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)

# plot for wet-dry contours with sig
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
        clevs = np.arange(-80, 80.1, 20)
        cs = map.contourf(x, y, plot_data[ss], clevs, cmap=cm.PuOr_r, alpha=0.9, extend="both")
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[:]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-80, 80.1, 20)
        cs = map.contourf(x, y, plot_data[ss], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
        temptest = plot_test[ss]
        levels = [0., 2.01, temptest.max()]
        csm = map.contourf(x, y, plot_test[ss], levels=levels, colors='none', hatches=['', '//////'], alpha=0)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[:]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=3)
    cbar.ax.set_title(r'$\times 10^{-3} Pa/s$', fontsize=4, loc='left')

plt.savefig(outdir+fname+'_wtsig.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=5, y=0.95)
plt.savefig(outdir+fname+'_wtsig.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)


# plot for wet-dry contours with sig
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# axes[ss].set_title(, fontsize=5, pad=-0.3)
map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l', ax=ax)

map.drawcoastlines(linewidth=0.3)
map.drawcountries()
parallels = np.arange(latbounds[0], latbounds[1], 20)
meridians = np.arange(lonbounds[0], lonbounds[1], 20)
map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6, linewidth=0.1)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6, linewidth=0.1)

x, y = np.meshgrid(model_lons, model_lats)

clevs = np.arange(-80, 80.1, 20)
cs = map.contourf(x, y, plot_data[2], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
temptest = plot_test[ss]
levels = [0., 2.01, temptest.max()]
csm = map.contourf(x, y, plot_test[2], levels=levels, colors='none', hatches=['', '//////'], alpha=0)


fig.subplots_adjust(bottom=0.23, wspace=0.2, hspace=0.2)
cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
ticks = clevs[:]
ticks = np.round(ticks, 2)
ticklabels = [str(itick) for itick in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)
cbar.ax.tick_params(labelsize=7)
cbar.set_label(r'$\times 10^{-3} Pa/s$', fontsize=8, labelpad=0.5)

plt.savefig(outdir+fname+'_diffwtsig.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=8, y=0.95)
plt.savefig(outdir+fname+'_diffwtsig.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
