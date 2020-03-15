# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_cesm import readcesm
from modules.plot.mod_plt_contour import plot_2Dcontour
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.1


plt.switch_backend('agg')


# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = "/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/"


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/pre/drywet/'

# set up variable names and file name
varname = 'PRECT'
var_longname = 'Precipitation'
varstr = "prect"
var_unit = 'mm/day'


# define inital year and end year
iniyear = 1980
endyear = 2005
yearts = np.arange(iniyear, endyear+1)
nyears = endyear - iniyear + 1

# define the contour plot region
latbounds = [-30, 50]
lonbounds = [40, 180]

# define Southeast region
reg_lats = [10, 20]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'mon'


# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monsoon_period = [7, 8, 9]
monsoon_period_str = 'JAS'
outdir = outdir+monsoon_period_str+'/'

# define dty-wet years
years_dry = []
years_wet = []


############################################################################
# define functions
############################################################################


def cal_diff(var1, var2, std1, std2, n1, n2):
    res = var1-var2
    SE = np.sqrt((std1**2/n1) + (std2**2/n2))
    res_sig = res/SE
    res_sig = np.abs(res_sig)

    return res, res_sig


############################################################################
# read data
############################################################################

# read vrcesm

print('Reading CESM data...')

resolution = 'fv19'
varname = 'PRECT'
varfname = 'PREC'
model_var, model_time, model_lats, model_lons = readcesm(varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

model_var = model_var*86400*1000
print(model_time)
print(model_var.shape)

model_var_seas = np.zeros((nyears, len(model_lats), len(model_lons)))
for iyear in range(nyears):
    model_var_seas[iyear, :, :] = np.mean(model_var[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)


# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))


# select specific level and convert the unit to 10^-3
model_var_ann_ts = np.sum(np.sum(model_var_seas[:, model_latl: model_latu+1, model_lonl: model_lonr+1], axis=1), axis=1)
model_var_ann_ts = model_var_ann_ts/((model_latu-model_latl+1)*(model_lonr-model_lonl+1))

print(model_var_seas[0, :, :])
print(model_var_ann_ts)
print(model_var_ann_ts.shape)

############################################################################
# calculate the dry-wet years for selected period
############################################################################

monsoon_std = np.std(model_var_ann_ts)
monsoon_mean = np.mean(model_var_ann_ts)

for iyear in range(nyears):
    if (model_var_ann_ts[iyear] >= (monsoon_mean + monsoon_std)):
        years_wet.append(iniyear+iyear)
    if (model_var_ann_ts[iyear] <= (monsoon_mean - monsoon_std)):
        years_dry.append(iniyear+iyear)

print('Wet years are:')
print(years_wet)
print('Dry years are:')
print(years_dry)

############################################################################
# calculate the seasonal mean contour for dey and wet years
############################################################################
wet_index = np.in1d(yearts, years_wet)
dry_index = np.in1d(yearts, years_dry)
wet_time = yearts[wet_index]
dry_time = yearts[dry_index]
print(wet_time)
print(dry_time)

model_var_wet_mean = np.mean(model_var_seas[wet_index, :, :], axis=0)
model_var_dry_mean = np.mean(model_var_seas[dry_index, :, :], axis=0)

model_var_wet_std = np.std(model_var_seas[wet_index, :, :], axis=0)
model_var_dry_std = np.std(model_var_seas[dry_index, :, :], axis=0)

var_diff, var_sig = cal_diff(model_var_wet_mean, model_var_dry_mean, model_var_wet_std, model_var_dry_std, len(years_wet), len(years_dry))
temp_lat = np.argmin(np.abs(model_lats - (18)))
temp_lon = np.argmin(np.abs(model_lons - 90))
# print(model_var_seas[dry_index, temp_lat, temp_lon])
# print(var_diff[temp_lat, temp_lon])
# print(var_sig[temp_lat, temp_lon])
# print(model_var_wet_std[temp_lat, temp_lon])
# print(model_var_dry_std[temp_lat, temp_lon])
# temp = var_sig.flatten()
# print(temp[temp > 2.01])
# print(var_sig)

model_sig = np.zeros((len(model_lats), len(model_lons)))
model_sig[:, :] = 0.5

plot_data = [model_var_wet_mean, model_var_dry_mean, var_diff]
plot_lats = [model_lats, model_lats, model_lats]
plot_lons = [model_lons, model_lons, model_lons]
plot_test = [model_sig, model_sig, var_sig]

legends = ['a) Wet', 'b) Dry', 'c) Wet-Dry']

title = ' CESM differences in Seasonal averaged '+var_longname+' between Wet and Dry years'
fname = 'icam5_' + varstr + '_SEA_seasonal_mean_contour_diff_drywet_'+monsoon_period_str

# plot for wet-dry contours
plt.clf()
fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ss in range(3):
    axes[ss].set_title(legends[ss], fontsize=5, pad=0.3)
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
        clevs = np.arange(0, 16, 1)
        cs = map.contourf(x, y, plot_data[ss], clevs, cmap=cm.YlGn, alpha=0.9, extend="both")
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[::3]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-6, 6.1, 0.5)
        cs = map.contourf(x, y, plot_data[ss], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[::4]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=4)
    cbar.ax.set_title('mm/day', fontsize=5)

plt.savefig(outdir+fname+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=7, y=0.95)
plt.savefig(outdir+fname+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)

# plot for wet-dry contours with sig
plt.clf()
fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ss in range(3):
    axes[ss].set_title(legends[ss], fontsize=5, pad=0.3)
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
        clevs = np.arange(0, 16, 1)
        cs = map.contourf(x, y, plot_data[ss], clevs, cmap=cm.YlGn, alpha=0.9, extend="both")
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[::3]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-6, 6.1, 0.5)
        cs = map.contourf(x, y, plot_data[ss], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
        temptest = plot_test[ss]
        levels = [0., 2.01, temptest.max()]
        csm = map.contourf(x, y, plot_test[ss], levels=levels, colors='none', hatches=['', '//////'], alpha=0)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[::4]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=4)
    cbar.ax.set_title('mm/day', fontsize=5)

plt.savefig(outdir+fname+'_wtsig.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=7, y=0.95)
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

clevs = np.arange(-6, 6.1, 0.5)
cs = map.contourf(x, y, plot_data[2], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
temptest = plot_test[ss]
levels = [0., 2.01, temptest.max()]
csm = map.contourf(x, y, plot_test[2], levels=levels, colors='none', hatches=['', '//////'], alpha=0)


fig.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
ticks = clevs[::4]
ticks = np.round(ticks, 2)
ticklabels = [str(itick) for itick in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)
cbar.ax.tick_params(labelsize=7)
cbar.set_label('mm/day', fontsize=8, labelpad=0.7)

plt.savefig(outdir+fname+'_diffwtsig.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=8, y=0.95)
plt.savefig(outdir+fname+'_diffwtsig.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
