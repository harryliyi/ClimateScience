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
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.plot.mod_plt_contour import plot_2Dcontour
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.1


plt.switch_backend('agg')


# set up output directory and output log
outdircesm = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/TCs/drywet/'

############################################################################
# set parameters
############################################################################
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

# define regions
latbounds = [-30, 50]
lonbounds = [60, 160]

# mainland Southeast Asia
reg_lats = [10, 20]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'mon'


# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monsoon_period = [5, 6, 7, 8, 9]
monsoon_period_str = 'MJJAS'
outdircesm = outdircesm+monsoon_period_str+'/'


# define the legends
cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']


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

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

# convert from m/s to mm/day
model_var1 = model_var1 * 86400 * 1000
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000
print(model_time1)
print(model_var1.shape)

model_var_seas1 = np.zeros((nyears, len(model_lats1), len(model_lons1)))
model_var_seas2 = np.zeros((nyears, len(model_lats2), len(model_lons2)))
model_var_seas3 = np.zeros((nyears, len(model_lats3), len(model_lons3)))
model_var_seas4 = np.zeros((nyears, len(model_lats4), len(model_lons4)))

for iyear in range(nyears):
    model_var_seas1[iyear, :, :] = np.mean(model_var1[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)
    model_var_seas2[iyear, :, :] = np.mean(model_var2[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)
    model_var_seas3[iyear, :, :] = np.mean(model_var3[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)
    model_var_seas4[iyear, :, :] = np.mean(model_var4[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)


# find regional lat/lon boundaries
model_latl1 = np.argmin(np.abs(model_lats1 - reg_lats[0]))
model_latu1 = np.argmin(np.abs(model_lats1 - reg_lats[1]))
model_lonl1 = np.argmin(np.abs(model_lons1 - reg_lons[0]))
model_lonr1 = np.argmin(np.abs(model_lons1 - reg_lons[1]))

model_latl2 = np.argmin(np.abs(model_lats2 - reg_lats[0]))
model_latu2 = np.argmin(np.abs(model_lats2 - reg_lats[1]))
model_lonl2 = np.argmin(np.abs(model_lons2 - reg_lons[0]))
model_lonr2 = np.argmin(np.abs(model_lons2 - reg_lons[1]))

model_latl3 = np.argmin(np.abs(model_lats3 - reg_lats[0]))
model_latu3 = np.argmin(np.abs(model_lats3 - reg_lats[1]))
model_lonl3 = np.argmin(np.abs(model_lons3 - reg_lons[0]))
model_lonr3 = np.argmin(np.abs(model_lons3 - reg_lons[1]))

model_latl4 = np.argmin(np.abs(model_lats4 - reg_lats[0]))
model_latu4 = np.argmin(np.abs(model_lats4 - reg_lats[1]))
model_lonl4 = np.argmin(np.abs(model_lons4 - reg_lons[0]))
model_lonr4 = np.argmin(np.abs(model_lons4 - reg_lons[1]))


# calculate regional mean time series
model_var_ann_ts1 = np.sum(np.sum(model_var_seas1[:, model_latl1: model_latu1+1, model_lonl1: model_lonr1+1], axis=1), axis=1)
model_var_ann_ts2 = np.sum(np.sum(model_var_seas2[:, model_latl2: model_latu2+1, model_lonl2: model_lonr2+1], axis=1), axis=1)
model_var_ann_ts3 = np.sum(np.sum(model_var_seas3[:, model_latl3: model_latu3+1, model_lonl3: model_lonr3+1], axis=1), axis=1)
model_var_ann_ts4 = np.sum(np.sum(model_var_seas4[:, model_latl4: model_latu4+1, model_lonl4: model_lonr4+1], axis=1), axis=1)


model_var_ann_ts1 = model_var_ann_ts1/((model_latu1-model_latl1+1)*(model_lonr1-model_lonl1+1))
model_var_ann_ts2 = model_var_ann_ts2/((model_latu2-model_latl2+1)*(model_lonr2-model_lonl2+1))
model_var_ann_ts3 = model_var_ann_ts3/((model_latu3-model_latl3+1)*(model_lonr3-model_lonl3+1))
model_var_ann_ts4 = model_var_ann_ts4/((model_latu4-model_latl4+1)*(model_lonr4-model_lonl4+1))

print(model_var_seas1[0, :, :])
print(model_var_ann_ts1)
print(model_var_ann_ts1.shape)

############################################################################
# calculate the dry-wet years for selected period
############################################################################

# calculate std
monsoon_std1 = np.std(model_var_ann_ts1)
monsoon_std2 = np.std(model_var_ann_ts2)
monsoon_std3 = np.std(model_var_ann_ts3)
monsoon_std4 = np.std(model_var_ann_ts4)

# calculate mean
monsoon_mean1 = np.mean(model_var_ann_ts1)
monsoon_mean2 = np.mean(model_var_ann_ts2)
monsoon_mean3 = np.mean(model_var_ann_ts3)
monsoon_mean4 = np.mean(model_var_ann_ts4)

# define dty-wet years
years_dry1 = []
years_dry2 = []
years_dry3 = []
years_dry4 = []

years_wet1 = []
years_wet2 = []
years_wet3 = []
years_wet4 = []

for iyear in range(nyears):
    if (model_var_ann_ts1[iyear] >= (monsoon_mean1 + monsoon_std1)):
        years_wet1.append(iniyear+iyear)
    if (model_var_ann_ts2[iyear] >= (monsoon_mean2 + monsoon_std2)):
        years_wet2.append(iniyear+iyear)
    if (model_var_ann_ts3[iyear] >= (monsoon_mean3 + monsoon_std3)):
        years_wet3.append(iniyear+iyear)
    if (model_var_ann_ts4[iyear] >= (monsoon_mean4 + monsoon_std4)):
        years_wet4.append(iniyear+iyear)

    if (model_var_ann_ts1[iyear] <= (monsoon_mean1 - monsoon_std1)):
        years_dry1.append(iniyear+iyear)
    if (model_var_ann_ts2[iyear] <= (monsoon_mean2 - monsoon_std2)):
        years_dry2.append(iniyear+iyear)
    if (model_var_ann_ts3[iyear] <= (monsoon_mean3 - monsoon_std3)):
        years_dry3.append(iniyear+iyear)
    if (model_var_ann_ts4[iyear] <= (monsoon_mean4 - monsoon_std4)):
        years_dry4.append(iniyear+iyear)

file1 = open(outdircesm+'years_drywet_'+monsoon_period_str+'.txt', 'w')
file1.write('Wet years are: \n')
file1.writelines(cesm_legends[0]+':')
strlines = [str(ii) for ii in years_wet1]
strlines.append(' \n')
file1.writelines(strlines)
file1.writelines(cesm_legends[1]+':')
strlines = [str(ii) for ii in years_wet2]
strlines.append(' \n')
file1.writelines(strlines)
file1.writelines(cesm_legends[2]+':')
strlines = [str(ii) for ii in years_wet3]
strlines.append(' \n')
file1.writelines(strlines)
file1.writelines(cesm_legends[3]+':')
strlines = [str(ii) for ii in years_wet4]
strlines.append(' \n')
file1.writelines(strlines)

file1.write('Dry years are: \n')
file1.writelines(cesm_legends[0]+':')
strlines = [str(ii) for ii in years_dry1]
strlines.append(' \n')
file1.writelines(strlines)
file1.writelines(cesm_legends[1]+':')
strlines = [str(ii) for ii in years_dry2]
strlines.append(' \n')
file1.writelines(strlines)
file1.writelines(cesm_legends[2]+':')
strlines = [str(ii) for ii in years_dry3]
strlines.append(' \n')
file1.writelines(strlines)
file1.writelines(cesm_legends[3]+':')
strlines = [str(ii) for ii in years_dry4]
strlines.append(' \n')
file1.writelines(strlines)
file1.close()

print('Wet years are:')
print(cesm_legends[0]+':')
print(years_wet1)
print(cesm_legends[1]+':')
print(years_wet2)
print(cesm_legends[2]+':')
print(years_wet3)
print(cesm_legends[3]+':')
print(years_wet4)

print('Dry years are:')
print(cesm_legends[0]+':')
print(years_dry1)
print(cesm_legends[1]+':')
print(years_dry2)
print(cesm_legends[2]+':')
print(years_dry3)
print(cesm_legends[3]+':')
print(years_dry4)

############################################################################
# calculate the seasonal mean contour for dey and wet years
############################################################################

years_wet = [years_wet1, years_wet2, years_wet3, years_wet4]
years_dry = [years_dry1, years_dry2, years_dry3, years_dry4]

var_seas = [model_var_seas1, model_var_seas2, model_var_seas3, model_var_seas4]

model_lats = [model_lats1, model_lats2, model_lats3, model_lats4]
model_lons = [model_lons1, model_lons2, model_lons3, model_lons4]

alldif = []
allsig = []

for idx in range(4):
    wet_index = np.in1d(yearts, years_wet[idx])
    dry_index = np.in1d(yearts, years_dry[idx])
    wet_time = yearts[wet_index]
    dry_time = yearts[dry_index]

    model_var_seas = var_seas[idx]
    model_var_wet_mean = np.mean(model_var_seas[wet_index, :, :], axis=0)
    model_var_dry_mean = np.mean(model_var_seas[dry_index, :, :], axis=0)

    model_var_wet_std = np.std(model_var_seas[wet_index, :, :], axis=0)
    model_var_dry_std = np.std(model_var_seas[dry_index, :, :], axis=0)

    var_diff, var_sig = cal_diff(model_var_wet_mean, model_var_dry_mean, model_var_wet_std, model_var_dry_std, len(years_wet), len(years_dry))

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
    plot_lats = [model_lats[idx], model_lats[idx], model_lats[idx]]
    plot_lons = [model_lons[idx], model_lons[idx], model_lons[idx]]
    plot_test = [model_sig, model_sig, var_sig]

    alldif.append(var_diff)
    allsig.append(var_sig)

    legends = ['a) Wet', 'b) Dry', 'c) Wet-Dry']

    title = cesm_legends[idx]+' differences in Seasonal averaged total precip between Wet and Dry years'
    fname = 'vrcesm_prect_SEA_seasonal_mean_contour_diff_drywet_'+monsoon_period_str+'_'+str(idx+1)

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

        x, y = np.meshgrid(plot_lons[ss], plot_lats[ss])
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
        cbar.ax.tick_params(labelsize=3)
        cbar.ax.set_title('mm/day', fontsize=4)

    plt.savefig(outdircesm+fname+'.png', bbox_inches='tight', dpi=600)
    plt.suptitle(title, fontsize=5, y=0.95)
    plt.savefig(outdircesm+fname+'.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)

plot_data = alldif
plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]
plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
plot_test = allsig

title = 'CESM differences in Seasonal averaged total precip between Wet and Dry years'
fname = 'vrcesm_prect_SEA_seasonal_mean_contour_diff_drywet_'+monsoon_period_str+'_allmodels'


# plot all difference together
# plot for wet-dry contours with sig
plt.clf()
fig = plt.figure()

for idx in range(4):
    ax = fig.add_subplot(2, 2, idx+1)

    ax.set_title(cesm_legends[idx], fontsize=5, pad=-0.3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l', ax=ax)

    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)

    x, y = np.meshgrid(plot_lons[idx], plot_lats[idx])

    clevs = np.arange(-6, 6.1, 0.5)
    cs = map.contourf(x, y, plot_data[idx], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
    temptest = plot_test[ss]
    levels = [0., 2.01, temptest.max()]
    csm = map.contourf(x, y, plot_test[idx], levels=levels, colors='none', hatches=['', '//////'], alpha=0)

fig.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
ticks = clevs[::4]
ticks = np.round(ticks, 2)
ticklabels = [str(itick) for itick in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)
cbar.ax.tick_params(labelsize=4)
cbar.set_label('mm/day', fontsize=5, labelpad=0.5)

plt.savefig(outdircesm+fname+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=5, y=0.95)
plt.savefig(outdircesm+fname+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
