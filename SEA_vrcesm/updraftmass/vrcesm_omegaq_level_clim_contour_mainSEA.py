# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_vrcesm import readvrcesm, readvrcesm_3D
from modules.stats.mod_stats_clim import mon2clim
from modules.plot.mod_plt_lines import plot_lines
from modules.plot.mod_plt_regrid import data_regrid
from modules.plot.mod_plt_contour import plot_2Dcontour

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')


# set up data directories and filenames
case1 = "vrseasia_AMIP_1979_to_2005"
case2 = "ne30_ne30_AMIP_1979_to_2005"
case3 = "f19_f19_AMIP_1979_to_2005"
case4 = "f09_f09_AMIP_1979_to_2005"

expdir1 = "/scratch/d/dylan/harryli/cesm1/vrcesm/fields_archive/"+case1+"/atm/hist/"
expdir2 = "/scratch/d/dylan/harryli/cesm1/vrcesm/fields_archive/"+case2+"/atm/hist/"
expdir3 = "/scratch/d/dylan/harryli/cesm1/vrcesm/fields_archive/"+case3+"/atm/hist/"
expdir4 = "/scratch/d/dylan/harryli/cesm1/vrcesm/fields_archive/"+case4+"/atm/hist/"


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/updraftmass/'

# define pressure level
plevel = 850

# set up variable names and file name
varname = 'updraft mass'
var_longname = 'updraft mass'
varstr = str(plevel)+"hPa_updraftmass"
var_unit = 'mm/day'


# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-10, 25]
lonbounds = [90, 130]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'mon'

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# create seasons
seasons = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
seasnames = ['DJF', 'MAM', 'JJA', 'SON', 'Annual']

############################################################################
# physics constant
############################################################################
oro = 1000  # water density (kg/m^3)
g = 9.8  # gravitational constant (N/kg)

############################################################################
# define functions
############################################################################


def get_vars(time, var, months):
    if np.isscalar(months):
        res_var = var[time.month == months, :, :]
    else:
        # print(time[np.in1d(time.month, months)])
        res_var = var[np.in1d(time.month, months), :, :]

    return res_var

############################################################################
# read data
############################################################################

# read vrcesm


print('Reading VRCESM data...')

# read OMEGA
varname = 'OMEGA'

resolution = 'fv09'
varfname = 'OMEGA'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'OMEGA'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'OMEGA'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'OMEGA'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

print(model_var1.shape)
print(model_levs1)

# read Q
varname = 'Q'

resolution = 'fv09'
varfname = 'Q'
case = 'vrseasia_AMIP_1979_to_2005'
model_q_var1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'Q'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_q_var2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'Q'
case = 'f09_f09_AMIP_1979_to_2005'
model_q_var3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'Q'
case = 'f19_f19_AMIP_1979_to_2005'
model_q_var4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

print(model_q_var1[0, :, 10:15, 10])

# read PS
varname = 'PS'

resolution = 'fv09'
varfname = 'PS'
case = 'vrseasia_AMIP_1979_to_2005'
model_ps1, model_time1,  model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'PS'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_ps2, model_time2,  model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'PS'
case = 'f09_f09_AMIP_1979_to_2005'
model_ps3, model_time3,  model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'PS'
case = 'f19_f19_AMIP_1979_to_2005'
model_ps4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

model_ps1 = model_ps1/100
model_ps2 = model_ps2/100
model_ps3 = model_ps3/100
model_ps4 = model_ps4/100

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

model_lev1 = np.argmin(np.abs(model_levs1 - plevel))
model_lev2 = np.argmin(np.abs(model_levs2 - plevel))
model_lev3 = np.argmin(np.abs(model_levs3 - plevel))
model_lev4 = np.argmin(np.abs(model_levs4 - plevel))

print(model_lev1)

# select specific level and calculate the updraft mass, convert to mm/day
model_var1 = model_var1[:, model_lev1, :, :]
model_var2 = model_var2[:, model_lev2, :, :]
model_var3 = model_var3[:, model_lev3, :, :]
model_var4 = model_var4[:, model_lev4, :, :]

model_q_var1 = model_q_var1[:, model_lev1, :, :]
model_q_var2 = model_q_var2[:, model_lev2, :, :]
model_q_var3 = model_q_var3[:, model_lev3, :, :]
model_q_var4 = model_q_var4[:, model_lev4, :, :]

model_var1 = -1.0 * model_var1 * model_q_var1/oro/g*86400*1000
model_var2 = -1.0 * model_var2 * model_q_var2/oro/g*86400*1000
model_var3 = -1.0 * model_var3 * model_q_var3/oro/g*86400*1000
model_var4 = -1.0 * model_var4 * model_q_var4/oro/g*86400*1000

# mask the region if ps<plevel
model_var1 = np.ma.masked_where(model_ps1 < plevel, model_var1)
model_var2 = np.ma.masked_where(model_ps2 < plevel, model_var2)
model_var3 = np.ma.masked_where(model_ps3 < plevel, model_var3)
model_var4 = np.ma.masked_where(model_ps4 < plevel, model_var4)

############################################################################
# calculate climatological mean and plot
############################################################################

print('Plotting for seasonality...')

model_mean1, model_std1 = mon2clim(
    model_var1[:, model_latl1: model_latu1 + 1, model_lonl1: model_lonr1 + 1])
model_mean2, model_std2 = mon2clim(
    model_var2[:, model_latl2: model_latu2 + 1, model_lonl2: model_lonr2 + 1])
model_mean3, model_std3 = mon2clim(
    model_var3[:, model_latl3: model_latu3 + 1, model_lonl3: model_lonr3 + 1])
model_mean4, model_std4 = mon2clim(
    model_var4[:, model_latl4: model_latu4 + 1, model_lonl4: model_lonr4 + 1])


############################################################################
# plot only for cesm
plot_data = [model_mean1, model_mean2, model_mean3, model_mean4]

plot_err = [model_std1, model_std2, model_std3, model_std4]

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5']

cesm_colors = ['red', 'yellow', 'green', 'blue']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed']

# line plot
xlabel = 'Month'
ylabel = var_longname + ' ' + var_unit
title = str(iniyear) + ' to ' + str(endyear) + ' CESM ' + var_longname + ' climatology'
fname = 'vrseasia_' + varstr + '_'+reg_name+'_clim_line_overland.pdf'
plot_lines(months, plot_data, cesm_colors, cesm_line_types,
           cesm_legends, xlabel, ylabel, title, outdir+fname, yerr=plot_err, xticks=months, xticknames=monnames)


# bar plot
fig = plt.figure()
ax = fig.add_subplot(111)
bar_width = 0.85/len(plot_data)
index = months - bar_width*len(plot_data)/2
opacity = 0.8

for idx in range(len(plot_data)):
    plt.bar(index+idx*bar_width, plot_data[idx], bar_width, alpha=opacity,
            yerr=plot_err[idx], color=cesm_colors[idx], label=cesm_legends[idx])

plt.legend(handlelength=4, fontsize=5)

plt.xticks(months, monnames, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Month', fontsize=8)
plt.ylabel(var_longname + ' ' + var_unit, fontsize=8)


title = str(iniyear) + ' to ' + str(endyear) + ' CESM ' + var_longname + ' climatology'
fname = 'vrseasia_' + varstr + '_'+reg_name+'_clim_bar_overland.pdf'
plt.suptitle('CESM ' + var_longname + ' climatology', fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')


############################################################################
# calculate and plot seasonal mean
############################################################################

print('Plotting for seasonal mean...')
# calculate seasonal mean
plot_data = []
plot_err = []

plot_cesm_data = []
plot_cesm_err = []

seasons_list = ['DJF', 'MAM', 'JJA', 'SON']
for idx in range(len(seasons_list)):

    model_mean1, model_std1 = mon2clim(
        model_var1[:, model_latl1: model_latu1 + 1, model_lonl1: model_lonr1 + 1], opt=4, season=seasons_list[idx])
    model_mean2, model_std2 = mon2clim(
        model_var2[:, model_latl2: model_latu2 + 1, model_lonl2: model_lonr2 + 1], opt=4, season=seasons_list[idx])
    model_mean3, model_std3 = mon2clim(
        model_var3[:, model_latl3: model_latu3 + 1, model_lonl3: model_lonr3 + 1], opt=4, season=seasons_list[idx])
    model_mean4, model_std4 = mon2clim(
        model_var4[:, model_latl4: model_latu4 + 1, model_lonl4: model_lonr4 + 1], opt=4, season=seasons_list[idx])

    plot_cesm_data.extend([model_mean1, model_mean2, model_mean3, model_mean4])
    plot_cesm_err.extend([model_std1, model_std2, model_std3, model_std4])


# bar plot for cesm only
fig = plt.figure()
ax = fig.add_subplot(111)

ndatasets = 4
bar_width = 0.85/ndatasets
index = np.arange(1, 5) - bar_width*(ndatasets/2-0.5)
opacity = 0.8

cesm_shape_type = ['', '', '', '']

for idx in range(ndatasets):
    plt.bar(index+idx*bar_width, plot_cesm_data[idx::ndatasets], bar_width, alpha=opacity,
            color=cesm_colors[idx], label=cesm_legends[idx], hatch=cesm_shape_type[idx])
    plt.errorbar(index+idx*bar_width, plot_cesm_data[idx::ndatasets],
                 yerr=plot_cesm_err[idx::ndatasets], elinewidth=0.5, ecolor='black', fmt='none', alpha=opacity)

plt.legend(handlelength=4, fontsize=5)

plt.xticks(np.arange(1, 5), seasons_list, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Season', fontsize=8)
plt.ylabel(var_longname + ' ' + var_unit, fontsize=8)


title = str(iniyear) + ' to ' + str(endyear) + ' CESM seasonal mean '+var_longname
fname = 'vrseasia_' + varstr + '_'+reg_name+'_seasonal_mean_bar_overland.pdf'
plt.suptitle('CESM seasonal mean '+var_longname, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')


############################################################################
# calculate and plot monthly mean contour
############################################################################
# calculate monthly mean contour
print('Plotting for monthly mean contour...')


# regrid for contour plot
# lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
#
#
# model_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
# model_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
# model_var3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)

# calculate monthly mean

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

plot_list = monnames
plot_list.append('Annual')

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean...')

    # print(model_test1.shape)
    plot_data = [model_mean1[idx], model_mean2[idx], model_mean3[idx], model_mean4[idx]]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-12, 13, 1)
    colormap = cm.bwr

    title = str(iniyear)+' to '+str(endyear)+' '+imonname+' mean '+var_longname
    if idx != 12:
        fname = 'vrseasia_' + varstr + '_SEA_monthly_mean_contour_'+str(idx+1)
    else:
        fname = 'vrseasia_' + varstr + '_SEA_annual_mean_contour'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)

############################################################################
# calculate and plot seasonal mean difference
############################################################################

print('Plotting for monthly mean contour difference...')


# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
model_var_regrid1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
model_var_regrid2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
model_var_regrid3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)

# calculate monthly mean

model_mean1, model_std1 = mon2clim(model_var_regrid1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var_regrid2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var_regrid3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

model_mean1 = model_mean1 - model_mean2
model_mean3 = model_mean3 - model_mean4

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean...')

    # print(model_test1.shape)
    plot_data = [model_mean1[idx], model_mean3[idx]]
    plot_lons = [model_lons4, model_lons4]
    plot_lats = [model_lats4, model_lats4]

    legends = ['CESM-vrseasia vs CESM-ne30', 'CESM-fv0.9x1.25 vs CESM-fv1.9x2.5']

    clevs = np.arange(-6, 6.4, 0.4)
    colormap = cm.RdBu_r

    title = str(iniyear)+' to '+str(endyear)+' '+imonname+' mean '+var_longname
    if idx != 12:
        fname = 'vrseasia_' + varstr + '_SEA_monthly_mean_contour_diff_'+str(idx+1)
    else:
        fname = 'vrseasia_' + varstr + '_SEA_annual_mean_contour_diff'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)

############################################################################
# calculate and plot seasonal mean contour
############################################################################

for idx_seas in range(len(seasons)):
    print('Plot for '+seasnames[idx_seas]+':')

    model_var_sub1 = get_vars(model_time1, model_var1, seasons[idx_seas])
    model_var_sub2 = get_vars(model_time2, model_var2, seasons[idx_seas])
    model_var_sub3 = get_vars(model_time3, model_var3, seasons[idx_seas])
    model_var_sub4 = get_vars(model_time4, model_var4, seasons[idx_seas])

    model_var_mean1 = np.mean(model_var_sub1, axis=0)
    model_var_mean2 = np.mean(model_var_sub2, axis=0)
    model_var_mean3 = np.mean(model_var_sub3, axis=0)
    model_var_mean4 = np.mean(model_var_sub4, axis=0)

    # cesm only
    plot_data = [model_var_mean1, model_var_mean2, model_var_mean3, model_var_mean4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-12, 13, 1)
    colormap = cm.bwr

    title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + ' surface updraft mass over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_' + varstr + '_SEA_'+seasnames[idx_seas]+'_mean_contour'
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)

    print('Plot for '+seasnames[idx_seas]+': difference')

    # regrid for contour plot
    lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
    model_var_regrid1 = data_regrid(model_var_sub1, model_lons1, model_lats1, lonsout, latsout)
    model_var_regrid2 = data_regrid(model_var_sub2, model_lons2, model_lats2, lonsout, latsout)
    model_var_regrid3 = data_regrid(model_var_sub3, model_lons3, model_lats3, lonsout, latsout)

    model_var_mean1 = np.mean(model_var_regrid1, axis=0)
    model_var_mean2 = np.mean(model_var_regrid2, axis=0)
    model_var_mean3 = np.mean(model_var_regrid3, axis=0)

    model_var_mean1 = model_var_mean1 - model_var_mean2
    model_var_mean3 = model_var_mean3 - model_var_mean4

    plot_data = [model_var_mean1, model_var_mean3]
    plot_lons = [model_lons4, model_lons4]
    plot_lats = [model_lats4, model_lats4]

    legends = ['CESM-vrseasia vs CESM-ne30', 'CESM-fv0.9x1.25 vs CESM-fv1.9x2.5']

    clevs = np.arange(-6, 6.4, 0.4)
    colormap = cm.RdBu_r

    title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + ' surface updraft mass difference over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_' + varstr + '_SEA_'+seasnames[idx_seas]+'_mean_contour_diff'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)
