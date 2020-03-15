# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-calculate extreme
# S3-plot contour
#
# Written by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.plot.mod_plt_contour import plot_2Dcontour
from modules.plot.mod_plt_regrid import data_regrid
from modules.stats.mod_stats_clim import mon2clim
from modules.plot.mod_plt_lines import plot_lines
from modules.datareader.mod_dataread_vrcesm import readvrcesm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')

############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/pre/prect/'

############################################################################
# set parameters
############################################################################
# variable info
var_longname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# time bounds
iniyear = 1980
endyear = 2005

# define regions
latbounds = [-20, 50]
lonbounds = [40, 160]

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]

# set data frequency
frequency = 'mon'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

############################################################################
# read data
############################################################################

# read vrcesm

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'prec'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
model_mask_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
model_mask_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
model_mask_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
model_mask_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

model_var1 = model_var1 * 86400 * 1000
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000

model_mask_var1 = model_mask_var1 * 86400 * 1000
model_mask_var2 = model_mask_var2 * 86400 * 1000
model_mask_var3 = model_mask_var3 * 86400 * 1000
model_mask_var4 = model_mask_var4 * 86400 * 1000

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


############################################################################
# calculate climatological mean and plot for all grid
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
# plot line for cesm
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
fname = 'vrseasia_' + varstr + '_mainSEA_clim_line_allgrids.pdf'
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
fname = 'vrseasia_' + varstr + '_mainSEA_clim_bar_allgrids.pdf'
plt.suptitle('CESM ' + var_longname + ' climatology', fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')


# plot for over land only
model_mean1, model_std1 = mon2clim(
    model_mask_var1[:, model_latl1: model_latu1 + 1, model_lonl1: model_lonr1 + 1])
model_mean2, model_std2 = mon2clim(
    model_mask_var2[:, model_latl2: model_latu2 + 1, model_lonl2: model_lonr2 + 1])
model_mean3, model_std3 = mon2clim(
    model_mask_var3[:, model_latl3: model_latu3 + 1, model_lonl3: model_lonr3 + 1])
model_mean4, model_std4 = mon2clim(
    model_mask_var4[:, model_latl4: model_latu4 + 1, model_lonl4: model_lonr4 + 1])


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
fname = 'vrseasia_' + varstr + '_mainSEA_clim_line_overland.pdf'
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
fname = 'vrseasia_' + varstr + '_mainSEA_clim_bar_overland.pdf'
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
fname = 'vrseasia_' + varstr + '_mainSEA_seasonal_mean_bar_allgrids.pdf'
plt.suptitle('CESM seasonal mean '+var_longname, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')


# calculate seasonal mean and plot for overland
plot_data = []
plot_err = []

plot_cesm_data = []
plot_cesm_err = []

seasons_list = ['DJF', 'MAM', 'JJA', 'SON']
for idx in range(len(seasons_list)):

    model_mean1, model_std1 = mon2clim(
        model_mask_var1[:, model_latl1: model_latu1 + 1, model_lonl1: model_lonr1 + 1], opt=4, season=seasons_list[idx])
    model_mean2, model_std2 = mon2clim(
        model_mask_var2[:, model_latl2: model_latu2 + 1, model_lonl2: model_lonr2 + 1], opt=4, season=seasons_list[idx])
    model_mean3, model_std3 = mon2clim(
        model_mask_var3[:, model_latl3: model_latu3 + 1, model_lonl3: model_lonr3 + 1], opt=4, season=seasons_list[idx])
    model_mean4, model_std4 = mon2clim(
        model_mask_var4[:, model_latl4: model_latu4 + 1, model_lonl4: model_lonr4 + 1], opt=4, season=seasons_list[idx])

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
fname = 'vrseasia_' + varstr + '_mainSEA_seasonal_mean_bar_overland.pdf'
plt.suptitle('CESM seasonal mean '+var_longname, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')


############################################################################
# calculate and plot monthly mean contour
############################################################################
# calculate monthly mean contour
print('Plotting for monthly mean contour...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)


model_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
model_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
model_var3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)

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
    plot_lons = [model_lons4, model_lons4, model_lons4, model_lons4]
    plot_lats = [model_lats4, model_lats4, model_lats4, model_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(0, 10, 0.5)
    colormap = cm.Greens

    title = str(iniyear)+' to '+str(endyear)+' '+imonname+' mean '+var_longname
    if idx != 12:
        fname = 'vrseasia_' + varstr + '_SEA_monthly_mean_contour_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_' + varstr + '_SEA_annual_mean_contour.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)

model_mean1 = model_mean1 - model_mean2
model_mean3 = model_mean3 - model_mean4

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' difference...')

    # print(model_test1.shape)
    plot_data = [model_mean1[idx], model_mean3[idx]]
    plot_lons = [model_lons4, model_lons4]
    plot_lats = [model_lats4, model_lats4]

    legends = ['CESM-vrseasia vs CESM-ne30', 'CESM-fv0.9x1.25 vs CESM-fv1.9x2.5']

    clevs = np.arange(-3, 3.2, 0.2)
    colormap = cm.BrBG

    title = str(iniyear)+' to '+str(endyear)+' '+imonname+' mean '+var_longname
    if idx != 12:
        fname = 'vrseasia_' + varstr + '_SEA_monthly_mean_contour_diff_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_' + varstr + '_SEA_annual_mean_contour_diff.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)
