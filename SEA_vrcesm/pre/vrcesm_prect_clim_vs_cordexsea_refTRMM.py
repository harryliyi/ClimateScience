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
from modules.stats.mod_stats_clim import mon2clim, getstats_2Dsig
from modules.datareader.mod_dataread_obs_pre import readobs_pre_mon
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex
from modules.datareader.mod_dataread_obs_TRMM import readobs_pre_TRMM_mon
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# import modules

############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/1998-2005/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# time bounds
iniyear = 1998
endyear = 2005

# define regions
latbounds = [-15, 25]
lonbounds = [90, 145]

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

print('Reading CORDEX-SEA data...')

# read cordex
project = 'SEA-22'
varname = 'pr'
cordex_models = ['ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-LR', 'MPI-M-MPI-ESM-MR', 'MOHC-HadGEM2-ES']

modelname = 'ICHEC-EC-EARTH'
cordex_var1, cordex_time1, cordex_lats1, cordex_lons1 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=1)
modelname = 'IPSL-IPSL-CM5A-LR'
cordex_var2, cordex_time2, cordex_lats2, cordex_lons2 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=1)
modelname = 'MPI-M-MPI-ESM-MR'
cordex_var3, cordex_time3, cordex_lats3, cordex_lons3 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=1)
modelname = 'MOHC-HadGEM2-ES'
cordex_var4, cordex_time4, cordex_lats4, cordex_lons4 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=1)

# find regional lat/lon boundaries
cordex_latl1 = np.argmin(np.abs(cordex_lats1 - reg_lats[0]))
cordex_latu1 = np.argmin(np.abs(cordex_lats1 - reg_lats[1]))
cordex_lonl1 = np.argmin(np.abs(cordex_lons1 - reg_lons[0]))
cordex_lonr1 = np.argmin(np.abs(cordex_lons1 - reg_lons[1]))

cordex_latl2 = np.argmin(np.abs(cordex_lats2 - reg_lats[0]))
cordex_latu2 = np.argmin(np.abs(cordex_lats2 - reg_lats[1]))
cordex_lonl2 = np.argmin(np.abs(cordex_lons2 - reg_lons[0]))
cordex_lonr2 = np.argmin(np.abs(cordex_lons2 - reg_lons[1]))

cordex_latl3 = np.argmin(np.abs(cordex_lats3 - reg_lats[0]))
cordex_latu3 = np.argmin(np.abs(cordex_lats3 - reg_lats[1]))
cordex_lonl3 = np.argmin(np.abs(cordex_lons3 - reg_lons[0]))
cordex_lonr3 = np.argmin(np.abs(cordex_lons3 - reg_lons[1]))

cordex_latl4 = np.argmin(np.abs(cordex_lats4 - reg_lats[0]))
cordex_latu4 = np.argmin(np.abs(cordex_lats4 - reg_lats[1]))
cordex_lonl4 = np.argmin(np.abs(cordex_lons4 - reg_lons[0]))
cordex_lonr4 = np.argmin(np.abs(cordex_lons4 - reg_lons[1]))


# read vrcesm

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'prec'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

model_var1 = model_var1 * 86400 * 1000           # convert to mm/day
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000

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

# read Observations

print('Reading Obs data...')

# read CRU
project = 'CRU'
obs_var1, obs_time1, obs_lats1, obs_lons1 = readobs_pre_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read GPCC
project = 'GPCP'
obs_var2, obs_time2, obs_lats2, obs_lons2 = readobs_pre_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read GPCC
project = 'GPCC'
obs_var3, obs_time3, obs_lats3, obs_lons3 = readobs_pre_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read GPCC
project = 'ERA-interim'
obs_var4, obs_time4, obs_lats4, obs_lons4 = readobs_pre_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read APHRODITE
project = 'APHRODITE'
obs_var5, obs_time5, obs_lats5, obs_lons5 = readobs_pre_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read TRMM
obs_var6, obs_time6, obs_lats6, obs_lons6 = readobs_pre_TRMM_mon(
    'precipitation', iniyear, endyear, latbounds, lonbounds, oceanmask=1)

print(obs_var6[0, :, :])

# find regional lat/lon boundaries
obs_latl1 = np.argmin(np.abs(obs_lats1 - reg_lats[0]))
obs_latu1 = np.argmin(np.abs(obs_lats1 - reg_lats[1]))
obs_lonl1 = np.argmin(np.abs(obs_lons1 - reg_lons[0]))
obs_lonr1 = np.argmin(np.abs(obs_lons1 - reg_lons[1]))

obs_latl2 = np.argmin(np.abs(obs_lats2 - reg_lats[0]))
obs_latu2 = np.argmin(np.abs(obs_lats2 - reg_lats[1]))
obs_lonl2 = np.argmin(np.abs(obs_lons2 - reg_lons[0]))
obs_lonr2 = np.argmin(np.abs(obs_lons2 - reg_lons[1]))

obs_latl3 = np.argmin(np.abs(obs_lats3 - reg_lats[0]))
obs_latu3 = np.argmin(np.abs(obs_lats3 - reg_lats[1]))
obs_lonl3 = np.argmin(np.abs(obs_lons3 - reg_lons[0]))
obs_lonr3 = np.argmin(np.abs(obs_lons3 - reg_lons[1]))

obs_latl4 = np.argmin(np.abs(obs_lats4 - reg_lats[0]))
obs_latu4 = np.argmin(np.abs(obs_lats4 - reg_lats[1]))
obs_lonl4 = np.argmin(np.abs(obs_lons4 - reg_lons[0]))
obs_lonr4 = np.argmin(np.abs(obs_lons4 - reg_lons[1]))

obs_latl5 = np.argmin(np.abs(obs_lats5 - reg_lats[0]))
obs_latu5 = np.argmin(np.abs(obs_lats5 - reg_lats[1]))
obs_lonl5 = np.argmin(np.abs(obs_lons5 - reg_lons[0]))
obs_lonr5 = np.argmin(np.abs(obs_lons5 - reg_lons[1]))

obs_latl6 = np.argmin(np.abs(obs_lats6 - reg_lats[0]))
obs_latu6 = np.argmin(np.abs(obs_lats6 - reg_lats[1]))
obs_lonl6 = np.argmin(np.abs(obs_lons6 - reg_lons[0]))
obs_lonr6 = np.argmin(np.abs(obs_lons6 - reg_lons[1]))

print(obs_var1[0, obs_latl1: obs_latu1 + 1, obs_lonl1: obs_lonr1 + 1])
# $print(obs_lats1[obs_latl1 : obs_latu1 + 1])

############################################################################
# calculate climatological mean and plot
############################################################################

print('Plotting for seasonality...')

# calculate seasonality
cordex_mean1, codex_std1 = mon2clim(
    cordex_var1[:, cordex_latl1: cordex_latu1 + 1, cordex_lonl1: cordex_lonr1 + 1])
cordex_mean2, codex_std2 = mon2clim(
    cordex_var2[:, cordex_latl2: cordex_latu2 + 1, cordex_lonl2: cordex_lonr2 + 1])
cordex_mean3, codex_std3 = mon2clim(
    cordex_var3[:, cordex_latl3: cordex_latu3 + 1, cordex_lonl3: cordex_lonr3 + 1])
cordex_mean4, codex_std4 = mon2clim(
    cordex_var4[:, cordex_latl4: cordex_latu4 + 1, cordex_lonl4: cordex_lonr4 + 1])

model_mean1, model_std1 = mon2clim(
    model_var1[:, model_latl1: model_latu1 + 1, model_lonl1: model_lonr1 + 1])
model_mean2, model_std2 = mon2clim(
    model_var2[:, model_latl2: model_latu2 + 1, model_lonl2: model_lonr2 + 1])
model_mean3, model_std3 = mon2clim(
    model_var3[:, model_latl3: model_latu3 + 1, model_lonl3: model_lonr3 + 1])
model_mean4, model_std4 = mon2clim(
    model_var4[:, model_latl4: model_latu4 + 1, model_lonl4: model_lonr4 + 1])

obs_mean1, obs_std1 = mon2clim(obs_var1[:, obs_latl1: obs_latu1 + 1, obs_lonl1: obs_lonr1 + 1])
obs_mean3, obs_std3 = mon2clim(obs_var3[:, obs_latl3: obs_latu3 + 1, obs_lonl3: obs_lonr3 + 1])
obs_mean5, obs_std5 = mon2clim(obs_var5[:, obs_latl5: obs_latu5 + 1, obs_lonl5: obs_lonr5 + 1])
obs_mean6, obs_std6 = mon2clim(obs_var6[:, obs_latl6: obs_latu6 + 1, obs_lonl6: obs_lonr6 + 1])
print(obs_mean1)

plot_data = [model_mean1, model_mean2, model_mean3, model_mean4, cordex_mean1, cordex_mean2, cordex_mean3, cordex_mean4, obs_mean1, obs_mean3, obs_mean5, obs_mean6]

plot_err = [model_std1, model_std2, model_std3, model_std4, codex_std1, codex_std2, codex_std3, codex_std4, obs_std1, obs_std3, obs_std5, obs_std6]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES', 'CRU-TS3.21', 'GPCC-v7', 'APHRODITE-MA', 'TRMM']

colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod', 'darkcyan', 'darkmagenta', 'black', 'brown', 'midnightblue', 'darkslategrey']
line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-.', '-.', '-.', '-.', '-', '-', '-', '-']


# line plot
fig = plt.figure()
ax = fig.add_subplot(111)

for idx in range(len(plot_data)):
    # print(plot_data[idx])
    plt.plot(months, plot_data[idx], color=colors[idx], marker='o', markersize=1,
             linestyle=line_types[idx], linewidth=1.5, label=legends[idx])
    plt.errorbar(months, plot_data[idx], yerr=plot_err[idx], fmt='o',
                 markersize=1, elinewidth=1, color=colors[idx])


plt.legend(handlelength=4, fontsize=5)

plt.xticks(months, monnames, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Month', fontsize=8)
plt.ylabel('Precip (mm/day)', fontsize=8)

plt.suptitle(str(iniyear)+' to '+str(endyear)+' VRCESM vs CORDEX-SEA Precip climatology', fontsize=9, y=0.95)

fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_mainSEA_clim_line_vs_cordex_overland.pdf'
plt.savefig(outdir+fname, bbox_inches='tight')


# bar plot
fig = plt.figure()
ax = fig.add_subplot(111)
bar_width = 0.85/len(plot_data)
index = months - bar_width*len(plot_data)/2
opacity = 0.8

for idx in range(len(plot_data)):
    plt.bar(index+idx*bar_width, plot_data[idx], bar_width, alpha=opacity,
            yerr=plot_err[idx], color=colors[idx], label=legends[idx])

plt.legend(handlelength=4, fontsize=5)

plt.xticks(months, monnames, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Month', fontsize=8)
plt.ylabel('Precip (mm/day)', fontsize=8)

plt.suptitle(str(iniyear)+' to '+str(endyear)+' VRCESM vs CORDEX-SEA Precip climatology', fontsize=9, y=0.95)

fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_mainSEA_clim_bar_vs_cordex_overland.pdf'
plt.savefig(outdir+fname, bbox_inches='tight')

############################################################################
# plot only for cesm
plot_data = [model_mean1, model_mean2, model_mean3, model_mean4, obs_mean1, obs_mean3, obs_mean5, obs_mean6]

plot_err = [model_std1, model_std2, model_std3, model_std4, obs_std1, obs_std3, obs_std5, obs_std6]

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5', 'CRU-TS3.21', 'GPCC-v7', 'APHRODITE-MA', 'TRMM']

cesm_colors = ['red', 'yellow', 'green', 'blue', 'black', 'brown', 'midnightblue', 'darkslategrey']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-', '-', '-', '-']

# line plot
fig = plt.figure()
ax = fig.add_subplot(111)

for idx in range(len(plot_data)):
    # print(plot_data[idx])
    plt.plot(months, plot_data[idx], color=cesm_colors[idx], marker='o', markersize=1,
             linestyle=cesm_line_types[idx], linewidth=1.5, label=cesm_legends[idx])
    plt.errorbar(months, plot_data[idx], yerr=plot_err[idx],
                 fmt='o', markersize=1, elinewidth=1, ecolor='black')


plt.legend(handlelength=4, fontsize=5)

plt.xticks(months, monnames, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Month', fontsize=8)
plt.ylabel('Precip (mm/day)', fontsize=8)

plt.suptitle(str(iniyear)+' to '+str(endyear)+' CESM Precip climatology', fontsize=9, y=0.95)

fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_mainSEA_clim_line_overland.pdf'
plt.savefig(outdir+fname, bbox_inches='tight')


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
plt.ylabel('Precip (mm/day)', fontsize=8)

plt.suptitle(str(iniyear)+' to '+str(endyear)+' CESM Precip climatology', fontsize=9, y=0.95)

fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_mainSEA_clim_bar_overland.pdf'
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
    cordex_mean1, codex_std1 = mon2clim(
        cordex_var1[:, cordex_latl1: cordex_latu1 + 1, cordex_lonl1: cordex_lonr1 + 1], opt=4, season=seasons_list[idx])
    cordex_mean2, codex_std2 = mon2clim(
        cordex_var2[:, cordex_latl2: cordex_latu2 + 1, cordex_lonl2: cordex_lonr2 + 1], opt=4, season=seasons_list[idx])
    cordex_mean3, codex_std3 = mon2clim(
        cordex_var3[:, cordex_latl3: cordex_latu3 + 1, cordex_lonl3: cordex_lonr3 + 1], opt=4, season=seasons_list[idx])
    cordex_mean4, codex_std4 = mon2clim(
        cordex_var4[:, cordex_latl4: cordex_latu4 + 1, cordex_lonl4: cordex_lonr4 + 1], opt=4, season=seasons_list[idx])

    model_mean1, model_std1 = mon2clim(
        model_var1[:, model_latl1: model_latu1 + 1, model_lonl1: model_lonr1 + 1], opt=4, season=seasons_list[idx])
    model_mean2, model_std2 = mon2clim(
        model_var2[:, model_latl2: model_latu2 + 1, model_lonl2: model_lonr2 + 1], opt=4, season=seasons_list[idx])
    model_mean3, model_std3 = mon2clim(
        model_var3[:, model_latl3: model_latu3 + 1, model_lonl3: model_lonr3 + 1], opt=4, season=seasons_list[idx])
    model_mean4, model_std4 = mon2clim(
        model_var4[:, model_latl4: model_latu4 + 1, model_lonl4: model_lonr4 + 1], opt=4, season=seasons_list[idx])

    obs_mean1, obs_std1 = mon2clim(
        obs_var1[:, obs_latl1: obs_latu1 + 1, obs_lonl1: obs_lonr1 + 1], opt=4, season=seasons_list[idx])
    obs_mean3, obs_std3 = mon2clim(
        obs_var3[:, obs_latl3: obs_latu3 + 1, obs_lonl3: obs_lonr3 + 1], opt=4, season=seasons_list[idx])
    obs_mean5, obs_std5 = mon2clim(
        obs_var5[:, obs_latl5: obs_latu5 + 1, obs_lonl5: obs_lonr5 + 1], opt=4, season=seasons_list[idx])
    obs_mean6, obs_std6 = mon2clim(
        obs_var6[:, obs_latl6: obs_latu6 + 1, obs_lonl6: obs_lonr6 + 1], opt=4, season=seasons_list[idx])

    plot_data.extend([model_mean1, model_mean2, model_mean3, model_mean4, cordex_mean1,
                      cordex_mean2, cordex_mean3, cordex_mean4, obs_mean1, obs_mean3, obs_mean5, obs_mean6])
    plot_err.extend([model_std1, model_std2, model_std3, model_std4, codex_std1,
                     codex_std2, codex_std3, codex_std4, obs_std1, obs_std3, obs_std5, obs_std6])
    plot_cesm_data.extend([model_mean1, model_mean2, model_mean3,
                           model_mean4, obs_mean1, obs_mean3, obs_mean5, obs_mean6])
    plot_cesm_err.extend([model_std1, model_std2, model_std3,
                          model_std4, obs_std1, obs_std3, obs_std5, obs_std6])


# bar plot
fig = plt.figure()
ax = fig.add_subplot(111)

ndatasets = 12
bar_width = 0.85/ndatasets
index = np.arange(1, 5) - bar_width*(ndatasets/2-0.5)
opacity = 0.8

shape_type = ['', '', '', '', '..', '..', '..', '..', '//', '//', '//', '//']

for idx in range(ndatasets):
    plt.bar(index+idx*bar_width, plot_data[idx::ndatasets], bar_width,
            alpha=opacity, color=colors[idx], label=legends[idx], hatch=shape_type[idx])
    plt.errorbar(index+idx*bar_width, plot_data[idx::ndatasets], yerr=plot_err[idx::ndatasets],
                 elinewidth=0.5, ecolor='black', fmt='none', alpha=opacity)

plt.legend(handlelength=4, fontsize=5)

plt.xticks(np.arange(1, 5), seasons_list, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Season', fontsize=8)
plt.ylabel('Precip (mm/day)', fontsize=8)

plt.suptitle(str(iniyear)+' to '+str(endyear)+' VRCESM vs CORDEX-SEA seasonal mean Precip', fontsize=9, y=0.95)

fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_mainSEA_seasonal_mean_bar_vs_cordex_overland.pdf'
plt.savefig(outdir+fname, bbox_inches='tight')

# bar plot for cesm only
fig = plt.figure()
ax = fig.add_subplot(111)

ndatasets = 8
bar_width = 0.85/ndatasets
index = np.arange(1, 5) - bar_width*(ndatasets/2-0.5)
opacity = 0.8

cesm_shape_type = ['', '', '', '', '//', '//', '//', '//']

for idx in range(ndatasets):
    plt.bar(index+idx*bar_width, plot_cesm_data[idx::ndatasets], bar_width, alpha=opacity,
            color=cesm_colors[idx], label=cesm_legends[idx], hatch=cesm_shape_type[idx])
    plt.errorbar(index+idx*bar_width, plot_cesm_data[idx::ndatasets],
                 yerr=plot_cesm_err[idx::ndatasets], elinewidth=0.5, ecolor='black', fmt='none', alpha=opacity)

plt.legend(handlelength=4, fontsize=5)

plt.xticks(np.arange(1, 5), seasons_list, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Season', fontsize=8)
plt.ylabel('Precip (mm/day)', fontsize=8)

plt.suptitle(str(iniyear)+' to '+str(endyear)+' CESM seasonal mean Precip', fontsize=9, y=0.95)

fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_mainSEA_seasonal_mean_bar_overland.pdf'
plt.savefig(outdir+fname, bbox_inches='tight')


############################################################################
# calculate and plot monthly mean contour
############################################################################
# calculate monthly mean contour
print('Plotting for monthly mean contour...')

# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)

# print(cordex_var1.shape)
cordex_var1 = data_regrid(cordex_var1, cordex_lons1, cordex_lats1, lonsout, latsout)
cordex_var2 = data_regrid(cordex_var2, cordex_lons2, cordex_lats2, lonsout, latsout)
cordex_var3 = data_regrid(cordex_var3, cordex_lons3, cordex_lats3, lonsout, latsout)
cordex_var4 = data_regrid(cordex_var4, cordex_lons4, cordex_lats4, lonsout, latsout)
# print(cordex_var1.shape)
# print(cordex_var1[0,:,:])
# print(model_var4[0,:,:])

model_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
model_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
model_var3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)

obs_var1 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
obs_var3 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
obs_var5 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
obs_var6 = data_regrid(obs_var6, obs_lons6, obs_lats6, lonsout, latsout)

# calculate monthly mean
cordex_mean1, codex_std1 = mon2clim(cordex_var1[:, :, :], opt=2)
cordex_mean2, codex_std2 = mon2clim(cordex_var2[:, :, :], opt=2)
cordex_mean3, codex_std3 = mon2clim(cordex_var3[:, :, :], opt=2)
cordex_mean4, codex_std4 = mon2clim(cordex_var4[:, :, :], opt=2)

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

obs_mean1, obs_std1 = mon2clim(obs_var1[:, :, :], opt=2)
obs_mean3, obs_std3 = mon2clim(obs_var3[:, :, :], opt=2)
obs_mean5, obs_std5 = mon2clim(obs_var5[:, :, :], opt=2)
obs_mean6, obs_std6 = mon2clim(obs_var6[:, :, :], opt=2)

plot_list = monnames
plot_list.append('Annual')

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean...')
    # vs CRU
    cordex_diff1, cordex_test1 = getstats_2Dsig(
        cordex_mean1[idx, :, :], obs_mean1[idx, :, :], codex_std1[idx, :, :], obs_std1[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff2, cordex_test2 = getstats_2Dsig(
        cordex_mean2[idx, :, :], obs_mean1[idx, :, :], codex_std2[idx, :, :], obs_std1[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff3, cordex_test3 = getstats_2Dsig(
        cordex_mean3[idx, :, :], obs_mean1[idx, :, :], codex_std3[idx, :, :], obs_std1[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff4, cordex_test4 = getstats_2Dsig(
        cordex_mean4[idx, :, :], obs_mean1[idx, :, :], codex_std4[idx, :, :], obs_std1[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)

    model_diff1, model_test1 = getstats_2Dsig(
        model_mean1[idx, :, :], obs_mean1[idx, :, :], model_std1[idx, :, :], obs_std1[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff2, model_test2 = getstats_2Dsig(
        model_mean2[idx, :, :], obs_mean1[idx, :, :], model_std2[idx, :, :], obs_std1[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff3, model_test3 = getstats_2Dsig(
        model_mean3[idx, :, :], obs_mean1[idx, :, :], model_std3[idx, :, :], obs_std1[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff4, model_test4 = getstats_2Dsig(
        model_mean4[idx, :, :], obs_mean1[idx, :, :], model_std4[idx, :, :], obs_std1[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons4, model_lons4, model_lons4, model_lons4,
                 model_lons4, model_lons4, model_lons4, model_lons4]
    plot_lats = [model_lats4, model_lats4, model_lats4, model_lats4,
                 model_lats4, model_lats4, model_lats4, model_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5, 5, 0.5)
    colormap = cm.BrBG

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' bias (CRU)'
    if idx != 12:
        fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_SEA_monthly_mean_contour_vs_cordex_refCRU_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_SEA_annual_mean_contour_vs_cordex_refCRU.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=1, sig_test=plot_test)

    # vs GPCC
    cordex_diff1, cordex_test1 = getstats_2Dsig(
        cordex_mean1[idx, :, :], obs_mean3[idx, :, :], codex_std1[idx, :, :], obs_std3[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff2, cordex_test2 = getstats_2Dsig(
        cordex_mean2[idx, :, :], obs_mean3[idx, :, :], codex_std2[idx, :, :], obs_std3[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff3, cordex_test3 = getstats_2Dsig(
        cordex_mean3[idx, :, :], obs_mean3[idx, :, :], codex_std3[idx, :, :], obs_std3[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff4, cordex_test4 = getstats_2Dsig(
        cordex_mean4[idx, :, :], obs_mean3[idx, :, :], codex_std4[idx, :, :], obs_std3[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)

    model_diff1, model_test1 = getstats_2Dsig(
        model_mean1[idx, :, :], obs_mean3[idx, :, :], model_std1[idx, :, :], obs_std3[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff2, model_test2 = getstats_2Dsig(
        model_mean2[idx, :, :], obs_mean3[idx, :, :], model_std2[idx, :, :], obs_std3[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff3, model_test3 = getstats_2Dsig(
        model_mean3[idx, :, :], obs_mean3[idx, :, :], model_std3[idx, :, :], obs_std3[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff4, model_test4 = getstats_2Dsig(
        model_mean4[idx, :, :], obs_mean3[idx, :, :], model_std4[idx, :, :], obs_std3[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' bias (GPCC)'
    if idx != 12:
        fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_SEA_monthly_mean_contour_vs_cordex_refGPCC_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_SEA_annual_mean_contour_vs_cordex_refGPCC.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=1, sig_test=plot_test)

    # vs APHRODITE
    cordex_diff1, cordex_test1 = getstats_2Dsig(
        cordex_mean1[idx, :, :], obs_mean5[idx, :, :], codex_std1[idx, :, :], obs_std5[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff2, cordex_test2 = getstats_2Dsig(
        cordex_mean2[idx, :, :], obs_mean5[idx, :, :], codex_std2[idx, :, :], obs_std5[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff3, cordex_test3 = getstats_2Dsig(
        cordex_mean3[idx, :, :], obs_mean5[idx, :, :], codex_std3[idx, :, :], obs_std5[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff4, cordex_test4 = getstats_2Dsig(
        cordex_mean4[idx, :, :], obs_mean5[idx, :, :], codex_std4[idx, :, :], obs_std5[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)

    model_diff1, model_test1 = getstats_2Dsig(
        model_mean1[idx, :, :], obs_mean5[idx, :, :], model_std1[idx, :, :], obs_std5[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff2, model_test2 = getstats_2Dsig(
        model_mean2[idx, :, :], obs_mean5[idx, :, :], model_std2[idx, :, :], obs_std5[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff3, model_test3 = getstats_2Dsig(
        model_mean3[idx, :, :], obs_mean5[idx, :, :], model_std3[idx, :, :], obs_std5[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff4, model_test4 = getstats_2Dsig(
        model_mean4[idx, :, :], obs_mean5[idx, :, :], model_std4[idx, :, :], obs_std5[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' bias (APHRODITE)'
    if idx != 12:
        fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_SEA_monthly_mean_contour_vs_cordex_refAPHRODITE_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_SEA_annual_mean_contour_vs_cordex_refAPHRODITE.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=1, sig_test=plot_test)

    # vs TRMM
    cordex_diff1, cordex_test1 = getstats_2Dsig(
        cordex_mean1[idx, :, :], obs_mean6[idx, :, :], codex_std1[idx, :, :], obs_std6[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff2, cordex_test2 = getstats_2Dsig(
        cordex_mean2[idx, :, :], obs_mean6[idx, :, :], codex_std2[idx, :, :], obs_std6[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff3, cordex_test3 = getstats_2Dsig(
        cordex_mean3[idx, :, :], obs_mean6[idx, :, :], codex_std3[idx, :, :], obs_std6[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    cordex_diff4, cordex_test4 = getstats_2Dsig(
        cordex_mean4[idx, :, :], obs_mean6[idx, :, :], codex_std4[idx, :, :], obs_std6[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)

    model_diff1, model_test1 = getstats_2Dsig(
        model_mean1[idx, :, :], obs_mean6[idx, :, :], model_std1[idx, :, :], obs_std6[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff2, model_test2 = getstats_2Dsig(
        model_mean2[idx, :, :], obs_mean6[idx, :, :], model_std2[idx, :, :], obs_std6[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff3, model_test3 = getstats_2Dsig(
        model_mean3[idx, :, :], obs_mean6[idx, :, :], model_std3[idx, :, :], obs_std6[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)
    model_diff4, model_test4 = getstats_2Dsig(
        model_mean4[idx, :, :], obs_mean6[idx, :, :], model_std4[idx, :, :], obs_std6[idx, :, :], (endyear-iniyear)+1, (endyear-iniyear)+1)

    print(model_mean4[idx, :, :])
    print(obs_mean6[idx, :, :])
    print(model_diff4)

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' bias (TRMM)'
    if idx != 12:
        fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_SEA_monthly_mean_contour_vs_cordex_refTRMM_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_prect_'+str(iniyear)+'to'+str(endyear)+'_SEA_annual_mean_contour_vs_cordex_refTRMM.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=1, sig_test=plot_test)
