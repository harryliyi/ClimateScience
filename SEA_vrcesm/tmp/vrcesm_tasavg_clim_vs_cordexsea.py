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
from modules.stats.mod_stats_clim import mon2clim, getstats_2D_ttest, getstats_2D_ftest
from modules.datareader.mod_dataread_obs_tmp import readobs_tmp_mon
from modules.datareader.mod_dataread_obs_CPC import readobs_tmp_CPC
from modules.datareader.mod_dataread_obs_CRU import readobs_tmp_CRU
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# import modules

############################################################################
# setup directory
############################################################################
outdircordex = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/tmp/tasavg/VRseasia_vs_CORDEX_SEA/'
outdircesm = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/tmp/tasavg/'
outdirnotes = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/notes/tmp/tasavg/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Mean 2m Temperature'
varstr = 'tasavg'
var_unit = r'$^{\circ}C$'

# time bounds
iniyear = 1980
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
varname = 'tas'
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

cordex_var1[cordex_var1.mask] = np.nan
cordex_var2[cordex_var2.mask] = np.nan
cordex_var3[cordex_var3.mask] = np.nan
cordex_var4[cordex_var4.mask] = np.nan

# convert to degree Celsius
cordex_var1 = cordex_var1 - 273.15
cordex_var2 = cordex_var2 - 273.15
cordex_var3 = cordex_var3 - 273.15
cordex_var4 = cordex_var4 - 273.15

print(cordex_var1[0, :, :])

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

varname = 'TREFHT'

resolution = 'fv02'
varfname = 'TS'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'TS'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'TS'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv19'
varfname = 'TS'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

model_var1[model_var1.mask] = np.nan
model_var2[model_var2.mask] = np.nan
model_var3[model_var3.mask] = np.nan
model_var4[model_var4.mask] = np.nan

# convert to degree Celsius
model_var1 = model_var1 - 273.15
model_var2 = model_var2 - 273.15
model_var3 = model_var3 - 273.15
model_var4 = model_var4 - 273.15

print(model_var1[0, :, :])


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
obs_var1, obs_time1, obs_lats1, obs_lons1 = readobs_tmp_CRU(
    'tasavg', iniyear, endyear, latbounds, lonbounds)

# read CRU
project = 'GHCN-CAMS'
obs_var2, obs_time2, obs_lats2, obs_lons2 = readobs_tmp_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read University of Delaware
project = 'University of Delaware'
obs_var3, obs_time3, obs_lats3, obs_lons3 = readobs_tmp_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read ERA-interim
project = 'ERA-interim'
obs_var4, obs_time4, obs_lats4, obs_lons4 = readobs_tmp_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read CPC
obs_var5, obs_time5, obs_lats5, obs_lons5 = readobs_tmp_CPC('tasavg', iniyear, endyear, frequency, latbounds, lonbounds)

obs_var1[obs_var1.mask] = np.nan
obs_var2[obs_var2.mask] = np.nan
obs_var3[obs_var3.mask] = np.nan
obs_var4[obs_var4.mask] = np.nan
obs_var5[obs_var5.mask] = np.nan

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

# print(obs_var1[0, obs_latl1: obs_latu1 + 1, obs_lonl1: obs_lonr1 + 1])
# print(obs_lats1[obs_latl1 : obs_latu1 + 1])

# record the mask for each data set
model_mask1 = np.isnan(model_var1[0, :, :])
model_mask2 = np.isnan(model_var2[0, :, :])
model_mask3 = np.isnan(model_var3[0, :, :])
model_mask4 = np.isnan(model_var4[0, :, :])

cordex_mask1 = np.isnan(cordex_var1[0, :, :])
cordex_mask2 = np.isnan(cordex_var2[0, :, :])
cordex_mask3 = np.isnan(cordex_var3[0, :, :])
cordex_mask4 = np.isnan(cordex_var4[0, :, :])

obs_mask1 = np.isnan(obs_var1[0, :, :])
obs_mask2 = np.isnan(obs_var2[0, :, :])
obs_mask3 = np.isnan(obs_var3[0, :, :])
obs_mask4 = np.isnan(obs_var4[0, :, :])
obs_mask5 = np.isnan(obs_var5[0, :, :])

print(obs_mask5)

############################################################################
# calculate climatological mean and plot
############################################################################

print('Plotting for seasonality...')

# calculate seasonality
cordex_mean1, cordex_std1 = mon2clim(
    cordex_var1[:, cordex_latl1: cordex_latu1 + 1, cordex_lonl1: cordex_lonr1 + 1])
cordex_mean2, cordex_std2 = mon2clim(
    cordex_var2[:, cordex_latl2: cordex_latu2 + 1, cordex_lonl2: cordex_lonr2 + 1])
cordex_mean3, cordex_std3 = mon2clim(
    cordex_var3[:, cordex_latl3: cordex_latu3 + 1, cordex_lonl3: cordex_lonr3 + 1])
cordex_mean4, cordex_std4 = mon2clim(
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
obs_mean2, obs_std2 = mon2clim(obs_var2[:, obs_latl2: obs_latu2 + 1, obs_lonl2: obs_lonr2 + 1])
obs_mean3, obs_std3 = mon2clim(obs_var3[:, obs_latl3: obs_latu3 + 1, obs_lonl3: obs_lonr3 + 1])
obs_mean4, obs_std4 = mon2clim(obs_var4[:, obs_latl4: obs_latu4 + 1, obs_lonl4: obs_lonr4 + 1])
obs_mean5, obs_std5 = mon2clim(obs_var5[:, obs_latl5: obs_latu5 + 1, obs_lonl5: obs_lonr5 + 1])
# print(obs_mean1)

plot_data = [model_mean1, model_mean2, model_mean3, model_mean4, cordex_mean1,
             cordex_mean2, cordex_mean3, cordex_mean4, obs_mean1, obs_mean2, obs_mean3,  obs_mean4, obs_mean5]

plot_err = [model_std1, model_std2, model_std3, model_std4, cordex_std1,
            cordex_std2, cordex_std3, cordex_std4, obs_std1, obs_std2, obs_std3, obs_std4, obs_std5]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES',
           'CRU-TS4.03', 'GHCN-CAMS', 'University of Delaware', 'ERA-interim', 'CPC']

colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod',
          'darkcyan', 'darkmagenta', 'black', 'brown', 'midnightblue', 'darkslategray', 'darkred']
line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-.', '-.', '-.', '-.', '-', '-', '-', '-', '-']


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
plt.ylabel(varname+' ('+var_unit+')', fontsize=8)

plt.suptitle('VRCESM vs CORDEX-SEA averaged 2m Temperature climatology', fontsize=9, y=0.95)

fname = 'vrseasia_tasavg_mainSEA_clim_line_vs_cordex_overland.pdf'
plt.savefig(outdircordex+fname, bbox_inches='tight')


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
plt.ylabel(varname+' ('+var_unit+')', fontsize=8)

plt.suptitle('VRCESM vs CORDEX-SEA averaged 2m Temperature climatology', fontsize=9, y=0.95)

fname = 'vrseasia_tasavg_mainSEA_clim_bar_vs_cordex_overland.pdf'
plt.savefig(outdircordex+fname, bbox_inches='tight')

############################################################################
# plot only for cesm
plot_data = [model_mean1, model_mean2, model_mean3, model_mean4, obs_mean1, obs_mean2, obs_mean3,  obs_mean4, obs_mean5]

plot_err = [model_std1, model_std2, model_std3, model_std4, obs_std1, obs_std2, obs_std3, obs_std4, obs_std5]

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5', 'CRU-TS4.03', 'GHCN-CAMS', 'University of Delaware', 'ERA-interim', 'CPC']

cesm_colors = ['red', 'yellow', 'green', 'blue', 'black', 'brown', 'midnightblue', 'darkslategray', 'darkred']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-', '-', '-', '-', '-']

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
plt.ylabel(varname+' ('+var_unit+')', fontsize=8)

plt.suptitle('CESM averaged 2m Temperature climatology', fontsize=9, y=0.95)

fname = 'vrseasia_tasavg_mainSEA_clim_line_overland.pdf'
plt.savefig(outdircesm+fname, bbox_inches='tight')


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
plt.ylabel(varname+' ('+var_unit+')', fontsize=8)

plt.suptitle('CESM averaged 2m Temperature climatology', fontsize=9, y=0.95)

fname = 'vrseasia_tasavg_mainSEA_clim_bar_overland.pdf'
plt.savefig(outdircesm+fname, bbox_inches='tight')


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
    cordex_mean1, cordex_std1 = mon2clim(
        cordex_var1[:, cordex_latl1: cordex_latu1 + 1, cordex_lonl1: cordex_lonr1 + 1], opt=4, season=seasons_list[idx])
    cordex_mean2, cordex_std2 = mon2clim(
        cordex_var2[:, cordex_latl2: cordex_latu2 + 1, cordex_lonl2: cordex_lonr2 + 1], opt=4, season=seasons_list[idx])
    cordex_mean3, cordex_std3 = mon2clim(
        cordex_var3[:, cordex_latl3: cordex_latu3 + 1, cordex_lonl3: cordex_lonr3 + 1], opt=4, season=seasons_list[idx])
    cordex_mean4, cordex_std4 = mon2clim(
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
    obs_mean2, obs_std2 = mon2clim(
        obs_var2[:, obs_latl2: obs_latu2 + 1, obs_lonl2: obs_lonr2 + 1], opt=4, season=seasons_list[idx])
    obs_mean3, obs_std3 = mon2clim(
        obs_var3[:, obs_latl3: obs_latu3 + 1, obs_lonl3: obs_lonr3 + 1], opt=4, season=seasons_list[idx])
    obs_mean4, obs_std4 = mon2clim(
        obs_var4[:, obs_latl4: obs_latu4 + 1, obs_lonl4: obs_lonr4 + 1], opt=4, season=seasons_list[idx])
    obs_mean5, obs_std5 = mon2clim(
        obs_var5[:, obs_latl5: obs_latu5 + 1, obs_lonl5: obs_lonr5 + 1], opt=4, season=seasons_list[idx])

    plot_data.extend([model_mean1, model_mean2, model_mean3, model_mean4, cordex_mean1,
                      cordex_mean2, cordex_mean3, cordex_mean4, obs_mean1, obs_mean2, obs_mean3, obs_mean4, obs_mean5])
    plot_err.extend([model_std1, model_std2, model_std3, model_std4, cordex_std1,
                     cordex_std2, cordex_std3, cordex_std4, obs_std1, obs_std2, obs_std3, obs_std4, obs_std5])
    plot_cesm_data.extend([model_mean1, model_mean2, model_mean3,
                           model_mean4, obs_mean1, obs_mean2, obs_mean3, obs_mean4, obs_mean5])
    plot_cesm_err.extend([model_std1, model_std2, model_std3,
                          model_std4, obs_std1, obs_std2, obs_std3, obs_std4, obs_std5])


# bar plot
fig = plt.figure()
ax = fig.add_subplot(111)

ndatasets = len(legends)
bar_width = 0.85/ndatasets
index = np.arange(1, 5) - bar_width*(ndatasets/2-0.5)
opacity = 0.8

shape_type = ['', '', '', '', '..', '..', '..', '..', '//', '//', '//', '//', '//']

for idx in range(ndatasets):
    plt.bar(index+idx*bar_width, plot_data[idx::ndatasets], bar_width,
            alpha=opacity, color=colors[idx], label=legends[idx], hatch=shape_type[idx])
    plt.errorbar(index+idx*bar_width, plot_data[idx::ndatasets], yerr=plot_err[idx::ndatasets],
                 elinewidth=0.5, ecolor='black', fmt='none', alpha=opacity)

plt.legend(handlelength=4, fontsize=5)

plt.xticks(np.arange(1, 5), seasons_list, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Seasons', fontsize=8)
plt.ylabel(varname+' ('+var_unit+')', fontsize=8)

plt.suptitle('VRCESM vs CORDEX-SEA seasonal mean 2m Temperature', fontsize=9, y=0.95)

fname = 'vrseasia_tasavg_mainSEA_seasonal_mean_bar_vs_cordex_overland.pdf'
plt.savefig(outdircordex+fname, bbox_inches='tight')

# bar plot for cesm only
fig = plt.figure()
ax = fig.add_subplot(111)

ndatasets = len(cesm_legends)
bar_width = 0.85/ndatasets
index = np.arange(1, 5) - bar_width*(ndatasets/2-0.5)
opacity = 0.8

cesm_shape_type = ['', '', '', '', '//', '//', '//', '//', '//']

for idx in range(ndatasets):
    plt.bar(index+idx*bar_width, plot_cesm_data[idx::ndatasets], bar_width, alpha=opacity,
            color=cesm_colors[idx], label=cesm_legends[idx], hatch=cesm_shape_type[idx])
    plt.errorbar(index+idx*bar_width, plot_cesm_data[idx::ndatasets],
                 yerr=plot_cesm_err[idx::ndatasets], elinewidth=0.5, ecolor='black', fmt='none', alpha=opacity)

plt.legend(handlelength=4, fontsize=5)

plt.xticks(np.arange(1, 5), seasons_list, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Seasons', fontsize=8)
plt.ylabel(varname+' ('+var_unit+')', fontsize=8)

plt.suptitle('CESM seasonal mean 2m Temperature', fontsize=9, y=0.95)

fname = 'vrseasia_tasavg_mainSEA_seasonal_mean_bar_overland.pdf'
plt.savefig(outdircesm+fname, bbox_inches='tight')


############################################################################
# calculate and plot monthly mean contour
############################################################################
# calculate monthly mean contour
print('Plotting for monthly mean contour...')

plot_list = monnames
plot_list.append('Annual')

# variable info
varname = '2m Temperature'
varstr = 'tasavg'
var_unit = r'$^{\circ}C$'

# calculate monthly mean
cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=2)
cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=2)
cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=2)
cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=2)

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

obs_mean1, obs_std1 = mon2clim(obs_var1[:, :, :], opt=2)
obs_mean2, obs_std2 = mon2clim(obs_var2[:, :, :], opt=2)
obs_mean3, obs_std3 = mon2clim(obs_var3[:, :, :], opt=2)
obs_mean4, obs_std4 = mon2clim(obs_var4[:, :, :], opt=2)

# mask the variability map
model_std1[np.broadcast_to(model_mask1, model_std1.shape)] = np.nan
model_std2[np.broadcast_to(model_mask2, model_std2.shape)] = np.nan
model_std3[np.broadcast_to(model_mask3, model_std3.shape)] = np.nan
model_std4[np.broadcast_to(model_mask4, model_std4.shape)] = np.nan

cordex_std1[np.broadcast_to(cordex_mask1, cordex_std1.shape)] = np.nan
cordex_std2[np.broadcast_to(cordex_mask2, cordex_std2.shape)] = np.nan
cordex_std3[np.broadcast_to(cordex_mask3, cordex_std3.shape)] = np.nan
cordex_std4[np.broadcast_to(cordex_mask4, cordex_std4.shape)] = np.nan

obs_std1[np.broadcast_to(obs_mask1, obs_std1.shape)] = np.nan
obs_std2[np.broadcast_to(obs_mask2, obs_std2.shape)] = np.nan
obs_std3[np.broadcast_to(obs_mask3, obs_std3.shape)] = np.nan
obs_std4[np.broadcast_to(obs_mask4, obs_std4.shape)] = np.nan


for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean...')

    # plot for both cesm and cordex
    plot_data = [model_mean1[idx, :, :], model_mean2[idx, :, :], model_mean3[idx, :, :], model_mean4[idx, :, :],
                 cordex_mean1[idx, :, :], cordex_mean2[idx, :, :], cordex_mean3[idx, :, :], cordex_mean4[idx, :, :],
                 obs_mean1[idx, :, :], obs_mean2[idx, :, :], obs_mean3[idx, :, :], obs_mean4[idx, :, :]]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4,
                 obs_lons1, obs_lons2, obs_lons3, obs_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4,
                 obs_lats1, obs_lats2, obs_lats3, obs_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-EC-Earth-RegCM4',
               'CORDEX-IPSL-CM5A-RegCM4', 'CORDEX-MPI-ESM-RegCM4', 'CORDEX-HadGEM2-ES-RCA4',
               'CRU-TS4.03', 'GHCN-CAMS', 'University of Delaware', 'ERA-interim']

    clevs = np.arange(10, 34, 2)
    colormap = cm.rainbow

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname

    # without significance level
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_std1[idx, :, :], model_std2[idx, :, :], model_std3[idx, :, :], model_std4[idx, :, :],
                 cordex_std1[idx, :, :], cordex_std2[idx, :, :], cordex_std3[idx, :, :], cordex_std4[idx, :, :],
                 obs_std1[idx, :, :], obs_std2[idx, :, :], obs_std3[idx, :, :], obs_std4[idx, :, :]]

    clevs = np.arange(0.2, 2.2, 0.2)
    colormap = cm.Spectral_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability'

    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_mean1[idx, :, :], model_mean2[idx, :, :], model_mean3[idx, :, :], model_mean4[idx, :, :],
                 obs_mean1[idx, :, :], obs_mean2[idx, :, :], obs_mean3[idx, :, :], obs_mean4[idx, :, :]]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 obs_lons1, obs_lons2, obs_lons3, obs_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 obs_lats1, obs_lats2, obs_lats3, obs_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5',
                    'CRU-TS4.03', 'GHCN-CAMS', 'University of Delaware', 'ERA-interim']

    clevs = np.arange(10, 34, 2)
    colormap = cm.rainbow

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname

    # without significance level
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_std1[idx, :, :], model_std2[idx, :, :], model_std3[idx, :, :], model_std4[idx, :, :],
                 obs_std1[idx, :, :], obs_std2[idx, :, :], obs_std3[idx, :, :], obs_std4[idx, :, :]]

    clevs = np.arange(0.2, 2.2, 0.2)
    colormap = cm.Spectral_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability'

    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)


############################################################################
# calculate and plot monthly mean difference
############################################################################
# calculate monthly mean difference
print('Plotting for monthly mean difference...')

# variable info
varname = '2m Temperature'
varstr = 'tasavg'
var_unit = r'$\deg C$'

############################################################################
# plot against CRU
project = 'CRU'
projectstr = 'CRU'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)

# calculate monthly mean
cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=2)
cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=2)
cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=2)
cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=2)

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=2)
obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=2)
obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=2)
obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=2)

obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=2)
obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=2)
obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=2)
obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=2)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' differences...')

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1[idx, :, :], obs_mean_cordex1[idx, :, :], cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :], expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2[idx, :, :], obs_mean_cordex2[idx, :, :], cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :], expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3[idx, :, :], obs_mean_cordex3[idx, :, :], cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :], expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4[idx, :, :], obs_mean_cordex4[idx, :, :], cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :], expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1[idx, :, :], obs_mean_model1[idx, :, :], model_std1[idx, :, :], obs_std_model1[idx, :, :], expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2[idx, :, :], obs_mean_model2[idx, :, :], model_std2[idx, :, :], obs_std_model2[idx, :, :], expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3[idx, :, :], obs_mean_model3[idx, :, :], model_std3[idx, :, :], obs_std_model3[idx, :, :], expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4[idx, :, :], obs_mean_model4[idx, :, :], model_std4[idx, :, :], obs_std_model4[idx, :, :], expdf, refdf)

    # calculate the variability difference
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :])
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :])
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :])
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :])

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1[idx, :, :], obs_std_model1[idx, :, :])
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2[idx, :, :], obs_std_model2[idx, :, :])
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3[idx, :, :], obs_std_model3[idx, :, :])
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4[idx, :, :], obs_std_model4[idx, :, :])

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)
'''
############################################################################
# plot against GHCN-CAMS
project = 'GHCN-CAMS'
projectstr = 'GHCN-CAMS'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)

# calculate monthly mean
cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=2)
cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=2)
cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=2)
cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=2)

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=2)
obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=2)
obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=2)
obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=2)

obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=2)
obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=2)
obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=2)
obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=2)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' differences...')

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1[idx, :, :], obs_mean_cordex1[idx, :, :], cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :], expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2[idx, :, :], obs_mean_cordex2[idx, :, :], cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :], expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3[idx, :, :], obs_mean_cordex3[idx, :, :], cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :], expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4[idx, :, :], obs_mean_cordex4[idx, :, :], cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :], expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1[idx, :, :], obs_mean_model1[idx, :, :], model_std1[idx, :, :], obs_std_model1[idx, :, :], expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2[idx, :, :], obs_mean_model2[idx, :, :], model_std2[idx, :, :], obs_std_model2[idx, :, :], expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3[idx, :, :], obs_mean_model3[idx, :, :], model_std3[idx, :, :], obs_std_model3[idx, :, :], expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4[idx, :, :], obs_mean_model4[idx, :, :], model_std4[idx, :, :], obs_std_model4[idx, :, :], expdf, refdf)

    # calculate the variability difference
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :])
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :])
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :])
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :])

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1[idx, :, :], obs_std_model1[idx, :, :])
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2[idx, :, :], obs_std_model2[idx, :, :])
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3[idx, :, :], obs_std_model3[idx, :, :])
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4[idx, :, :], obs_std_model4[idx, :, :])

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

############################################################################
# plot against University of Delaware
project = 'University of Delaware'
projectstr = 'UDel'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)

# calculate monthly mean
cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=2)
cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=2)
cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=2)
cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=2)

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=2)
obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=2)
obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=2)
obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=2)

obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=2)
obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=2)
obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=2)
obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=2)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' differences...')

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1[idx, :, :], obs_mean_cordex1[idx, :, :], cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :], expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2[idx, :, :], obs_mean_cordex2[idx, :, :], cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :], expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3[idx, :, :], obs_mean_cordex3[idx, :, :], cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :], expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4[idx, :, :], obs_mean_cordex4[idx, :, :], cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :], expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1[idx, :, :], obs_mean_model1[idx, :, :], model_std1[idx, :, :], obs_std_model1[idx, :, :], expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2[idx, :, :], obs_mean_model2[idx, :, :], model_std2[idx, :, :], obs_std_model2[idx, :, :], expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3[idx, :, :], obs_mean_model3[idx, :, :], model_std3[idx, :, :], obs_std_model3[idx, :, :], expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4[idx, :, :], obs_mean_model4[idx, :, :], model_std4[idx, :, :], obs_std_model4[idx, :, :], expdf, refdf)

    # calculate the variability difference
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :])
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :])
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :])
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :])

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1[idx, :, :], obs_std_model1[idx, :, :])
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2[idx, :, :], obs_std_model2[idx, :, :])
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3[idx, :, :], obs_std_model3[idx, :, :])
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4[idx, :, :], obs_std_model4[idx, :, :])

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

############################################################################
# plot against ERA-interim
project = 'ERA-interim'
projectstr = 'erainterim'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)

# calculate monthly mean
cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=2)
cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=2)
cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=2)
cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=2)

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=2)
obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=2)
obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=2)
obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=2)

obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=2)
obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=2)
obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=2)
obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=2)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' differences...')

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1[idx, :, :], obs_mean_cordex1[idx, :, :], cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :], expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2[idx, :, :], obs_mean_cordex2[idx, :, :], cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :], expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3[idx, :, :], obs_mean_cordex3[idx, :, :], cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :], expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4[idx, :, :], obs_mean_cordex4[idx, :, :], cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :], expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1[idx, :, :], obs_mean_model1[idx, :, :], model_std1[idx, :, :], obs_std_model1[idx, :, :], expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2[idx, :, :], obs_mean_model2[idx, :, :], model_std2[idx, :, :], obs_std_model2[idx, :, :], expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3[idx, :, :], obs_mean_model3[idx, :, :], model_std3[idx, :, :], obs_std_model3[idx, :, :], expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4[idx, :, :], obs_mean_model4[idx, :, :], model_std4[idx, :, :], obs_std_model4[idx, :, :], expdf, refdf)

    # calculate the variability difference
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :])
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :])
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :])
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :])

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1[idx, :, :], obs_std_model1[idx, :, :])
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2[idx, :, :], obs_std_model2[idx, :, :])
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3[idx, :, :], obs_std_model3[idx, :, :])
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4[idx, :, :], obs_std_model4[idx, :, :])

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

############################################################################
# plot against CPC
project = 'CPC'
projectstr = 'CPC'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)

# calculate monthly mean
cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=2)
cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=2)
cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=2)
cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=2)

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=2)
obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=2)
obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=2)
obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=2)

obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=2)
obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=2)
obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=2)
obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=2)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' differences...')

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1[idx, :, :], obs_mean_cordex1[idx, :, :], cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :], expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2[idx, :, :], obs_mean_cordex2[idx, :, :], cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :], expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3[idx, :, :], obs_mean_cordex3[idx, :, :], cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :], expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4[idx, :, :], obs_mean_cordex4[idx, :, :], cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :], expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1[idx, :, :], obs_mean_model1[idx, :, :], model_std1[idx, :, :], obs_std_model1[idx, :, :], expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2[idx, :, :], obs_mean_model2[idx, :, :], model_std2[idx, :, :], obs_std_model2[idx, :, :], expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3[idx, :, :], obs_mean_model3[idx, :, :], model_std3[idx, :, :], obs_std_model3[idx, :, :], expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4[idx, :, :], obs_mean_model4[idx, :, :], model_std4[idx, :, :], obs_std_model4[idx, :, :], expdf, refdf)

    # calculate the variability difference
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1[idx, :, :], obs_std_cordex1[idx, :, :])
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2[idx, :, :], obs_std_cordex2[idx, :, :])
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3[idx, :, :], obs_std_cordex3[idx, :, :])
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4[idx, :, :], obs_std_cordex4[idx, :, :])

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1[idx, :, :], obs_std_model1[idx, :, :])
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2[idx, :, :], obs_std_model2[idx, :, :])
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3[idx, :, :], obs_std_model3[idx, :, :])
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4[idx, :, :], obs_std_model4[idx, :, :])

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' variability differece (Ref as '+project+')'
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_wtsig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_annual_mean_var_contour_ref'+projectstr+'_nosig_.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)
'''

############################################################################
# calculate and plot seasonal mean contour
############################################################################
# calculate monthly mean contour
print('Plotting for seasonal mean contour...')

# variable info
varname = '2m Temperature'
varstr = 'tasavg'
var_unit = r'$^{\circ}C$'

seasons_list = ['DJF', 'MAM', 'JJA', 'SON']
seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8],  'JJAS': [6, 7, 8, 9], 'AMJ': [4, 5, 6], 'SON': [9, 10, 11]}

for idx, iseason in enumerate(seasons_list):
    print('plotting for '+iseason+' mean...')

    # calculate seasonal mean
    cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=3, season=iseason)
    cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=3, season=iseason)
    cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=3, season=iseason)
    cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=3, season=iseason)

    model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=3, season=iseason)
    model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=3, season=iseason)
    model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=3, season=iseason)
    model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=3, season=iseason)

    obs_mean1, obs_std1 = mon2clim(obs_var1[:, :, :], opt=3, season=iseason)
    obs_mean2, obs_std2 = mon2clim(obs_var2[:, :, :], opt=3, season=iseason)
    obs_mean3, obs_std3 = mon2clim(obs_var3[:, :, :], opt=3, season=iseason)
    obs_mean4, obs_std4 = mon2clim(obs_var4[:, :, :], opt=3, season=iseason)

    # mask the variability map
    model_std1[np.broadcast_to(model_mask1, model_std1.shape)] = np.nan
    model_std2[np.broadcast_to(model_mask2, model_std2.shape)] = np.nan
    model_std3[np.broadcast_to(model_mask3, model_std3.shape)] = np.nan
    model_std4[np.broadcast_to(model_mask4, model_std4.shape)] = np.nan

    cordex_std1[np.broadcast_to(cordex_mask1, cordex_std1.shape)] = np.nan
    cordex_std2[np.broadcast_to(cordex_mask2, cordex_std2.shape)] = np.nan
    cordex_std3[np.broadcast_to(cordex_mask3, cordex_std3.shape)] = np.nan
    cordex_std4[np.broadcast_to(cordex_mask4, cordex_std4.shape)] = np.nan

    obs_std1[np.broadcast_to(obs_mask1, obs_std1.shape)] = np.nan
    obs_std2[np.broadcast_to(obs_mask2, obs_std2.shape)] = np.nan
    obs_std3[np.broadcast_to(obs_mask3, obs_std3.shape)] = np.nan
    obs_std4[np.broadcast_to(obs_mask4, obs_std4.shape)] = np.nan

    # plot for both cesm and cordex
    plot_data = [model_mean1, model_mean2, model_mean3, model_mean4,
                 cordex_mean1, cordex_mean2, cordex_mean3, cordex_mean4,
                 obs_mean1, obs_mean2, obs_mean3, obs_mean4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4,
                 obs_lons1, obs_lons2, obs_lons3, obs_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4,
                 obs_lats1, obs_lats2, obs_lats3, obs_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES',
               'CRU-TS4.03', 'GHCN-CAMS', 'University of Delaware', 'ERA-interim']

    clevs = np.arange(10, 34, 2)
    colormap = cm.rainbow

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname

    # without significance level
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_std1, model_std2, model_std3, model_std4,
                 cordex_std1, cordex_std2, cordex_std3, cordex_std4,
                 obs_std1, obs_std2, obs_std3, obs_std4]

    clevs = np.arange(0.2, 2.2, 0.2)
    colormap = cm.Spectral_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability'

    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_mean1, model_mean2, model_mean3, model_mean4,
                 obs_mean1, obs_mean2, obs_mean3, obs_mean4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 obs_lons1, obs_lons2, obs_lons3, obs_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 obs_lats1, obs_lats2, obs_lats3, obs_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5',
                    'CRU-TS4.03', 'GHCN-CAMS', 'University of Delaware', 'ERA-interim']

    clevs = np.arange(10, 34, 2)
    colormap = cm.rainbow

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname

    # without significance level
    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_std1, model_std2, model_std3, model_std4,
                 obs_std1, obs_std2, obs_std3, obs_std4]

    clevs = np.arange(0.2, 2.2, 0.2)
    colormap = cm.Spectral_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability'

    if idx != 12:
        fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

############################################################################
# calculate and plot seasonal mean difference
############################################################################
# calculate seasonal mean difference
print('Plotting for seasonal mean difference...')

# variable info
varname = '2m Temperature'
varstr = 'tasavg'
var_unit = r'$\deg C$'


############################################################################
# plot against CRU
project = 'CRU'
projectstr = 'CRU'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, iseason in enumerate(seasons_list):
    print('plotting for '+iseason+' differences...')

    # calculate seasonal mean
    cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=3, season=iseason)
    cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=3, season=iseason)
    cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=3, season=iseason)
    cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=3, season=iseason)

    model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=3, season=iseason)
    model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=3, season=iseason)
    model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=3, season=iseason)
    model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=3, season=iseason)

    obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=3, season=iseason)
    obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=3, season=iseason)
    obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=3, season=iseason)
    obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=3, season=iseason)

    obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=3, season=iseason)
    obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=3, season=iseason)
    obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=3, season=iseason)
    obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=3, season=iseason)

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1, obs_mean_cordex1, cordex_std1, obs_std_cordex1, expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2, obs_mean_cordex2, cordex_std2, obs_std_cordex2, expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3, obs_mean_cordex3, cordex_std3, obs_std_cordex3, expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4, obs_mean_cordex4, cordex_std4, obs_std_cordex4, expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1, obs_mean_model1, model_std1, obs_std_model1, expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2, obs_mean_model2, model_std2, obs_std_model2, expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3, obs_mean_model3, model_std3, obs_std_model3, expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4, obs_mean_model4, model_std4, obs_std_model4, expdf, refdf)

    # calculate the variability difference and f-test
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1, obs_std_cordex1)
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2, obs_std_cordex2)
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3, obs_std_cordex3)
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4, obs_std_cordex4)

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1, obs_std_model1)
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2, obs_std_model2)
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3, obs_std_model3)
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4, obs_std_model4)

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)
'''
############################################################################
# plot against GHCN-CAMS
project = 'GHCN-CAMS'
projectstr = 'GHCN-CAMS'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, iseason in enumerate(seasons_list):
    print('plotting for '+iseason+' differences...')

    # calculate seasonal mean
    cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=3, season=iseason)
    cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=3, season=iseason)
    cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=3, season=iseason)
    cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=3, season=iseason)

    model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=3, season=iseason)
    model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=3, season=iseason)
    model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=3, season=iseason)
    model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=3, season=iseason)

    obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=3, season=iseason)
    obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=3, season=iseason)
    obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=3, season=iseason)
    obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=3, season=iseason)

    obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=3, season=iseason)
    obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=3, season=iseason)
    obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=3, season=iseason)
    obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=3, season=iseason)

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1, obs_mean_cordex1, cordex_std1, obs_std_cordex1, expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2, obs_mean_cordex2, cordex_std2, obs_std_cordex2, expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3, obs_mean_cordex3, cordex_std3, obs_std_cordex3, expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4, obs_mean_cordex4, cordex_std4, obs_std_cordex4, expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1, obs_mean_model1, model_std1, obs_std_model1, expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2, obs_mean_model2, model_std2, obs_std_model2, expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3, obs_mean_model3, model_std3, obs_std_model3, expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4, obs_mean_model4, model_std4, obs_std_model4, expdf, refdf)

    # calculate the variability difference and f-test
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1, obs_std_cordex1)
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2, obs_std_cordex2)
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3, obs_std_cordex3)
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4, obs_std_cordex4)

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1, obs_std_model1)
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2, obs_std_model2)
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3, obs_std_model3)
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4, obs_std_model4)

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

############################################################################
# plot against University of Delaware
project = 'University of Delaware'
projectstr = 'UDel'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, iseason in enumerate(seasons_list):
    print('plotting for '+iseason+' differences...')

    # calculate seasonal mean
    cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=3, season=iseason)
    cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=3, season=iseason)
    cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=3, season=iseason)
    cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=3, season=iseason)

    model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=3, season=iseason)
    model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=3, season=iseason)
    model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=3, season=iseason)
    model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=3, season=iseason)

    obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=3, season=iseason)
    obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=3, season=iseason)
    obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=3, season=iseason)
    obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=3, season=iseason)

    obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=3, season=iseason)
    obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=3, season=iseason)
    obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=3, season=iseason)
    obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=3, season=iseason)

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1, obs_mean_cordex1, cordex_std1, obs_std_cordex1, expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2, obs_mean_cordex2, cordex_std2, obs_std_cordex2, expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3, obs_mean_cordex3, cordex_std3, obs_std_cordex3, expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4, obs_mean_cordex4, cordex_std4, obs_std_cordex4, expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1, obs_mean_model1, model_std1, obs_std_model1, expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2, obs_mean_model2, model_std2, obs_std_model2, expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3, obs_mean_model3, model_std3, obs_std_model3, expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4, obs_mean_model4, model_std4, obs_std_model4, expdf, refdf)

    # calculate the variability difference and f-test
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1, obs_std_cordex1)
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2, obs_std_cordex2)
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3, obs_std_cordex3)
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4, obs_std_cordex4)

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1, obs_std_model1)
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2, obs_std_model2)
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3, obs_std_model3)
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4, obs_std_model4)

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

############################################################################
# plot against ERA-interim
project = 'ERA-interim'
projectstr = 'erainterim'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, iseason in enumerate(seasons_list):
    print('plotting for '+iseason+' differences...')

    # calculate seasonal mean
    cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=3, season=iseason)
    cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=3, season=iseason)
    cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=3, season=iseason)
    cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=3, season=iseason)

    model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=3, season=iseason)
    model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=3, season=iseason)
    model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=3, season=iseason)
    model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=3, season=iseason)

    obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=3, season=iseason)
    obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=3, season=iseason)
    obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=3, season=iseason)
    obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=3, season=iseason)

    obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=3, season=iseason)
    obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=3, season=iseason)
    obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=3, season=iseason)
    obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=3, season=iseason)

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1, obs_mean_cordex1, cordex_std1, obs_std_cordex1, expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2, obs_mean_cordex2, cordex_std2, obs_std_cordex2, expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3, obs_mean_cordex3, cordex_std3, obs_std_cordex3, expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4, obs_mean_cordex4, cordex_std4, obs_std_cordex4, expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1, obs_mean_model1, model_std1, obs_std_model1, expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2, obs_mean_model2, model_std2, obs_std_model2, expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3, obs_mean_model3, model_std3, obs_std_model3, expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4, obs_mean_model4, model_std4, obs_std_model4, expdf, refdf)

    # calculate the variability difference and f-test
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1, obs_std_cordex1)
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2, obs_std_cordex2)
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3, obs_std_cordex3)
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4, obs_std_cordex4)

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1, obs_std_model1)
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2, obs_std_model2)
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3, obs_std_model3)
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4, obs_std_model4)

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

############################################################################
# plot against CPC
project = 'CPC'
projectstr = 'CPC'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
obs_var_cordex1 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
obs_var_cordex2 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
obs_var_cordex3 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
obs_var_cordex4 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, iseason in enumerate(seasons_list):
    print('plotting for '+iseason+' differences...')

    # calculate seasonal mean
    cordex_mean1, cordex_std1 = mon2clim(cordex_var1[:, :, :], opt=3, season=iseason)
    cordex_mean2, cordex_std2 = mon2clim(cordex_var2[:, :, :], opt=3, season=iseason)
    cordex_mean3, cordex_std3 = mon2clim(cordex_var3[:, :, :], opt=3, season=iseason)
    cordex_mean4, cordex_std4 = mon2clim(cordex_var4[:, :, :], opt=3, season=iseason)

    model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=3, season=iseason)
    model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=3, season=iseason)
    model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=3, season=iseason)
    model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=3, season=iseason)

    obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=3, season=iseason)
    obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=3, season=iseason)
    obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=3, season=iseason)
    obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=3, season=iseason)

    obs_mean_cordex1, obs_std_cordex1 = mon2clim(obs_var_cordex1[:, :, :], opt=3, season=iseason)
    obs_mean_cordex2, obs_std_cordex2 = mon2clim(obs_var_cordex2[:, :, :], opt=3, season=iseason)
    obs_mean_cordex3, obs_std_cordex3 = mon2clim(obs_var_cordex3[:, :, :], opt=3, season=iseason)
    obs_mean_cordex4, obs_std_cordex4 = mon2clim(obs_var_cordex4[:, :, :], opt=3, season=iseason)

    # calculate the mean difference and t-test results
    cordex_diff1, cordex_test1 = getstats_2D_ttest(
        cordex_mean1, obs_mean_cordex1, cordex_std1, obs_std_cordex1, expdf, refdf)
    cordex_diff2, cordex_test2 = getstats_2D_ttest(
        cordex_mean2, obs_mean_cordex2, cordex_std2, obs_std_cordex2, expdf, refdf)
    cordex_diff3, cordex_test3 = getstats_2D_ttest(
        cordex_mean3, obs_mean_cordex3, cordex_std3, obs_std_cordex3, expdf, refdf)
    cordex_diff4, cordex_test4 = getstats_2D_ttest(
        cordex_mean4, obs_mean_cordex4, cordex_std4, obs_std_cordex4, expdf, refdf)

    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1, obs_mean_model1, model_std1, obs_std_model1, expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2, obs_mean_model2, model_std2, obs_std_model2, expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3, obs_mean_model3, model_std3, obs_std_model3, expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4, obs_mean_model4, model_std4, obs_std_model4, expdf, refdf)

    # calculate the variability difference and f-test
    cordex_var_diff1, cordex_var_ftest1 = getstats_2D_ftest(cordex_std1, obs_std_cordex1)
    cordex_var_diff2, cordex_var_ftest2 = getstats_2D_ftest(cordex_std2, obs_std_cordex2)
    cordex_var_diff3, cordex_var_ftest3 = getstats_2D_ftest(cordex_std3, obs_std_cordex3)
    cordex_var_diff4, cordex_var_ftest4 = getstats_2D_ftest(cordex_std4, obs_std_cordex4)

    model_var_diff1, model_var_ftest1 = getstats_2D_ftest(model_std1, obs_std_model1)
    model_var_diff2, model_var_ftest2 = getstats_2D_ftest(model_std2, obs_std_model2)
    model_var_diff3, model_var_ftest3 = getstats_2D_ftest(model_std3, obs_std_model3)
    model_var_diff4, model_var_ftest4 = getstats_2D_ftest(model_std4, obs_std_model4)

    # mask the results
    cordex_diff1[cordex_mask1] = np.nan
    cordex_diff2[cordex_mask2] = np.nan
    cordex_diff3[cordex_mask3] = np.nan
    cordex_diff4[cordex_mask4] = np.nan

    cordex_test1[cordex_mask1] = np.nan
    cordex_test2[cordex_mask2] = np.nan
    cordex_test3[cordex_mask3] = np.nan
    cordex_test4[cordex_mask4] = np.nan

    cordex_var_diff1[cordex_mask1] = np.nan
    cordex_var_diff2[cordex_mask2] = np.nan
    cordex_var_diff3[cordex_mask3] = np.nan
    cordex_var_diff4[cordex_mask4] = np.nan

    cordex_var_ftest1[cordex_mask1] = np.nan
    cordex_var_ftest2[cordex_mask2] = np.nan
    cordex_var_ftest3[cordex_mask3] = np.nan
    cordex_var_ftest4[cordex_mask4] = np.nan

    model_diff1[model_mask1] = np.nan
    model_diff2[model_mask2] = np.nan
    model_diff3[model_mask3] = np.nan
    model_diff4[model_mask4] = np.nan

    model_test1[model_mask1] = np.nan
    model_test2[model_mask2] = np.nan
    model_test3[model_mask3] = np.nan
    model_test4[model_mask4] = np.nan

    model_var_diff1[model_mask1] = np.nan
    model_var_diff2[model_mask2] = np.nan
    model_var_diff3[model_mask3] = np.nan
    model_var_diff4[model_mask4] = np.nan

    model_var_ftest1[model_mask1] = np.nan
    model_var_ftest2[model_mask2] = np.nan
    model_var_ftest3[model_mask3] = np.nan
    model_var_ftest4[model_mask4] = np.nan

    # print(model_test1.shape)
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4,
                 cordex_diff1, cordex_diff2, cordex_diff3, cordex_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4,
                 cordex_test1, cordex_test2, cordex_test3, cordex_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
               'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4,
                 cordex_var_diff1, cordex_var_diff2, cordex_var_diff3, cordex_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4,
                 cordex_var_ftest1, cordex_var_ftest2, cordex_var_ftest3, cordex_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircordex+fname, opt=0)

    # plot for cesm only
    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-5., 5.5, 0.5)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

    # plot for variability
    plot_data = [model_var_diff1, model_var_diff2, model_var_diff3, model_var_diff4]
    plot_test = [model_var_ftest1, model_var_ftest2, model_var_ftest3, model_var_ftest4]

    clevs = np.arange(-1., 1.2, 0.2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+iseason+' mean '+varname+' variability differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test, sig_thres=2.2693)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_seasonal_mean_var_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)
'''
