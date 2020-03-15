# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-calculate extreme
# S3-plot contour
#
# Written by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.stats.mod_stats_clim import mon2clim
from modules.plot.mod_plt_multidatasets import plt_multidatasets_2Dcontour_mean, plt_multidatasets_2Dcontour_bias
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
outdircordex = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/tmp/tasmin/VRseasia_vs_CORDEX_SEA/'
outdircesm = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/tmp/tasmin/'
outdirnotes = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/notes/tmp/tasmin/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Minium 2m Temperature'
varstr = 'tasmin'
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
varname = 'tasmin'
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

varname = 'TREFHTMN'

resolution = 'fv02'
varfname = 'TS_mean_from_h1'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'TS_mean_from_h1'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'TS_mean_from_h1'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv19'
varfname = 'TS_mean_from_h1'
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
project = 'CRU'
obs_var1, obs_time1, obs_lats1, obs_lons1 = readobs_tmp_CRU('tasmin', iniyear, endyear, latbounds, lonbounds)

# read CPC
project = 'CPC'
obs_var2, obs_time2, obs_lats2, obs_lons2 = readobs_tmp_CPC('tasmin', iniyear, endyear, frequency, latbounds, lonbounds)

obs_var1[obs_var1.mask] = np.nan
obs_var2[obs_var2.mask] = np.nan

# find regional lat/lon boundaries
obs_latl1 = np.argmin(np.abs(obs_lats1 - reg_lats[0]))
obs_latu1 = np.argmin(np.abs(obs_lats1 - reg_lats[1]))
obs_lonl1 = np.argmin(np.abs(obs_lons1 - reg_lons[0]))
obs_lonr1 = np.argmin(np.abs(obs_lons1 - reg_lons[1]))

obs_latl2 = np.argmin(np.abs(obs_lats2 - reg_lats[0]))
obs_latu2 = np.argmin(np.abs(obs_lats2 - reg_lats[1]))
obs_lonl2 = np.argmin(np.abs(obs_lons2 - reg_lons[0]))
obs_lonr2 = np.argmin(np.abs(obs_lons2 - reg_lons[1]))

# print(obs_var1[0, obs_latl1: obs_latu1 + 1, obs_lonl1: obs_lonr1 + 1])
# print(obs_lats1[obs_latl1 : obs_latu1 + 1])
print(obs_var2.shape)

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

print(obs_var2[0, :, :])

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

# print(obs_mean1)

plot_data = [model_mean1, model_mean2, model_mean3, model_mean4, cordex_mean1,
             cordex_mean2, cordex_mean3, cordex_mean4, obs_mean1, obs_mean2]

plot_err = [model_std1, model_std2, model_std3, model_std4, cordex_std1,
            cordex_std2, cordex_std3, cordex_std4, obs_std1, obs_std2]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES',
           'CRU-TS4.03', 'CPC']

colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod',
          'darkcyan', 'darkmagenta', 'black', 'brown', 'midnightblue', 'darkslategray', 'darkred']
line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-.', '-.', '-.', '-.', '-', '-']


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

plt.suptitle('VRCESM vs CORDEX-SEA averaged '+varname+' climatology', fontsize=9, y=0.95)

fname = 'vrseasia_'+varstr+'_mainSEA_clim_line_vs_cordex_overland.pdf'
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

plt.suptitle('VRCESM vs CORDEX-SEA averaged '+varname+' climatology', fontsize=9, y=0.95)

fname = 'vrseasia_'+varstr+'_mainSEA_clim_bar_vs_cordex_overland.pdf'
plt.savefig(outdircordex+fname, bbox_inches='tight')

############################################################################
# plot only for cesm
plot_data = [model_mean1, model_mean2, model_mean3, model_mean4, obs_mean1, obs_mean2]

plot_err = [model_std1, model_std2, model_std3, model_std4, obs_std1, obs_std2]

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5', 'CRU-TS4.03', 'CPC']

cesm_colors = ['red', 'yellow', 'green', 'blue', 'black', 'brown', 'midnightblue', 'darkslategray', 'darkred']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-', '-']

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

plt.suptitle('CESM averaged '+varname+' climatology', fontsize=9, y=0.95)

fname = 'vrseasia_'+varstr+'_mainSEA_clim_line_overland.pdf'
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

plt.suptitle('CESM averaged '+varname+' climatology', fontsize=9, y=0.95)

fname = 'vrseasia_'+varstr+'_mainSEA_clim_bar_overland.pdf'
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

    plot_data.extend([model_mean1, model_mean2, model_mean3, model_mean4, cordex_mean1,
                      cordex_mean2, cordex_mean3, cordex_mean4, obs_mean1, obs_mean2])
    plot_err.extend([model_std1, model_std2, model_std3, model_std4, cordex_std1,
                     cordex_std2, cordex_std3, cordex_std4, obs_std1, obs_std2])
    plot_cesm_data.extend([model_mean1, model_mean2, model_mean3,
                           model_mean4, obs_mean1, obs_mean2])
    plot_cesm_err.extend([model_std1, model_std2, model_std3,
                          model_std4, obs_std1, obs_std2])


# bar plot
fig = plt.figure()
ax = fig.add_subplot(111)

ndatasets = len(legends)
bar_width = 0.85/ndatasets
index = np.arange(1, 5) - bar_width*(ndatasets/2-0.5)
opacity = 0.8

shape_type = ['', '', '', '', '..', '..', '..', '..', '//', '//']

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

plt.suptitle('VRCESM vs CORDEX-SEA seasonal '+varname, fontsize=9, y=0.95)

fname = 'vrseasia_'+varstr+'_mainSEA_seasonal_mean_bar_vs_cordex_overland.pdf'
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

plt.suptitle('CESM seasonal '+varname, fontsize=9, y=0.95)

fname = 'vrseasia_'+varstr+'_mainSEA_seasonal_mean_bar_overland.pdf'
plt.savefig(outdircesm+fname, bbox_inches='tight')


############################################################################
# calculate and plot monthly and seasonal mean contour
############################################################################
# calculate monthly and seasonal mean contour
print('Plotting for mean contour for all datasets...')

# variable info
varname = 'Minium 2m Temperature'
varstr = 'tasmin'
var_unit = r'$^{\circ}C$'

# plot for cesm and cordex
project = 'vrseasia'
region = 'SEA'

model_vars = [model_var1, model_var2, model_var3, model_var4,
              cordex_var1, cordex_var2, cordex_var3, cordex_var4,
              obs_var1, obs_var2]

model_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
              cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4,
              obs_lons1, obs_lons2]

model_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
              cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4,
              obs_lats1, obs_lats2]

model_masks = [model_mask1, model_mask2, model_mask3, model_mask4,
               cordex_mask1, cordex_mask2, cordex_mask3, cordex_mask4,
               obs_mask1, obs_mask2]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES',
           'CRU-TS4.03', 'CPC']

mean_colormap = cm.rainbow
mean_clevs = np.arange(10, 34, 2)

var_colormap = cm.Spectral_r
var_clevs = np.arange(0.2, 2.2, 0.2)

outdir = outdircordex
fname = 'contour_vs_cordex'

print('For cesm and cordex...')
plt_multidatasets_2Dcontour_mean(project, region, iniyear, endyear, varname, varstr, var_unit, model_vars, model_lons, model_lats, lonbounds,
                                 latbounds, legends, mean_colormap, mean_clevs, var_colormap, var_clevs, outdir, fname, model_masks=model_masks)

# plot for cesm only
model_vars = [model_var1, model_var2, model_var3, model_var4,
              obs_var1, obs_var2]

model_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
              obs_lons1, obs_lons2]

model_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
              obs_lats1, obs_lats2]

model_masks = [model_mask1, model_mask2, model_mask3, model_mask4,
               obs_mask1, obs_mask2]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5',
           'CRU-TS4.03', 'CPC']

mean_colormap = cm.rainbow
mean_clevs = np.arange(10, 34, 2)

var_colormap = cm.Spectral_r
var_clevs = np.arange(0.2, 2.2, 0.2)

outdir = outdircesm
fname = 'contour'

print('For cesm only...')
plt_multidatasets_2Dcontour_mean(project, region, iniyear, endyear, varname, varstr, var_unit, model_vars, model_lons, model_lats, lonbounds,
                                 latbounds, legends, mean_colormap, mean_clevs, var_colormap, var_clevs, outdir, fname, model_masks=model_masks)

############################################################################
# calculate and plot monthly and seasonal mean differences
############################################################################
# calculate monthly and seasonal mean differences
print('Plotting for mean differences ...')

# variable info
varname = 'Minium 2m Temperature'
varstr = 'tasmin'
var_unit = r'$\deg C$'

# plot for cesm and cordex
project = 'vrseasia'
region = 'SEA'

print('For cesm and cordex...')
############################################################################
# plot against CRU
reference = 'CRU'
referencestr = 'CRU'

model_vars = [model_var1, model_var2, model_var3, model_var4,
              cordex_var1, cordex_var2, cordex_var3, cordex_var4]

model_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
              cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]

model_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
              cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

model_masks = [model_mask1, model_mask2, model_mask3, model_mask4,
               cordex_mask1, cordex_mask2, cordex_mask3, cordex_mask4]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

diff_colormap = cm.RdBu_r
diff_clevs = np.arange(-5., 5.5, 0.5)

var_colormap = cm.RdBu_r
var_clevs = np.arange(-1., 1.2, 0.2)

outdir = outdircordex
fname = 'contour_vs_cordex'

plt_multidatasets_2Dcontour_bias(project, region, iniyear, endyear, varname, varstr, var_unit, model_vars, model_lons, model_lats, reference, referencestr,
                                 obs_var1, obs_lons1, obs_lats1, lonbounds, latbounds, legends, diff_colormap, diff_clevs, var_colormap, var_clevs, outdir, fname, model_masks=model_masks)

############################################################################
# plot against CPC
reference = 'CPC'
referencestr = 'CPC'

plt_multidatasets_2Dcontour_bias(project, region, iniyear, endyear, varname, varstr, var_unit, model_vars, model_lons, model_lats, reference, referencestr,
                                 obs_var2, obs_lons2, obs_lats2, lonbounds, latbounds, legends, diff_colormap, diff_clevs, var_colormap, var_clevs, outdir, fname, model_masks=model_masks)

############################################################################
# plot for cesm only
print('For cesm only...')
# plot against CRU
reference = 'CRU'
referencestr = 'CRU'

model_vars = [model_var1, model_var2, model_var3, model_var4]

model_lons = [model_lons1, model_lons2, model_lons3, model_lons4]

model_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

model_masks = [model_mask1, model_mask2, model_mask3, model_mask4]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

diff_colormap = cm.RdBu_r
diff_clevs = np.arange(-5., 5.5, 0.5)

var_colormap = cm.RdBu_r
var_clevs = np.arange(-1., 1.2, 0.2)

outdir = outdircesm
fname = 'contour'

plt_multidatasets_2Dcontour_bias(project, region, iniyear, endyear, varname, varstr, var_unit, model_vars, model_lons, model_lats, reference, referencestr,
                                 obs_var1, obs_lons1, obs_lats1, lonbounds, latbounds, legends, diff_colormap, diff_clevs, var_colormap, var_clevs, outdir, fname, model_masks=model_masks)

# plot against CPC
reference = 'CPC'
referencestr = 'CPC'

plt_multidatasets_2Dcontour_bias(project, region, iniyear, endyear, varname, varstr, var_unit, model_vars, model_lons, model_lats, reference, referencestr,
                                 obs_var2, obs_lons2, obs_lats2, lonbounds, latbounds, legends, diff_colormap, diff_clevs, var_colormap, var_clevs, outdir, fname, model_masks=model_masks)
