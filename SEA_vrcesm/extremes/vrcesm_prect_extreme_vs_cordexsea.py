# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-calculate extreme
# S3-plot contour
#
# Written by Harry Li

# import libraries
from mod_dataread_obs_pre import readobs_pre_day
from mod_stats_extremes import climdex_RxNday
from mod_dataread_vrcesm import readvrcesm
from mod_dataread_cordex_sea import readcordex
from mod_plt_contour import plot_2Dcontour

import datetime as datetime
import math as math
import pandas as pd
import matplotlib.cm as cm
import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc
from mpl_toolkits import basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# import modules

############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/extremes/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# time bounds
iniyear = 1980
endyear = 2005
yearts = np.arange(iniyear, endyear+1)
# yearts    = np.delete(yearts,9,None)
print(yearts)

# define regions
latbounds = [5, 25]
lonbounds = [95, 120]

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]

# set data frequency
frequency = 'day'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# percentile
percentile = 99

# RxNdays parameter
ndays = 1
freq = 'annual'

# set up plot parameter
legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES', 'ERA-interim', 'APHRODITE-MA', 'CPC']
colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod',
          'darkcyan', 'darkmagenta', 'black', 'brown', 'midnightblue']
line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-.', '-.', '-.', '-.', '-', '-', '-']

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5', 'ERA-interim', 'APHRODITE-MA', 'CPC']
cesm_colors = ['red', 'yellow', 'green', 'blue', 'black', 'brown', 'midnightblue']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-', '-', '-']


############################################################################
# read data
############################################################################

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'prec'
case = 'vrseasia_AMIP_1979_to_2005'
print('Dataset: '+case+'...')
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
print('Dataset: '+case+'...')
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
print('Dataset: '+case+'...')
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
print('Dataset: '+case+'...')
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

# print(model_var1.shape)

print('Reading CORDEX-SEA data...')

# read cordex
project = 'SEA-22'
varname = 'pr'
cordex_models = ['ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-LR', 'MPI-M-MPI-ESM-MR', 'MOHC-HadGEM2-ES']

modelname = 'ICHEC-EC-EARTH'
print('Dataset: '+modelname+'...')
cordex_var1, cordex_time1, cordex_lats1, cordex_lons1 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
modelname = 'IPSL-IPSL-CM5A-LR'
print('Dataset: '+modelname+'...')
cordex_var2, cordex_time2, cordex_lats2, cordex_lons2 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
modelname = 'MPI-M-MPI-ESM-MR'
print('Dataset: '+modelname+'...')
cordex_var3, cordex_time3, cordex_lats3, cordex_lons3 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
modelname = 'MOHC-HadGEM2-ES'
print('Dataset: '+modelname+'...')
cordex_var4, cordex_time4, cordex_lats4, cordex_lons4 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)

# print(cordex_var1.shape)
# print(cordex_var4.shape)

print('Reading observational data...')

project = 'ERA-interim'
print('Dataset: '+project+'...')
obs_var1, obs_time1, obs_lats1, obs_lons1 = readobs_pre_day(
    project, iniyear, endyear, latbounds, lonbounds)

project = 'APHRODITE'
print('Dataset: '+project+'...')
obs_var2, obs_time2, obs_lats2, obs_lons2 = readobs_pre_day(
    project, iniyear, endyear, latbounds, lonbounds)

project = 'CPC'
print('Dataset: '+project+'...')
obs_var3, obs_time3, obs_lats3, obs_lons3 = readobs_pre_day(
    project, iniyear, endyear, latbounds, lonbounds)

# print(obs_var1.shape)

# calculate the precip maximum
# print(model_var1.shape)
# print(model_var1[:, 0, 0])
# print(model_var1[:, 0, 1])
# temp_var1 = model_var1.reshape((len(model_time1), len(model_lats1)*len(model_lons1)))
# print(temp_var1.shape)
# print(temp_var1[:, 0])
# print(temp_var1[:, 1])

print('Calculating Rx'+str(ndays)+'day for '+legends[0]+'...')
model_max1, model_idxmax1 = climdex_RxNday(model_var1, model_time1, ndays, freq=freq)
print('Calculating Rx'+str(ndays)+'day for '+legends[1]+'...')
model_max2, model_idxmax2 = climdex_RxNday(model_var2, model_time2, ndays, freq=freq)
print('Calculating Rx'+str(ndays)+'day for '+legends[2]+'...')
model_max3, model_idxmax3 = climdex_RxNday(model_var3, model_time3, ndays, freq=freq)
print('Calculating Rx'+str(ndays)+'day for '+legends[3]+'...')
model_max4, model_idxmax4 = climdex_RxNday(model_var4, model_time4, ndays, freq=freq)

print('Calculating Rx'+str(ndays)+'day for '+legends[4]+'...')
cordex_max1, cordex_idxmax1 = climdex_RxNday(cordex_var1, cordex_time1, ndays, freq=freq)
print('Calculating Rx'+str(ndays)+'day for '+legends[5]+'...')
cordex_max2, cordex_idxmax2 = climdex_RxNday(cordex_var2, cordex_time2, ndays, freq=freq)
print('Calculating Rx'+str(ndays)+'day for '+legends[6]+'...')
cordex_max3, cordex_idxmax3 = climdex_RxNday(cordex_var3, cordex_time3, ndays, freq=freq)
print('Calculating Rx'+str(ndays)+'day for '+legends[7]+'...')
cordex_max4, cordex_idxmax4 = climdex_RxNday(cordex_var4, cordex_time4, ndays, freq=freq)

print('Calculating Rx'+str(ndays)+'day for '+legends[8]+'...')
obs_max1, obs_idxmax1 = climdex_RxNday(obs_var1, obs_time1, ndays, freq=freq)
print('Calculating Rx'+str(ndays)+'day for '+legends[9]+'...')
obs_max2, obs_idxmax2 = climdex_RxNday(obs_var2, obs_time2, ndays, freq=freq)
print('Calculating Rx'+str(ndays)+'day for '+legends[10]+'...')
obs_max3, obs_idxmax3 = climdex_RxNday(obs_var3, obs_time3, ndays, freq=freq)

############################################################################
# calculate and plot RxNday contour
############################################################################
# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# calculate mean contour
model_max1 = np.mean(model_max1, axis=0)
model_max2 = np.mean(model_max2, axis=0)
model_max3 = np.mean(model_max3, axis=0)
model_max4 = np.mean(model_max4, axis=0)

cordex_max1 = np.mean(cordex_max1, axis=0)
cordex_max2 = np.mean(cordex_max2, axis=0)
cordex_max3 = np.mean(cordex_max3, axis=0)
cordex_max4 = np.mean(cordex_max4, axis=0)

obs_max1 = np.mean(obs_max1, axis=0)
obs_max2 = np.mean(obs_max2, axis=0)
obs_max3 = np.mean(obs_max3, axis=0)

# plot model only
plot_data = [model_max1, model_max2, model_max3, model_max4,
             cordex_max1, cordex_max2, cordex_max3, cordex_max4]
plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
             cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
             cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

model_clevs = np.arange(80, 310, 10)
colormap = cm.YlGnBu
title = str(iniyear)+'-'+str(endyear)+' '+str(ndays)+'-day consecutive maximum '+varname
fname = 'vrseasia_prect_SEA_extreme_contour_vs_cordex_Rx'+str(ndays)+'day.pdf'
plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, model_clevs, legends, lonbounds,
               latbounds, varname, var_unit, title, outdir+fname, opt=0)

# vs ERA_interim
plot_data = []
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
plot_data.append(model_max1 - basemap.interp(obs_max1,
                                             obs_lons1, obs_lats1, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
plot_data.append(model_max2 - basemap.interp(obs_max1,
                                             obs_lons1, obs_lats1, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
plot_data.append(model_max3 - basemap.interp(obs_max1,
                                             obs_lons1, obs_lats1, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
plot_data.append(model_max4 - basemap.interp(obs_max1,
                                             obs_lons1, obs_lats1, lonsout, latsout, order=1))

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
plot_data.append(cordex_max1 - basemap.interp(obs_max1,
                                              obs_lons1, obs_lats1, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
plot_data.append(cordex_max2 - basemap.interp(obs_max1,
                                              obs_lons1, obs_lats1, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
plot_data.append(cordex_max3 - basemap.interp(obs_max1,
                                              obs_lons1, obs_lats1, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
plot_data.append(cordex_max4 - basemap.interp(obs_max1,
                                              obs_lons1, obs_lats1, lonsout, latsout, order=1))

clevs = np.arange(-60, 65, 5)
colormap = cm.BrBG
title = str(iniyear)+'-'+str(endyear)+' '+str(ndays) + \
    '-day consecutive maximum '+varname + 'bias(ERA_interim)'
fname = 'vrseasia_prect_SEA_extreme_contour_vs_cordex_Rx'+str(ndays)+'day_refERAinterim.pdf'
plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
               latbounds, varname, var_unit, title, outdir+fname, opt=0)

# vs APHRODITE
plot_data = []
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
plot_data.append(model_max1 - basemap.interp(obs_max2,
                                             obs_lons2, obs_lats2, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
plot_data.append(model_max2 - basemap.interp(obs_max2,
                                             obs_lons2, obs_lats2, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
plot_data.append(model_max3 - basemap.interp(obs_max2,
                                             obs_lons2, obs_lats2, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
plot_data.append(model_max4 - basemap.interp(obs_max2,
                                             obs_lons2, obs_lats2, lonsout, latsout, order=1))

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
plot_data.append(cordex_max1 - basemap.interp(obs_max2,
                                              obs_lons2, obs_lats2, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
plot_data.append(cordex_max2 - basemap.interp(obs_max2,
                                              obs_lons2, obs_lats2, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
plot_data.append(cordex_max3 - basemap.interp(obs_max2,
                                              obs_lons2, obs_lats2, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
plot_data.append(cordex_max4 - basemap.interp(obs_max2,
                                              obs_lons2, obs_lats2, lonsout, latsout, order=1))

title = str(iniyear)+'-'+str(endyear)+' '+str(ndays) + \
    '-day consecutive maximum '+varname + 'bias(APHRODITE-MA)'
fname = 'vrseasia_prect_SEA_extreme_contour_vs_cordex_Rx'+str(ndays)+'day_refAPHRODITE.pdf'
plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
               latbounds, varname, var_unit, title, outdir+fname, opt=0)

# vs CPC
plot_data = []
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
plot_data.append(model_max1 - basemap.interp(obs_max3,
                                             obs_lons3, obs_lats3, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
plot_data.append(model_max2 - basemap.interp(obs_max3,
                                             obs_lons3, obs_lats3, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
plot_data.append(model_max3 - basemap.interp(obs_max3,
                                             obs_lons3, obs_lats3, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
plot_data.append(model_max4 - basemap.interp(obs_max3,
                                             obs_lons3, obs_lats3, lonsout, latsout, order=1))

lonsout, latsout = np.meshgrid(cordex_lons1, cordex_lats1)
plot_data.append(cordex_max1 - basemap.interp(obs_max3,
                                              obs_lons3, obs_lats3, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(cordex_lons2, cordex_lats2)
plot_data.append(cordex_max2 - basemap.interp(obs_max3,
                                              obs_lons3, obs_lats3, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(cordex_lons3, cordex_lats3)
plot_data.append(cordex_max3 - basemap.interp(obs_max3,
                                              obs_lons3, obs_lats3, lonsout, latsout, order=1))
lonsout, latsout = np.meshgrid(cordex_lons4, cordex_lats4)
plot_data.append(cordex_max4 - basemap.interp(obs_max3,
                                              obs_lons3, obs_lats3, lonsout, latsout, order=1))

title = str(iniyear)+'-'+str(endyear)+' '+str(ndays) + \
    '-day consecutive maximum '+varname + 'bias(CPC)'
fname = 'vrseasia_prect_SEA_extreme_contour_vs_cordex_Rx'+str(ndays)+'day_refCPC.pdf'
plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
               latbounds, varname, var_unit, title, outdir+fname, opt=0)
