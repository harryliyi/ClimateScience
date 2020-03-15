# This script is used to pre-calculate the statistics for taylor diagram with vrcesm and CORDEX-SEA
#
# Written by Harry Li

# import libraries
import skill_metrics as sm
from mod_dataread_obs_pre import readobs_pre_mon
from mod_stats_clim import mon2clim
from mod_stats_clustering import kmeans_cluster
from mod_stats_extremes import climdex_RxNday
from mod_plt_bars import plot_bars
from mod_plt_lines import plot_lines
from mod_plt_findstns import data_findstns
from mod_dataread_vrcesm import readvrcesm
from mod_dataread_cordex_sea import readcordex
from mod_plt_regrid import data_regrid

import datetime as datetime
import math as math
import pandas as pd
import matplotlib.cm as cm
import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc
import mpl_toolkits.basemap as basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# import modules

############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/'

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

# percentile
percentile = 99

# RxNdays parameter
ndays = 5
freq = 'annaul'

# set up plot parameter
legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES', 'CRU', 'GPCC', 'GPCP', 'ERA-interim', 'APHRODITE-MA']
colors = ['r', 'r', 'r', 'r', 'b', 'b',
          'b', 'b', 'g', 'g', 'g', 'g']


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
    varname, iniyear, endyear, resolution, varfname, case, frequency, reg_lats, reg_lons, oceanmask=1)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
print('Dataset: '+case+'...')
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, reg_lats, reg_lons, oceanmask=1)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
print('Dataset: '+case+'...')
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, reg_lats, reg_lons, oceanmask=1)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
print('Dataset: '+case+'...')
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, reg_lats, reg_lons, oceanmask=1)

# print(model_var1.shape)


print('Reading CORDEX-SEA data...')

# read cordex
project = 'SEA-22'
varname = 'pr'
cordex_models = ['ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-LR', 'MPI-M-MPI-ESM-MR', 'MOHC-HadGEM2-ES']

modelname = 'ICHEC-EC-EARTH'
print('Dataset: '+modelname+'...')
cordex_var1, cordex_time1, cordex_lats1, cordex_lons1 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, reg_lats, reg_lons, oceanmask=1)
modelname = 'IPSL-IPSL-CM5A-LR'
print('Dataset: '+modelname+'...')
cordex_var2, cordex_time2, cordex_lats2, cordex_lons2 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, reg_lats, reg_lons, oceanmask=1)
modelname = 'MPI-M-MPI-ESM-MR'
print('Dataset: '+modelname+'...')
cordex_var3, cordex_time3, cordex_lats3, cordex_lons3 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, reg_lats, reg_lons, oceanmask=1)
modelname = 'MOHC-HadGEM2-ES'
print('Dataset: '+modelname+'...')
cordex_var4, cordex_time4, cordex_lats4, cordex_lons4 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, reg_lats, reg_lons, oceanmask=1)

# print(cordex_var1.shape)
# print(cordex_var4.shape)

print('Reading observational data...')

# read CRU
project = 'CRU'
print('Dataset: '+project+'...')
obs_var1, obs_time1, obs_lats1, obs_lons1 = readobs_pre_mon(
    project, iniyear, endyear, reg_lats, reg_lons)

# read GPCC
project = 'GPCP'
print('Dataset: '+project+'...')
obs_var2, obs_time2, obs_lats2, obs_lons2 = readobs_pre_mon(
    project, iniyear, endyear, reg_lats, reg_lons)

# read GPCC
project = 'GPCC'
print('Dataset: '+project+'...')
obs_var3, obs_time3, obs_lats3, obs_lons3 = readobs_pre_mon(
    project, iniyear, endyear, reg_lats, reg_lons)

# read ERA-interim
project = 'ERA-interim'
print('Dataset: '+project+'...')
obs_var4, obs_time4, obs_lats4, obs_lons4 = readobs_pre_mon(
    project, iniyear, endyear, reg_lats, reg_lons)

# read APHRODITE
project = 'APHRODITE'
print('Dataset: '+project+'...')
obs_var5, obs_time5, obs_lats5, obs_lons5 = readobs_pre_mon(
    project, iniyear, endyear, reg_lats, reg_lons)


# set reference to CRU
# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)

# regrid data to reference
print('Regridding data...')
model_regrid_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
model_regrid_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
model_regrid_var3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)
model_regrid_var4 = data_regrid(model_var4, model_lons4, model_lats4, lonsout, latsout)

cordex_regrid_var1 = data_regrid(cordex_var1, cordex_lons1, cordex_lats1, lonsout, latsout)
cordex_regrid_var2 = data_regrid(cordex_var2, cordex_lons2, cordex_lats2, lonsout, latsout)
cordex_regrid_var3 = data_regrid(cordex_var3, cordex_lons3, cordex_lats3, lonsout, latsout)
cordex_regrid_var4 = data_regrid(cordex_var4, cordex_lons4, cordex_lats4, lonsout, latsout)

obs_regrid_var1 = data_regrid(obs_var1, obs_lons1, obs_lats1, lonsout, latsout)
obs_regrid_var2 = data_regrid(obs_var2, obs_lons2, obs_lats2, lonsout, latsout)
obs_regrid_var3 = data_regrid(obs_var3, obs_lons3, obs_lats3, lonsout, latsout)
obs_regrid_var4 = data_regrid(obs_var4, obs_lons4, obs_lats4, lonsout, latsout)
obs_regrid_var5 = data_regrid(obs_var5, obs_lons5, obs_lats5, lonsout, latsout)

# mask array
ref_ma_var = obs_regrid_var1.flatten()
model_ma_var1 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_regrid_var1.flatten())
model_ma_var2 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_regrid_var2.flatten())
model_ma_var3 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_regrid_var3.flatten())
model_ma_var4 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_regrid_var4.flatten())

cordex_ma_var1 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_regrid_var1.flatten())
cordex_ma_var2 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_regrid_var2.flatten())
cordex_ma_var3 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_regrid_var3.flatten())
cordex_ma_var4 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_regrid_var4.flatten())

obs_ma_var2 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_regrid_var2.flatten())
obs_ma_var3 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_regrid_var3.flatten())
obs_ma_var4 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_regrid_var4.flatten())
obs_ma_var5 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_regrid_var5.flatten())

# print(ref_var)
# print(len(ref_var))
# print(model_var1)
# print(len(model_var1))

# calculate taylor statistics for 3D variables
taylor_model_stats1 = sm.taylor_statistics(model_ma_var1, ref_ma_var)
taylor_model_stats2 = sm.taylor_statistics(model_ma_var2, ref_ma_var)
taylor_model_stats3 = sm.taylor_statistics(model_ma_var3, ref_ma_var)
taylor_model_stats4 = sm.taylor_statistics(model_ma_var4, ref_ma_var)

taylor_cordex_stats1 = sm.taylor_statistics(cordex_ma_var1, ref_ma_var)
taylor_cordex_stats2 = sm.taylor_statistics(cordex_ma_var2, ref_ma_var)
taylor_cordex_stats3 = sm.taylor_statistics(cordex_ma_var3, ref_ma_var)
taylor_cordex_stats4 = sm.taylor_statistics(cordex_ma_var4, ref_ma_var)

taylor_obs_stats2 = sm.taylor_statistics(obs_ma_var2, ref_ma_var)
taylor_obs_stats3 = sm.taylor_statistics(obs_ma_var3, ref_ma_var)
taylor_obs_stats4 = sm.taylor_statistics(obs_ma_var4, ref_ma_var)
taylor_obs_stats5 = sm.taylor_statistics(obs_ma_var5, ref_ma_var)


# plot for CESMs
ccoef = [1.]
crmsd = [0.]
sdev = [taylor_model_stats1['sdev'][0]]
sdev.append(taylor_model_stats1['sdev'][1])
sdev.append(taylor_model_stats2['sdev'][1])
sdev.append(taylor_model_stats3['sdev'][1])
sdev.append(taylor_model_stats4['sdev'][1])
crmsd.append(taylor_model_stats1['crmsd'][1])
crmsd.append(taylor_model_stats2['crmsd'][1])
crmsd.append(taylor_model_stats3['crmsd'][1])
crmsd.append(taylor_model_stats4['crmsd'][1])
ccoef.append(taylor_model_stats1['ccoef'][1])
ccoef.append(taylor_model_stats2['ccoef'][1])
ccoef.append(taylor_model_stats3['ccoef'][1])
ccoef.append(taylor_model_stats4['ccoef'][1])


# overlay for CORDEX-SEA
sdev.append(taylor_cordex_stats1['sdev'][1])
sdev.append(taylor_cordex_stats2['sdev'][1])
sdev.append(taylor_cordex_stats3['sdev'][1])
sdev.append(taylor_cordex_stats4['sdev'][1])
crmsd.append(taylor_cordex_stats1['crmsd'][1])
crmsd.append(taylor_cordex_stats2['crmsd'][1])
crmsd.append(taylor_cordex_stats3['crmsd'][1])
crmsd.append(taylor_cordex_stats4['crmsd'][1])
ccoef.append(taylor_cordex_stats1['ccoef'][1])
ccoef.append(taylor_cordex_stats2['ccoef'][1])
ccoef.append(taylor_cordex_stats3['ccoef'][1])
ccoef.append(taylor_cordex_stats4['ccoef'][1])

# overlay for other obs
ccoef.append(taylor_obs_stats2['ccoef'][1])
ccoef.append(taylor_obs_stats3['ccoef'][1])
ccoef.append(taylor_obs_stats4['ccoef'][1])
ccoef.append(taylor_obs_stats5['ccoef'][1])
crmsd.append(taylor_obs_stats2['crmsd'][1])
crmsd.append(taylor_obs_stats3['crmsd'][1])
crmsd.append(taylor_obs_stats4['crmsd'][1])
crmsd.append(taylor_obs_stats5['crmsd'][1])
sdev.append(taylor_obs_stats2['sdev'][1])
sdev.append(taylor_obs_stats3['sdev'][1])
sdev.append(taylor_obs_stats4['sdev'][1])
sdev.append(taylor_obs_stats5['sdev'][1])


print(taylor_model_stats1)
print(taylor_model_stats2)
print(sdev)
print(crmsd)
print(ccoef)

sdev = np.array(sdev)
crmsd = np.array(crmsd)
ccoef = np.array(ccoef)
labels = [legends[8]]+legends[0:8]+legends[9:]
print(labels)
sm.taylor_diagram(sdev, crmsd, ccoef, MarkerDisplayed='marker', markerLabel=labels,
                  markerLabelColor='b', markerLegend='on', markerColor='b',
                  styleOBS='-', colOBS='g', markerobs='o',
                  markerSize=5, tickRMS=[0.0, 1.0, 2.0, 3.0, 4.0],
                  tickSTD=[0., 2.5, 5., 10.], showlabelsSTD='on',
                  tickRMSangle=115, showlabelsRMS='on',
                  titleRMS='off', titleOBS='Ref', checkstats='on')

plt.savefig(outdir+'vrseasia_prect_taylor_diagram_monthly_3D_vs_cordexsea_refCRU.pdf')


# calculate taylor statistics for annual mean 2D variables

# calculate mean and regrid
ref_ma_var = np.ma.mean(obs_var1, axis=0)
ref_ma_var = basemap.interp(ref_ma_var, obs_lons1, obs_lats1, lonsout, latsout, order=1)
ref_ma_var = ref_ma_var.flatten()

model_ma_var1 = np.ma.mean(model_var1, axis=0)
model_ma_var1 = basemap.interp(model_ma_var1, model_lons1, model_lats1, lonsout, latsout, order=1)
model_ma_var1 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_ma_var1.flatten())
model_ma_var2 = np.ma.mean(model_var2, axis=0)
model_ma_var2 = basemap.interp(model_ma_var2, model_lons2, model_lats2, lonsout, latsout, order=1)
model_ma_var2 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_ma_var2.flatten())
model_ma_var3 = np.ma.mean(model_var3, axis=0)
model_ma_var3 = basemap.interp(model_ma_var3, model_lons3, model_lats3, lonsout, latsout, order=1)
model_ma_var3 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_ma_var3.flatten())
model_ma_var4 = np.ma.mean(model_var4, axis=0)
model_ma_var4 = basemap.interp(model_ma_var4, model_lons4, model_lats4, lonsout, latsout, order=1)
model_ma_var4 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_ma_var4.flatten())

cordex_ma_var1 = np.ma.mean(cordex_var1, axis=0)
cordex_ma_var1 = basemap.interp(cordex_ma_var1, cordex_lons1,
                                cordex_lats1, lonsout, latsout, order=1)
cordex_ma_var1 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_ma_var1.flatten())
cordex_ma_var2 = np.ma.mean(cordex_var2, axis=0)
cordex_ma_var2 = basemap.interp(cordex_ma_var2, cordex_lons2,
                                cordex_lats2, lonsout, latsout, order=1)
cordex_ma_var2 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_ma_var2.flatten())
cordex_ma_var3 = np.ma.mean(cordex_var3, axis=0)
cordex_ma_var3 = basemap.interp(cordex_ma_var3, cordex_lons3,
                                cordex_lats3, lonsout, latsout, order=1)
cordex_ma_var3 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_ma_var3.flatten())
cordex_ma_var4 = np.ma.mean(cordex_var4, axis=0)
cordex_ma_var4 = basemap.interp(cordex_ma_var4, cordex_lons4,
                                cordex_lats4, lonsout, latsout, order=1)
cordex_ma_var4 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_ma_var4.flatten())

obs_ma_var2 = np.ma.mean(obs_var2, axis=0)
obs_ma_var2 = basemap.interp(obs_ma_var2, obs_lons2, obs_lats2, lonsout, latsout, order=1)
obs_ma_var2 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_ma_var2.flatten())
obs_ma_var3 = np.ma.mean(obs_var3, axis=0)
obs_ma_var3 = basemap.interp(obs_ma_var3, obs_lons3, obs_lats3, lonsout, latsout, order=1)
obs_ma_var3 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_ma_var3.flatten())
obs_ma_var4 = np.ma.mean(obs_var4, axis=0)
obs_ma_var4 = basemap.interp(obs_ma_var4, obs_lons4, obs_lats4, lonsout, latsout, order=1)
obs_ma_var4 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_ma_var4.flatten())
obs_ma_var5 = np.ma.mean(obs_var5, axis=0)
obs_ma_var5 = basemap.interp(obs_ma_var5, obs_lons5, obs_lats5, lonsout, latsout, order=1)
obs_ma_var5 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_ma_var5.flatten())

taylor_model_stats1 = sm.taylor_statistics(model_ma_var1, ref_ma_var)
taylor_model_stats2 = sm.taylor_statistics(model_ma_var2, ref_ma_var)
taylor_model_stats3 = sm.taylor_statistics(model_ma_var3, ref_ma_var)
taylor_model_stats4 = sm.taylor_statistics(model_ma_var4, ref_ma_var)

taylor_cordex_stats1 = sm.taylor_statistics(cordex_ma_var1, ref_ma_var)
taylor_cordex_stats2 = sm.taylor_statistics(cordex_ma_var2, ref_ma_var)
taylor_cordex_stats3 = sm.taylor_statistics(cordex_ma_var3, ref_ma_var)
taylor_cordex_stats4 = sm.taylor_statistics(cordex_ma_var4, ref_ma_var)

taylor_obs_stats2 = sm.taylor_statistics(obs_ma_var2, ref_ma_var)
taylor_obs_stats3 = sm.taylor_statistics(obs_ma_var3, ref_ma_var)
taylor_obs_stats4 = sm.taylor_statistics(obs_ma_var4, ref_ma_var)
taylor_obs_stats5 = sm.taylor_statistics(obs_ma_var5, ref_ma_var)

# plot for CESMs
ccoef = [1.]
crmsd = [0.]
sdev = [taylor_model_stats1['sdev'][0]]
sdev.append(taylor_model_stats1['sdev'][1])
sdev.append(taylor_model_stats2['sdev'][1])
sdev.append(taylor_model_stats3['sdev'][1])
sdev.append(taylor_model_stats4['sdev'][1])
crmsd.append(taylor_model_stats1['crmsd'][1])
crmsd.append(taylor_model_stats2['crmsd'][1])
crmsd.append(taylor_model_stats3['crmsd'][1])
crmsd.append(taylor_model_stats4['crmsd'][1])
ccoef.append(taylor_model_stats1['ccoef'][1])
ccoef.append(taylor_model_stats2['ccoef'][1])
ccoef.append(taylor_model_stats3['ccoef'][1])
ccoef.append(taylor_model_stats4['ccoef'][1])


# overlay for CORDEX-SEA
sdev.append(taylor_cordex_stats1['sdev'][1])
sdev.append(taylor_cordex_stats2['sdev'][1])
sdev.append(taylor_cordex_stats3['sdev'][1])
sdev.append(taylor_cordex_stats4['sdev'][1])
crmsd.append(taylor_cordex_stats1['crmsd'][1])
crmsd.append(taylor_cordex_stats2['crmsd'][1])
crmsd.append(taylor_cordex_stats3['crmsd'][1])
crmsd.append(taylor_cordex_stats4['crmsd'][1])
ccoef.append(taylor_cordex_stats1['ccoef'][1])
ccoef.append(taylor_cordex_stats2['ccoef'][1])
ccoef.append(taylor_cordex_stats3['ccoef'][1])
ccoef.append(taylor_cordex_stats4['ccoef'][1])

# overlay for other obs
ccoef.append(taylor_obs_stats2['ccoef'][1])
ccoef.append(taylor_obs_stats3['ccoef'][1])
ccoef.append(taylor_obs_stats4['ccoef'][1])
ccoef.append(taylor_obs_stats5['ccoef'][1])
crmsd.append(taylor_obs_stats2['crmsd'][1])
crmsd.append(taylor_obs_stats3['crmsd'][1])
crmsd.append(taylor_obs_stats4['crmsd'][1])
crmsd.append(taylor_obs_stats5['crmsd'][1])
sdev.append(taylor_obs_stats2['sdev'][1])
sdev.append(taylor_obs_stats3['sdev'][1])
sdev.append(taylor_obs_stats4['sdev'][1])
sdev.append(taylor_obs_stats5['sdev'][1])


print(taylor_model_stats1)
print(taylor_model_stats2)
print(sdev)
print(crmsd)
print(ccoef)

sdev = np.array(sdev)
crmsd = np.array(crmsd)
ccoef = np.array(ccoef)
labels = [legends[8]]+legends[0:8]+legends[9:]
print(labels)
plt.clf()
sm.taylor_diagram(sdev, crmsd, ccoef, numberPanels=1, MarkerDisplayed='marker', markerLabel=labels,
                  markerLabelColor='b', markerLegend='on', markerColor='b',
                  styleOBS='-', colOBS='g', markerobs='o',
                  markerSize=5, tickRMS=[0.0, 1.0, 2.0, 3.0],
                  tickRMSangle=115, showlabelsRMS='on',
                  titleRMS='off', titleOBS='Ref', checkstats='on')

plt.savefig(outdir+'vrseasia_prect_taylor_diagram_longterm_mean_2D_vs_cordexsea_refCRU.pdf')


# calculate taylor statistics for monthly time series variables
ref_ma_var = np.ma.mean(np.ma.mean(obs_var1, axis=1), axis=1)
ref_ma_var = ref_ma_var.flatten()
model_ma_var1 = np.ma.mean(np.ma.mean(model_var1, axis=1), axis=1)
model_ma_var1 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_ma_var1.flatten())
model_ma_var2 = np.ma.mean(np.ma.mean(model_var2, axis=1), axis=1)
model_ma_var2 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_ma_var2.flatten())
model_ma_var3 = np.ma.mean(np.ma.mean(model_var3, axis=1), axis=1)
model_ma_var3 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_ma_var3.flatten())
model_ma_var4 = np.ma.mean(np.ma.mean(model_var4, axis=1), axis=1)
model_ma_var4 = np.ma.masked_where(np.ma.getmask(ref_ma_var), model_ma_var4.flatten())

cordex_ma_var1 = np.ma.mean(np.ma.mean(cordex_var1, axis=1), axis=1)
cordex_ma_var1 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_ma_var1.flatten())
cordex_ma_var2 = np.ma.mean(np.ma.mean(cordex_var2, axis=1), axis=1)
cordex_ma_var2 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_ma_var2.flatten())
cordex_ma_var3 = np.ma.mean(np.ma.mean(cordex_var3, axis=1), axis=1)
cordex_ma_var3 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_ma_var3.flatten())
cordex_ma_var4 = np.ma.mean(np.ma.mean(cordex_var4, axis=1), axis=1)
cordex_ma_var4 = np.ma.masked_where(np.ma.getmask(ref_ma_var), cordex_ma_var4.flatten())

obs_ma_var2 = np.ma.mean(np.ma.mean(obs_var2, axis=1), axis=1)
obs_ma_var2 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_ma_var2.flatten())
obs_ma_var3 = np.ma.mean(np.ma.mean(obs_var3, axis=1), axis=1)
obs_ma_var3 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_ma_var3.flatten())
obs_ma_var4 = np.ma.mean(np.ma.mean(obs_var4, axis=1), axis=1)
obs_ma_var4 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_ma_var4.flatten())
obs_ma_var5 = np.ma.mean(np.ma.mean(obs_var5, axis=1), axis=1)
obs_ma_var5 = np.ma.masked_where(np.ma.getmask(ref_ma_var), obs_ma_var5.flatten())

taylor_model_stats1 = sm.taylor_statistics(model_ma_var1, ref_ma_var)
taylor_model_stats2 = sm.taylor_statistics(model_ma_var2, ref_ma_var)
taylor_model_stats3 = sm.taylor_statistics(model_ma_var3, ref_ma_var)
taylor_model_stats4 = sm.taylor_statistics(model_ma_var4, ref_ma_var)

taylor_cordex_stats1 = sm.taylor_statistics(cordex_ma_var1, ref_ma_var)
taylor_cordex_stats2 = sm.taylor_statistics(cordex_ma_var2, ref_ma_var)
taylor_cordex_stats3 = sm.taylor_statistics(cordex_ma_var3, ref_ma_var)
taylor_cordex_stats4 = sm.taylor_statistics(cordex_ma_var4, ref_ma_var)

taylor_obs_stats2 = sm.taylor_statistics(obs_ma_var2, ref_ma_var)
taylor_obs_stats3 = sm.taylor_statistics(obs_ma_var3, ref_ma_var)
taylor_obs_stats4 = sm.taylor_statistics(obs_ma_var4, ref_ma_var)
taylor_obs_stats5 = sm.taylor_statistics(obs_ma_var5, ref_ma_var)

# plot for CESMs
ccoef = [1.]
crmsd = [0.]
sdev = [taylor_model_stats1['sdev'][0]]
sdev.append(taylor_model_stats1['sdev'][1])
sdev.append(taylor_model_stats2['sdev'][1])
sdev.append(taylor_model_stats3['sdev'][1])
sdev.append(taylor_model_stats4['sdev'][1])
crmsd.append(taylor_model_stats1['crmsd'][1])
crmsd.append(taylor_model_stats2['crmsd'][1])
crmsd.append(taylor_model_stats3['crmsd'][1])
crmsd.append(taylor_model_stats4['crmsd'][1])
ccoef.append(taylor_model_stats1['ccoef'][1])
ccoef.append(taylor_model_stats2['ccoef'][1])
ccoef.append(taylor_model_stats3['ccoef'][1])
ccoef.append(taylor_model_stats4['ccoef'][1])


# overlay for CORDEX-SEA
sdev.append(taylor_cordex_stats1['sdev'][1])
sdev.append(taylor_cordex_stats2['sdev'][1])
sdev.append(taylor_cordex_stats3['sdev'][1])
sdev.append(taylor_cordex_stats4['sdev'][1])
crmsd.append(taylor_cordex_stats1['crmsd'][1])
crmsd.append(taylor_cordex_stats2['crmsd'][1])
crmsd.append(taylor_cordex_stats3['crmsd'][1])
crmsd.append(taylor_cordex_stats4['crmsd'][1])
ccoef.append(taylor_cordex_stats1['ccoef'][1])
ccoef.append(taylor_cordex_stats2['ccoef'][1])
ccoef.append(taylor_cordex_stats3['ccoef'][1])
ccoef.append(taylor_cordex_stats4['ccoef'][1])

# overlay for other obs
ccoef.append(taylor_obs_stats2['ccoef'][1])
ccoef.append(taylor_obs_stats3['ccoef'][1])
ccoef.append(taylor_obs_stats4['ccoef'][1])
ccoef.append(taylor_obs_stats5['ccoef'][1])
crmsd.append(taylor_obs_stats2['crmsd'][1])
crmsd.append(taylor_obs_stats3['crmsd'][1])
crmsd.append(taylor_obs_stats4['crmsd'][1])
crmsd.append(taylor_obs_stats5['crmsd'][1])
sdev.append(taylor_obs_stats2['sdev'][1])
sdev.append(taylor_obs_stats3['sdev'][1])
sdev.append(taylor_obs_stats4['sdev'][1])
sdev.append(taylor_obs_stats5['sdev'][1])


print(taylor_model_stats1)
print(taylor_model_stats2)
print(sdev)
print(crmsd)
print(ccoef)

sdev = np.array(sdev)
crmsd = np.array(crmsd)
ccoef = np.array(ccoef)
labels = [legends[8]]+legends[0:8]+legends[9:]
print(labels)
plt.clf()
sm.taylor_diagram(sdev, crmsd, ccoef, numberPanels=1, MarkerDisplayed='marker', markerLabel=labels,
                  markerLabelColor='b', markerLegend='on', markerColor='b',
                  styleOBS='-', colOBS='g', markerobs='o',
                  markerSize=5, tickRMS=[0.0, 1.0, 2.0, 3.0],
                  tickRMSangle=115, showlabelsRMS='on',
                  titleRMS='off', titleOBS='Ref', checkstats='on')

plt.savefig(outdir+'vrseasia_prect_taylor_diagram_monthly_ts_vs_cordexsea_refCRU.pdf')
