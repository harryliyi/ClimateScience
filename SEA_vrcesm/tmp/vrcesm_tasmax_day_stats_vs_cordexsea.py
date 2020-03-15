# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read tmp data from vrcesm and CORDEX-SEA and obs
# S2-calculate mean and std for data
# S3-calculate RMSD, MSD, MRD, Correlation between model data and obs
#
# Written by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_obs_CPC import readobs_tmp_CPC
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pickle
plt.switch_backend('agg')

# import modules

############################################################################
# setup directory
############################################################################
outdircordex = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/tmp/tasmax/VRseasia_vs_CORDEX_SEA/'
outdircesm = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/tmp/tasmax/'
outdirnotes = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/notes/tmp/tasmax/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Maxium 2m Temperature'
varstr = 'tasmax'
var_unit = r'$^{\circ}C$'

# time bounds
iniyear = 1980
endyear = 2005
monthts = pd.date_range(start=str(iniyear)+'-01-01', end=str(endyear)+'-12-01', freq='MS')

# define regions
latbounds = [-15, 25]
lonbounds = [90, 145]

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]

latbounds = reg_lats
lonbounds = reg_lons

# set data frequency
frequency = 'day'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

############################################################################
# read data
############################################################################
'''
print('Reading CORDEX-SEA data...')

# read cordex
project = 'SEA-22'
varname = 'tasmax'
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
'''
############################################################################
# read vrcesm

print('Reading VRCESM data...')

varname = 'TREFHTMX'

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

# print(model_time1)
print(model_var1.shape)
print(model_var1[0, :, :])

############################################################################
# read Observations

print('Reading Obs data...')

# read CPC
project = 'CPC'
obs_var, obs_time, obs_lats, obs_lons = readobs_tmp_CPC('tasmax', iniyear, endyear, frequency, latbounds, lonbounds)

obs_var[obs_var.mask] = np.nan


# record the mask for each data set
model_mask1 = np.isnan(model_var1[0, :, :])
model_mask2 = np.isnan(model_var2[0, :, :])
model_mask3 = np.isnan(model_var3[0, :, :])
model_mask4 = np.isnan(model_var4[0, :, :])

# cordex_mask1 = np.isnan(cordex_var1[0, :, :])
# cordex_mask2 = np.isnan(cordex_var2[0, :, :])
# cordex_mask3 = np.isnan(cordex_var3[0, :, :])
# cordex_mask4 = np.isnan(cordex_var4[0, :, :])

obs_mask = np.isnan(obs_var[0, :, :])

# print(obs_var2[0, :, :])

############################################################################
# calculate the statistics
############################################################################

# create name list
name_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
             'Oct', 'Nov', 'Dec', 'DJF', 'MAM', 'JJA', 'SON', 'JJAS', 'Annual']
idx_list = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [12, 1, 2], [
    3, 4, 5], [6, 7, 8], [9, 10, 11],  [6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

# calculate stats among multi-datasets
# datasets = [model_var1, model_var2, model_var3, model_var4,
#             cordex_var1, cordex_var2, cordex_var3, cordex_var4,
#             obs_var]
#
# datasets_times = [model_time1, model_time2, model_time3, model_time4,
#                   cordex_time1, cordex_time2, cordex_time3, cordex_time4,
#                   obs_time]
#
# legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
#            'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES',
#            'CPC']
#
# datasets_masks = [model_mask1, model_mask2, model_mask3, model_mask4,
#                   cordex_mask1, cordex_mask2, cordex_mask3, cordex_mask4,
#                   obs_mask]

datasets = [model_var1, model_var2, model_var3, model_var4, obs_var]

datasets_times = [model_time1, model_time2, model_time3, model_time4, obs_time]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CPC']

result = {'time_list': name_list}
result = pd.DataFrame(result)
result = result.set_index('time_list')

for idata, idata_var in enumerate(datasets):
    print('Calculating first for moments for '+legends[idata]+'...')
    idata_time = datasets_times[idata]

    result_mean = []
    result_variance = []
    result_skewness = []
    result_kurtosis = []
    for idx, itime in enumerate(name_list):
        temp_3d = idata_var[np.in1d(idata_time.month, idx_list[idx]), :, :]

        result_mean.append(np.nanmean(temp_3d[:, :, :]))
        result_variance.append(np.nanvar(temp_3d[:, :, :]))
        result_skewness.append(ss.skew(temp_3d[~np.isnan(temp_3d)]))
        result_kurtosis.append(ss.kurtosis(temp_3d[~np.isnan(temp_3d)], fisher=False))

    result[legends[idata]+'-mean'] = result_mean
    result[legends[idata]+'-var'] = result_variance
    result[legends[idata]+'-skew'] = result_skewness
    result[legends[idata]+'-kurt'] = result_kurtosis


# save pickle data
pickle.dump(result, open(outdirnotes+'vrseasia_'+varstr+'_day_stats_moments_result.p', "wb"))
result.to_csv(outdirnotes+'vrseasia_'+varstr+'_day_stats_moments_result.csv', sep=',', index=True)
'''
############################################################################
# plot histogram fro Tmax
colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod',
          'darkcyan', 'darkmagenta', 'black']
line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-.', '-.', '-.', '-.', '-']

# plot for both cesm and cordex
for idx, itime in enumerate(name_list):
    print('plot histogram for '+itime+'...')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idata, idata_var in enumerate(datasets):
        temp_3d = idata_var[np.in1d(idata_time.month, idx_list[idx]), :, :]
        density = ss.gaussian_kde(temp_3d[~np.isnan(temp_3d)])
        xs = np.linspace(10, 45, 300)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        plt.plot(xs, 100*density(xs), color=colors[idx], linestyle=line_types[idx], linewidth=1.5, label=legends[idx])

    plt.legend(handlelength=4, fontsize=5)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('Tmax', fontsize=8)
    plt.ylabel('Frequency (%)', fontsize=8)

    plt.suptitle('VRCESM vs CORDEX-SEA Frequency distribution of '+itime+' '+varname +
                 ' over the period '+str(iniyear)+'-'+str(endyear), fontsize=9, y=0.95)

    fname = 'vrseasia_'+varstr+'_mainSEA_day_line_frequency_dist_vs_cordex_overland_'+str(idx+1)+'.pdf'
    plt.savefig(outdircordex+fname, bbox_inches='tight')
'''

# plot for cesm only
datasets = [model_var1, model_var2, model_var3, model_var4, obs_var]

datasets_times = [model_time1, model_time2, model_time3, model_time4, obs_time]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CPC']

colors = ['red', 'yellow', 'green', 'blue', 'black']
line_types = ['dashed', 'dashed', 'dashed', 'dashed',  '-']

for idx, itime in enumerate(name_list):
    print('plot histogram for '+itime+'...')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idata, idata_var in enumerate(datasets):
        temp_3d = idata_var[np.in1d(idata_time.month, idx_list[idx]), :, :]
        density = ss.gaussian_kde(temp_3d[~np.isnan(temp_3d)])
        xs = np.linspace(10, 45, 300)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        plt.plot(xs, 100*density(xs), color=colors[idata],
                 linestyle=line_types[idata], linewidth=1.5, label=legends[idata])

    plt.legend(handlelength=4, fontsize=5)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('Tmax', fontsize=8)
    plt.ylabel('Frequency (%)', fontsize=8)

    plt.suptitle('VRCESM Frequency distribution of '+itime+' '+varname +
                 ' over the period '+str(iniyear)+'-'+str(endyear), fontsize=9, y=0.95)

    fname = 'vrseasia_'+varstr+'_mainSEA_day_line_frequency_dist_vs_cordex_overland_'+str(idx+1)+'.pdf'
    plt.savefig(outdircesm+fname, bbox_inches='tight')
    plt.close(fig)
