# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-calculate extreme
# S3-plot contour
#
# Written by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_obs_pre import readobs_pre_mon
from modules.datareader.mod_dataread_obs_CRU import readobs_pre_CRU
from modules.datareader.mod_dataread_obs_CPC import readobs_pre_CPC
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex
from modules.stats.mod_stats_calculations import cal_stats_clim_multidatasets, cal_stats_clim_multidatasets_reference_bias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
plt.switch_backend('agg')

# import modules

############################################################################
# setup directory
############################################################################
outdirnotes = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/notes/pre/'

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

# define regions
latbounds = [-15, 25]
lonbounds = [90, 145]

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]

latbounds = reg_lats
lonbounds = reg_lons

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

# convert from kg/(m^2*s) to mm/day
cordex_var1 = cordex_var1 * 86400 * 1000 / 997
cordex_var2 = cordex_var2 * 86400 * 1000 / 997
cordex_var3 = cordex_var3 * 86400 * 1000 / 997
cordex_var4 = cordex_var4 * 86400 * 1000 / 997

cordex_var1[cordex_var1.mask] = np.nan
cordex_var2[cordex_var2.mask] = np.nan
cordex_var3[cordex_var3.mask] = np.nan
cordex_var4[cordex_var4.mask] = np.nan

print(cordex_var1[0, :, :].mask)
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

############################################################################
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

model_var1[model_var1.mask] = np.nan
model_var2[model_var2.mask] = np.nan
model_var3[model_var3.mask] = np.nan
model_var4[model_var4.mask] = np.nan

# convert from m/s to mm/day
model_var1 = model_var1 * 86400 * 1000
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000

print(model_time1)
print(model_var1.shape)
print(model_var1[0, :, :])

############################################################################
# read Observations

print('Reading Obs data...')

# read CRU
project = 'CRU'
obs_var1, obs_time1, obs_lats1, obs_lons1 = readobs_pre_CRU(
    'precip', iniyear, endyear, latbounds, lonbounds)

# read GPCC
project = 'GPCC'
obs_var2, obs_time2, obs_lats2, obs_lons2 = readobs_pre_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read APHRODITE
project = 'APHRODITE'
obs_var3, obs_time3, obs_lats3, obs_lons3 = readobs_pre_mon(
    project, iniyear, endyear, latbounds, lonbounds)

# read ERA-interim
project = 'ERA-interim'
obs_var4, obs_time4, obs_lats4, obs_lons4 = readobs_pre_mon(
    project, iniyear, endyear, latbounds, lonbounds, oceanmask=1)

# read CPC
project = 'CPC'
obs_var5, obs_time5, obs_lats5, obs_lons5 = readobs_pre_CPC('precip', iniyear, endyear, frequency, latbounds, lonbounds)

# # read GPCC
# project = 'GPCP'
# obs_var5, obs_time5, obs_lats5, obs_lons5 = readobs_pre_mon(
#     project, iniyear, endyear, latbounds, lonbounds)
#
# # read ERA-interim without oceanmask
# project = 'ERA-interim'
# obs_var6, obs_time6, obs_lats6, obs_lons6 = readobs_pre_mon(
#     project, iniyear, endyear, latbounds, lonbounds, oceanmask=1)

obs_var1[obs_var1.mask] = np.nan
obs_var2[obs_var2.mask] = np.nan
obs_var3[obs_var3.mask] = np.nan
obs_var4[obs_var4.mask] = np.nan
obs_var5[obs_var5.mask] = np.nan

print(obs_var4[0, :, :])
# $print(obs_lats1[obs_latl1 : obs_latu1 + 1])

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

############################################################################
# calculate the statistics
############################################################################
# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# create name list
name_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
             'Oct', 'Nov', 'Dec', 'DJF', 'MAM', 'JJA', 'SON', 'JJAS', 'Annual']
idx_list = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [12, 1, 2], [
    3, 4, 5], [6, 7, 8], [9, 10, 11],  [6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

# calculate stats among multi-datasets
datasets = [model_var1, model_var2, model_var3, model_var4,
            cordex_var1, cordex_var2, cordex_var3, cordex_var4,
            obs_var1, obs_var2, obs_var3, obs_var4, obs_var5]

datasets_times = [model_time1, model_time2, model_time3, model_time4,
                  cordex_time1, cordex_time2, cordex_time3, cordex_time4,
                  obs_time1, obs_time2, obs_time3, obs_time4, obs_time5]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES',
           'CRU-TS4.03', 'GPCC', 'APHRODITE', 'ERA-interim', 'CPC']

datasets_masks = [model_mask1, model_mask2, model_mask3, model_mask4,
                  cordex_mask1, cordex_mask2, cordex_mask3, cordex_mask4,
                  obs_mask1, obs_mask2, obs_mask3, obs_mask4, obs_mask5]

dataset_stats = cal_stats_clim_multidatasets(datasets, datasets_times, legends, name_list, idx_list)
# print(dataset_stats)

# test pickle data
pickle.dump(dataset_stats, open(outdirnotes+'vrseasia_'+varstr+'_mon_stats_result_vs_cordex.p', "wb"))
dataset_stats.to_csv(outdirnotes+'vrseasia_'+varstr+'_mon_stats_result_vs_cordex.csv', sep=',', index=True)

############################################################################
# calculate stats between multi-datasets and reference dataset

# compare with CRU
ref_legend = 'CRU'
ref_str = 'CRU'
ref_data = obs_var1
ref_time = obs_time1
ref_lon = obs_lons1
ref_lat = obs_lats1

datasets = [model_var1, model_var2, model_var3, model_var4,
            cordex_var1, cordex_var2, cordex_var3, cordex_var4]

datasets_times = [model_time1, model_time2, model_time3, model_time4,
                  cordex_time1, cordex_time2, cordex_time3, cordex_time4]

datasets_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]

datasets_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES']

datasets_masks = [model_mask1, model_mask2, model_mask3, model_mask4,
                  cordex_mask1, cordex_mask2, cordex_mask3, cordex_mask4]

dataset_stats_bias = cal_stats_clim_multidatasets_reference_bias(datasets, datasets_times, datasets_lons, datasets_lats, legends,
                                                                 ref_data, ref_time, ref_lon, ref_lat, ref_legend, name_list, idx_list)

# save pickle data
pickle.dump(dataset_stats_bias, open(outdirnotes+'vrseasia_' + varstr +
                                     '_mon_stats_bias_result_vs_cordex_ref'+ref_str+'.p', "wb"))
dataset_stats_bias.to_csv(outdirnotes+'vrseasia_'+varstr+'_mon_stats_bias_result_vs_cordex_ref' +
                          ref_str+'.csv', sep=',', index=True)

# compare with GPCC
ref_legend = 'GPCC'
ref_str = 'GPCC'
ref_data = obs_var2
ref_time = obs_time2
ref_lon = obs_lons2
ref_lat = obs_lats2

dataset_stats_bias = cal_stats_clim_multidatasets_reference_bias(datasets, datasets_times, datasets_lons, datasets_lats, legends,
                                                                 ref_data, ref_time, ref_lon, ref_lat, ref_legend, name_list, idx_list)

# save pickle data
pickle.dump(dataset_stats_bias, open(outdirnotes+'vrseasia_'+varstr +
                                     '_mon_stats_bias_result_vs_cordex_ref'+ref_str+'.p', "wb"))
dataset_stats_bias.to_csv(outdirnotes+'vrseasia_'+varstr+'_mon_stats_bias_result_vs_cordex_ref' +
                          ref_str+'.csv', sep=',', index=True)

# compare with APHRODITE
ref_legend = 'APHRODITE'
ref_str = 'APHRODITE'
ref_data = obs_var3
ref_time = obs_time3
ref_lon = obs_lons3
ref_lat = obs_lats3

dataset_stats_bias = cal_stats_clim_multidatasets_reference_bias(datasets, datasets_times, datasets_lons, datasets_lats, legends,
                                                                 ref_data, ref_time, ref_lon, ref_lat, ref_legend, name_list, idx_list)

# save pickle data
pickle.dump(dataset_stats_bias, open(outdirnotes+'vrseasia_'+varstr +
                                     '_mon_stats_bias_result_vs_cordex_ref'+ref_str+'.p', "wb"))
dataset_stats_bias.to_csv(outdirnotes+'vrseasia_'+varstr+'_mon_stats_bias_result_vs_cordex_ref' +
                          ref_str+'.csv', sep=',', index=True)

# compare with ERA-interim
ref_legend = 'ERA-interim'
ref_str = 'erainterim'
ref_data = obs_var4
ref_time = obs_time4
ref_lon = obs_lons4
ref_lat = obs_lats4

dataset_stats_bias = cal_stats_clim_multidatasets_reference_bias(datasets, datasets_times, datasets_lons, datasets_lats, legends,
                                                                 ref_data, ref_time, ref_lon, ref_lat, ref_legend, name_list, idx_list)

# save pickle data
pickle.dump(dataset_stats_bias, open(outdirnotes+'vrseasia_'+varstr +
                                     '_mon_stats_bias_result_vs_cordex_ref'+ref_str+'.p', "wb"))
dataset_stats_bias.to_csv(outdirnotes+'vrseasia_'+varstr+'_mon_stats_bias_result_vs_cordex_ref' +
                          ref_str+'.csv', sep=',', index=True)

# compare with CPC
ref_legend = 'CPC'
ref_str = 'CPC'
ref_data = obs_var5
ref_time = obs_time5
ref_lon = obs_lons5
ref_lat = obs_lats5

dataset_stats_bias = cal_stats_clim_multidatasets_reference_bias(datasets, datasets_times, datasets_lons, datasets_lats, legends,
                                                                 ref_data, ref_time, ref_lon, ref_lat, ref_legend, name_list, idx_list)

# save pickle data
pickle.dump(dataset_stats_bias, open(outdirnotes+'vrseasia_'+varstr +
                                     '_mon_stats_bias_result_vs_cordex_ref'+ref_str+'.p', "wb"))
dataset_stats_bias.to_csv(outdirnotes+'vrseasia_'+varstr+'_mon_stats_bias_result_vs_cordex_ref' +
                          ref_str+'.csv', sep=',', index=True)
