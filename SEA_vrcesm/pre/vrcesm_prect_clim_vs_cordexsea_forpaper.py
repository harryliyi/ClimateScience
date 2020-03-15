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
from modules.stats.mod_stats_clim import getstats_2D_ttest, getstats_2D_ftest
from modules.datareader.mod_dataread_obs_CPC import readobs_pre_CPC
from modules.datareader.mod_dataread_obs_pre import readobs_pre_mon
from modules.datareader.mod_dataread_obs_CRU import readobs_pre_CRU
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import basemap
plt.switch_backend('agg')

# import modules

############################################################################
# setup directory
############################################################################
outdircordex = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/1980-2005/'
outdircesm = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/pre/prect/1980-2005/'

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

# set data frequency
frequency = 'mon'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# create seasons
seasons = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  [6, 7, 8], [12, 1, 2]]
seasnames = ['Annual', 'JJA', 'DJF']

# define the legends
legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-EC-Earth-RegCM4',
           'CORDEX-IPSL-CM5A-RegCM4', 'CORDEX-MPI-ESM-RegCM4', 'CORDEX-HadGEM2-ES-RCA4']
cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

############################################################################
# define functions
############################################################################


def get_rmse(var_diff):
    temp = var_diff.flatten()
    temp = temp[~np.isnan(temp)]
    res = np.sqrt((temp**2).mean())

    return res


def get_vars(time, var, months):
    if np.isscalar(months):
        res_var = var[time.month == months, :, :]
    else:
        # print(time[np.in1d(time.month, months)])
        temp = var[np.in1d(time.month, months), :, :]
        nmonths = len(months)
        nyears = int(temp.shape[0]/nmonths)
        res_var = np.zeros((nyears, temp.shape[1], temp.shape[2]))
        res_var[:, :, :] = np.nan
        for iyear in range(nyears):
            res_var[iyear, :, :] = np.mean(temp[iyear*nmonths:(iyear+1)*nmonths, :, :], axis=0)

    return res_var


def get_regrid(var, in_lons, in_lats, out_lons, out_lats):
    lonout, latout = np.meshgrid(out_lons, out_lats)
    res = basemap.interp(var, in_lons, in_lats, lonout, latout, order=1)

    return res


def cal_clim_mean(data_time, datasets, seasons):
    res_mean = []
    res_std = []

    for idx_data in range(len(datasets)):
        for iseason in seasons:
            temp = get_vars(data_time[idx_data], datasets[idx_data], iseason)
            temp_mean = np.mean(temp, axis=0)
            temp_std = np.std(temp, axis=0)
            res_mean.append(temp_mean)
            res_std.append(temp_std)

    return res_mean, res_std


def cal_clim_diff(model_means, model_stds, model_lats, model_lons, ref_mean, ref_std, ref_lats, ref_lons, ref_df):
    res_mean_diff = []
    res_mean_ttest = []
    res_var_diff = []
    res_var_ftest = []
    res_lats = []
    res_lons = []

    for idx_data in range(len(model_means)):
        idx_ref = idx_data % 3
        idx_model = int(idx_data/3)

        ref_mean_reg = get_regrid(ref_mean[idx_ref], ref_lons, ref_lats, model_lons[idx_model], model_lats[idx_model])
        ref_std_reg = get_regrid(ref_std[idx_ref], ref_lons, ref_lats, model_lons[idx_model], model_lats[idx_model])

        temp_mean_diff, temp_mean_ttest = getstats_2D_ttest(
            model_means[idx_data], ref_mean_reg, model_stds[idx_data], ref_std_reg, ref_df[idx_ref], ref_df[idx_ref])
        temp_var_diff, temp_var_ftest = getstats_2D_ftest(model_stds[idx_data], ref_std_reg)

        res_mean_diff.append(temp_mean_diff)
        res_mean_ttest.append(temp_mean_ttest)
        res_var_diff.append(temp_var_diff)
        res_var_ftest.append(temp_var_ftest)
        res_lats.append(model_lats[idx_model])
        res_lons.append(model_lons[idx_model])

    return res_mean_diff, res_mean_ttest, res_var_diff, res_var_ftest, res_lats, res_lons


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
varfname = 'PREC'
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

#
# obs_latl6 = np.argmin(np.abs(obs_lats6 - reg_lats[0]))
# obs_latu6 = np.argmin(np.abs(obs_lats6 - reg_lats[1]))
# obs_lonl6 = np.argmin(np.abs(obs_lons6 - reg_lons[0]))
# obs_lonr6 = np.argmin(np.abs(obs_lons6 - reg_lons[1]))

print(obs_var4[0, obs_latl4: obs_latu4 + 1, obs_lonl4: obs_lonr4 + 1])
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
# plot climatological mean for each dataset
############################################################################
print('Calculating climatological mean and variability of each model output...')

datasets = [model_var1, model_var2, model_var3, model_var4,
            cordex_var1, cordex_var2, cordex_var3, cordex_var4]

data_time = [model_time1, model_time2, model_time3, model_time4,
             cordex_time1, cordex_time2, cordex_time3, cordex_time4]

models_mean, models_std = cal_clim_mean(data_time, datasets, seasons)

model_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
              cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

model_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
              cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]

# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

############################################################################
# plot climatological difference against CRU
############################################################################
# plot against CRU
project = 'CRU'
projectstr = 'CRU'
print('Plotting against '+project+'...')

ref_data = [obs_var1]
ref_time = [obs_time1]
ref_lats = obs_lats1
ref_lons = obs_lons1
ref_df = [(endyear-iniyear+1)*12, (endyear-iniyear+1)*3, (endyear-iniyear+1)*3]

ref_mean, ref_std = cal_clim_mean(ref_time, ref_data, seasons)
# print(len(ref_mean))
# for irefer in ref_mean:
#     print(irefer.shape)

# calculate model differences and significance levels
models_mean_diff, models_mean_ttest, models_std_diff, models_std_ftest, plot_lats, plot_lons = cal_clim_diff(
    models_mean, models_std, model_lats, model_lons, ref_mean, ref_std, ref_lats, ref_lons, ref_df)

# plot for difference with significance level
clevs = np.arange(-8., 8.5, 0.5)
colormap = cm.RdBu_r

clim_labels = ['Annual', 'JJA', 'DJF']
for idx in range(len(models_mean_diff)-3):
    clim_labels.append('')

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_mean_ttest, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

# plot for variability
clevs = np.arange(-2., 2.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

print('print std msd:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = np.nanmean(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

print('print std rmse:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = get_rmse(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

# cesm only
models_mean_diff_total = models_mean_diff
models_mean_ttest_total = models_mean_ttest
models_std_diff_total = models_std_diff
models_std_ftest_total = models_std_ftest
plot_lons_total = plot_lons
plot_lats_total = plot_lats
clim_labels_total = clim_labels

models_mean_diff = models_mean_diff_total[0:12]
models_mean_ttest = models_mean_ttest_total[0:12]
models_std_diff = models_std_diff_total[0:12]
models_std_ftest = models_std_ftest_total[0:12]
plot_lons = plot_lons_total[0:12]
plot_lats = plot_lats_total[0:12]
clim_labels = clim_labels_total[0:12]

# plot for difference with significance level
clevs = np.arange(-5., 5.5, 0.5)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_mean_ttest, ylabel=cesm_legends)

fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_wtsig_inone_a'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_mean_ttest, ylabel=cesm_legends)


# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)

fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_nosig_inone_a'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=cesm_legends)

# plot for variability
clevs = np.arange(-1., 1.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=cesm_legends)

fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_wtsig_inone_a'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=cesm_legends)


# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)

fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_nosig_inone_a'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=cesm_legends)

# cordex only
models_mean_diff = models_mean_diff_total[12:24]
models_mean_ttest = models_mean_ttest_total[12:24]
models_std_diff = models_std_diff_total[12:24]
models_std_ftest = models_std_ftest_total[12:24]
plot_lons = plot_lons_total[12:24]
plot_lats = plot_lats_total[12:24]
clim_labels = clim_labels_total[12:24]

# plot for difference with significance level
clevs = np.arange(-5., 5.5, 0.5)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_wtsig_inone_b'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_mean_ttest, ylabel=cesm_legends)


# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_nosig_inone_b'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=cesm_legends)

# plot for variability
clevs = np.arange(-1., 1.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_wtsig_inone_b'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=cesm_legends)


# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_nosig_inone_b'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=cesm_legends)

############################################################################
# plot climatological difference against GPCC
############################################################################
# plot against GPCC
project = 'GPCC'
projectstr = 'GPCC'
print('Plotting against '+project+'...')

ref_data = [obs_var2]
ref_time = [obs_time2]
ref_lats = obs_lats2
ref_lons = obs_lons2
ref_df = [(endyear-iniyear+1)*12, (endyear-iniyear+1)*3, (endyear-iniyear+1)*3]

ref_mean, ref_std = cal_clim_mean(ref_time, ref_data, seasons)
# print(len(ref_mean))
# for irefer in ref_mean:
#     print(irefer.shape)

# calculate model differences and significance levels
models_mean_diff, models_mean_ttest, models_std_diff, models_std_ftest, plot_lats, plot_lons = cal_clim_diff(
    models_mean, models_std, model_lats, model_lons, ref_mean, ref_std, ref_lats, ref_lons, ref_df)

# plot for difference with significance level
clevs = np.arange(-8., 8.5, 0.5)
colormap = cm.RdBu_r

clim_labels = ['Annual', 'JJA', 'DJF']
for idx in range(len(models_mean_diff)-3):
    clim_labels.append('')

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_mean_ttest, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

# plot for variability
clevs = np.arange(-2., 2.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

print('print std msd:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = np.nanmean(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

print('print std rmse:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = get_rmse(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

# cesm only
models_mean_diff = models_mean_diff[0:12]
models_mean_ttest = models_mean_ttest[0:12]
models_std_diff = models_std_diff[0:12]
models_std_ftest = models_std_ftest[0:12]
plot_lons = plot_lons[0:12]
plot_lats = plot_lats[0:12]
clim_labels = clim_labels[0:12]

# plot for difference with significance level
clevs = np.arange(-5., 5.5, 0.5)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_mean_ttest, ylabel=cesm_legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)

# plot for variability
clevs = np.arange(-1., 1.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=cesm_legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)

############################################################################
# plot climatological difference against APHRODITE
############################################################################
# plot against APHRODITE
project = 'APHRODITE'
projectstr = 'APHRODITE'
print('Plotting against '+project+'...')

ref_data = [obs_var3]
ref_time = [obs_time3]
ref_lats = obs_lats3
ref_lons = obs_lons3
ref_df = [(endyear-iniyear+1)*12, (endyear-iniyear+1)*3, (endyear-iniyear+1)*3]

ref_mean, ref_std = cal_clim_mean(ref_time, ref_data, seasons)
# print(len(ref_mean))
# for irefer in ref_mean:
#     print(irefer.shape)

# calculate model differences and significance levels
models_mean_diff, models_mean_ttest, models_std_diff, models_std_ftest, plot_lats, plot_lons = cal_clim_diff(
    models_mean, models_std, model_lats, model_lons, ref_mean, ref_std, ref_lats, ref_lons, ref_df)

# plot for difference with significance level
clevs = np.arange(-8., 8.5, 0.5)
colormap = cm.RdBu_r

clim_labels = ['Annual', 'JJA', 'DJF']
for idx in range(len(models_mean_diff)-3):
    clim_labels.append('')

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_mean_ttest, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

# plot for variability
clevs = np.arange(-2., 2.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

print('print std msd:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = np.nanmean(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

print('print std rmse:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = get_rmse(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

# cesm only
models_mean_diff = models_mean_diff[0:12]
models_mean_ttest = models_mean_ttest[0:12]
models_std_diff = models_std_diff[0:12]
models_std_ftest = models_std_ftest[0:12]
plot_lons = plot_lons[0:12]
plot_lats = plot_lats[0:12]
clim_labels = clim_labels[0:12]

# plot for difference with significance level
clevs = np.arange(-5., 5.5, 0.5)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_mean_ttest, ylabel=cesm_legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)

# plot for variability
clevs = np.arange(-1., 1.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=cesm_legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)


############################################################################
# plot climatological difference against ERA-interim
############################################################################
# plot against ERA-interim
project = 'ERA-interim'
projectstr = 'erainterim'
print('Plotting against '+project+'...')

ref_data = [obs_var4]
ref_time = [obs_time4]
ref_lats = obs_lats4
ref_lons = obs_lons4
ref_df = [(endyear-iniyear+1)*12, (endyear-iniyear+1)*3, (endyear-iniyear+1)*3]

ref_mean, ref_std = cal_clim_mean(ref_time, ref_data, seasons)
# print(len(ref_mean))
# for irefer in ref_mean:
#     print(irefer.shape)

# calculate model differences and significance levels
models_mean_diff, models_mean_ttest, models_std_diff, models_std_ftest, plot_lats, plot_lons = cal_clim_diff(
    models_mean, models_std, model_lats, model_lons, ref_mean, ref_std, ref_lats, ref_lons, ref_df)

# plot for difference with significance level
clevs = np.arange(-8., 8.5, 0.5)
colormap = cm.RdBu_r

clim_labels = ['Annual', 'JJA', 'DJF']
for idx in range(len(models_mean_diff)-3):
    clim_labels.append('')

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_mean_ttest, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

# plot for variability
clevs = np.arange(-2., 2.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

print('print std msd:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = np.nanmean(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

print('print std rmse:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = get_rmse(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

# cesm only
models_mean_diff = models_mean_diff[0:12]
models_mean_ttest = models_mean_ttest[0:12]
models_std_diff = models_std_diff[0:12]
models_std_ftest = models_std_ftest[0:12]
plot_lons = plot_lons[0:12]
plot_lats = plot_lats[0:12]
clim_labels = clim_labels[0:12]

# plot for difference with significance level
clevs = np.arange(-5., 5.5, 0.5)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_mean_ttest, ylabel=cesm_legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)

# plot for variability
clevs = np.arange(-1., 1.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=cesm_legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)


############################################################################
# plot climatological difference against CPC
############################################################################
# plot against CPC
project = 'CPC'
projectstr = 'CPC'
print('Plotting against '+project+'...')

ref_data = [obs_var5]
ref_time = [obs_time5]
ref_lats = obs_lats5
ref_lons = obs_lons5
ref_df = [(endyear-iniyear+1)*12, (endyear-iniyear+1)*3, (endyear-iniyear+1)*3]

ref_mean, ref_std = cal_clim_mean(ref_time, ref_data, seasons)
# print(len(ref_mean))
# for irefer in ref_mean:
#     print(irefer.shape)

# calculate model differences and significance levels
models_mean_diff, models_mean_ttest, models_std_diff, models_std_ftest, plot_lats, plot_lons = cal_clim_diff(
    models_mean, models_std, model_lats, model_lons, ref_mean, ref_std, ref_lats, ref_lons, ref_df)

# plot for difference with significance level
clevs = np.arange(-8., 8.5, 0.5)
colormap = cm.RdBu_r

clim_labels = ['Annual', 'JJA', 'DJF']
for idx in range(len(models_mean_diff)-3):
    clim_labels.append('')

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_mean_ttest, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

# plot for variability
clevs = np.arange(-2., 2.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_vs_cordex_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircordex+fname, opt=0, ylabel=legends)

print('print std msd:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = np.nanmean(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

print('print std rmse:')
for imodel, stddiff in enumerate(models_std_diff):
    ilats = model_lats[int(imodel/3)]
    ilons = model_lons[int(imodel/3)]
    ilatl = np.argmin(np.abs(ilats - reg_lats[0]))
    ilatu = np.argmin(np.abs(ilats - reg_lats[1]))
    ilonl = np.argmin(np.abs(ilons - reg_lons[0]))
    ilonu = np.argmin(np.abs(ilons - reg_lons[1]))

    res = get_rmse(stddiff[ilatl: ilatu + 1, ilonl: ilonu + 1])
    print(res)

# cesm only
models_mean_diff = models_mean_diff[0:12]
models_mean_ttest = models_mean_ttest[0:12]
models_std_diff = models_std_diff[0:12]
models_std_ftest = models_std_ftest[0:12]
plot_lons = plot_lons[0:12]
plot_lats = plot_lats[0:12]
clim_labels = clim_labels[0:12]

# plot for difference with significance level
clevs = np.arange(-5., 5.5, 0.5)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_mean_ttest, ylabel=cesm_legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_diff_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_mean_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)

# plot for variability
clevs = np.arange(-1., 1.2, 0.2)
colormap = cm.RdBu_r

title = str(iniyear)+'-'+str(endyear)+' climatological mean '+varname+' variability differece (Ref as '+project+')'
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_wtsig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=models_std_ftest, sig_thres=2.2693, ylabel=cesm_legends)

# without significance levels
fname = 'vrseasia_'+varstr+'_SEA_clim_mean_var_contour_ref'+projectstr+'_nosig_inone'

plot_2Dcontour(models_std_diff, plot_lons, plot_lats, colormap, clevs, clim_labels, lonbounds,
               latbounds, varname, var_unit, title, outdircesm+fname, opt=0, ylabel=cesm_legends)
