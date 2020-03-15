# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-calculate extreme
# S3-plot contour
#
# Written by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.stats.mod_stats_extremes import gpdfit_moment
from modules.datareader.mod_dataread_obs_CPC import readobs_pre_CPC
from modules.datareader.mod_dataread_obs_TRMM import readobs_pre_TRMM_day
from modules.datareader.mod_dataread_obs_pre import read_SAOBS_pre
from modules.plot.mod_plt_lines import plot_lines
from modules.plot.mod_plt_findstns import data_findstns
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import basemap
import pickle
import pandas as pd
plt.switch_backend('agg')

############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/extremes/'

############################################################################
# set parameters
############################################################################
# variable info
var_longname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'
varname = 'PRECT'

# time bounds
iniyear = 1980
endyear = 2005

# define regions
latbounds = [10, 25]
lonbounds = [100, 110]

# mainland Southeast Asia
reg_lats = [0, 25]
reg_lons = [90, 120]
reg_name = 'mainSEA'

# set data frequency
frequency = 'day'

# set percentile
percentile = 99
obsrate = (100.-percentile)/100.
percents = [50, 70, 80, 90, 95, 97, 99, 99.5]

# return years
return_years = [2, 5, 10, 20, 50, 75,  100, 150]
m = np.array(return_years) * 365.

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# create seasons
seasons = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
seasnames = ['DJF', 'MAM', 'JJA', 'SON', 'Annual']

# plot legend
legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-EC-Earth-RegCM4',
           'CORDEX-IPSL-CM5A-RegCM4', 'CORDEX-MPI-ESM-RegCM4', 'CORDEX-HadGEM2-ES-RCA4']
colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod', 'darkcyan', 'darkmagenta']
markers = ['o', '+', '^', 'x', 'D', 's', 'P', 'h']
linetypes = ['-', '-', '-', '-', '--', '--', '--', '--']

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5']
cesm_colors = ['red', 'yellow', 'green', 'blue']
cesm_markers = ['o', '+', '^', 'x']
cesm_linetypes = ['-', '-', '-', '-']

############################################################################
# define functions
############################################################################


def rmse(model, reference):
    diff = model - reference
    diff = diff.flatten()
    diff = diff[~np.isnan(diff)]
    res = np.sqrt(np.mean(diff**2))

    return res


def get_vars(time, prect, months):
    if np.isscalar(months):
        res_prect = prect[time.month == months, :, :]
    else:
        # print(time[np.in1d(time.month, months)])
        res_prect = prect[np.in1d(time.month, months), :, :]

    return res_prect


def get_bias(var, ref_var, in_lons, in_lats, out_lons, out_lats):
    lonout, latout = np.meshgrid(out_lons, out_lats)
    temp = basemap.interp(ref_var, in_lons, in_lats, lonout, latout, order=1)
    res = var - temp

    return res


def get_percentiles(vars, percents):
    res = []
    for idata in vars:
        temp_data = idata.flatten()
        temp_data = temp_data[~np.isnan(temp_data)]
        temp = np.percentile(temp_data, percents)
        temp = np.round(temp, decimals=2)
        res.append(temp)

    return res


def get_mean_percentiles(vars, percents):
    res = []
    for idata in vars:
        var_percentiles = []
        for ipercent in percents:
            temp_percentile = np.nanpercentile(idata, ipercent, axis=0)
            var_percentiles.append(np.nanmean(temp_percentile))
        var_percentiles = np.array(var_percentiles)
        var_percentiles = np.round(var_percentiles, decimals=2)
        res.append(var_percentiles)

    return res


def get_percentiles_rmse(vars, ref_var, model_lons, model_lats, ref_lons, ref_lats, percents):
    res = []
    for idx_data, idata in enumerate(vars):
        model_rmse = []
        out_lons = model_lons[idx_data]
        out_lats = model_lats[idx_data]
        lonout, latout = np.meshgrid(out_lons, out_lats)
        for ipercent in percents:
            model_percentile = np.nanpercentile(idata, ipercent, axis=0)
            ref_percentile = np.nanpercentile(ref_var, ipercent, axis=0)
            ref_percentile = basemap.interp(ref_percentile, ref_lons, ref_lats, lonout, latout, order=1)
            model_rmse.append(rmse(model_percentile, ref_percentile))

        model_rmse = np.array(model_rmse)
        model_rmse = np.round(model_rmse, decimals=2)
        model_rmse = np.array(model_rmse)

        res.append(model_rmse)

    return res


def get_percentiles_rmse_SA(vars, ref_var, percents):
    res = []
    for idx_data, idata in enumerate(vars):
        model_rmse = []
        for ipercent in percents:
            model_percentile = np.nanpercentile(idata.values, ipercent, axis=0)
            ref_percentile = np.nanpercentile(ref_var.values, ipercent, axis=0)
            model_rmse.append(rmse(model_percentile, ref_percentile))

        model_rmse = np.array(model_rmse)
        model_rmse = np.round(model_rmse, decimals=2)
        model_rmse = np.array(model_rmse)

        res.append(model_rmse)

    return res


def get_returns(vars, percentile):
    res = []
    for idata in vars:
        temp_data = idata.flatten()
        temp_data = temp_data[~np.isnan(temp_data)]
        temp_percent = np.percentile(temp_data, percentile)
        temp_data_sub = temp_data[temp_data > temp_percent] - temp_percent
        temp_fit = gpdfit_moment(temp_data_sub)
        temp_return = temp_percent + temp_fit[1] / temp_fit[0] * ((m*obsrate) ** temp_fit[0] - 1)
        temp_return = np.array(temp_return)
        temp_return = np.round(temp_return, decimals=2)
        res.append(temp_return)

    return res


def get_returns_biases(returns):
    ref_return = returns[-1]
    res = []
    for ireturn in returns[:-1]:
        temp_bias = ireturn - ref_return
        temp_bias = np.array(temp_bias)
        temp_bias = np.round(temp_bias, decimals=2)
        res.append(temp_bias)

    return res

############################################################################
# plot data
############################################################################


def plot_scatters(plot_xdata, plot_data, colors, markers, line_types, legends, subtitles, title, fname, **kwargs):
    arranges = {1: [1, 1], 2: [2, 1], 3: [1, 3], 4: [2, 2], 5: [3, 2],
                6: [2, 4], 8: [2, 4], 9: [3, 3], 10: [3, 4], 12: [3, 4]}

    nfigs = len(plot_data)
    if nfigs not in arranges:
        print('plot_2Dcontour: Error! Too many Sub-figures, the maximum number is 9!')
        return -1

    plt.clf()
    fig = plt.figure()
    xdata = range(len(plot_xdata))

    for idx in range(len(plot_data)):
        ax = fig.add_subplot(arranges[nfigs][0], arranges[nfigs][1], idx+1)
        ax.set_title(subtitles[idx], fontsize=5, pad=2)

        temp_data = plot_data[idx]
        for idx_data, idata in enumerate(temp_data):
            print(line_types[idx_data])
            ax.plot(xdata, idata, c=colors[idx_data], markersize=3, marker=markers[idx_data], linestyle=line_types[idx_data], linewidth=0.75, alpha=0.85, label=legends[idx_data])

        # set x/y tick label size
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_xticks(xdata)
        ax.set_xticklabels(plot_xdata, fontsize=5)

        if 'xlabel' in kwargs:
            xlabel = kwargs['xlabel']

            if (len(plot_data)-idx) <= arranges[nfigs][1]:
                if type(xlabel) == str:
                    ax.set_xlabel(xlabel, fontsize=5)
                else:
                    ax.set_xlabel(xlabel[int(idx % (arranges[nfigs][1]))], fontsize=5)

        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
            if (idx % (arranges[nfigs][1]) == 0):
                if type(ylabel) == str:
                    ax.set_ylabel(ylabel, fontsize=5)
                else:
                    ax.set_ylabel(ylabel[int(idx/(arranges[nfigs][1]))], fontsize=5)

        if (idx == len(plot_data)-1):
            ax.legend(loc='upper left', fontsize=4.5, handlelength=3.5)

    # add title
    plt.savefig(fname+'.png', bbox_inches='tight', dpi=1200)
    plt.suptitle(title, fontsize=7, y=0.95)

    # save figure
    plt.savefig(fname+'.pdf', bbox_inches='tight', dpi=1200)
    plt.close(fig)


############################################################################
# read data
############################################################################

# read Observations
# read CPC


print('Reading CPC data...')

varname = 'precip'

obs_var1, obs_time1, obs_lats1, obs_lons1 = readobs_pre_CPC(
    varname, iniyear, endyear, frequency, latbounds, lonbounds)

obs_var1[obs_var1.mask] = np.nan
print(obs_var1[0, :, :])

# read TRMM

print('Reading TRMM data...')

obs_var2, obs_time2, obs_lats2, obs_lons2 = readobs_pre_TRMM_day(
    'precipitation', 1998, endyear, latbounds, lonbounds, oceanmask=0)

# read SA-OBS

print('Reading SA-OBS data...')

# read SA-OBS
version = 'countries'
countries = ['Thailand', 'Vietnam', 'Myanmar', 'Cambodia']
dataset3, obs_var3, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_SAOBS_pre(
    version, 1990, endyear, countries, missing_ratio=10, ignore_years=[1999])
obs_time3 = dataset3.index

############################################################################
# Calculate the percentiles and return levels against CPC
############################################################################
print('Calculating for CPC...')

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

cordex_var1[cordex_var1.mask] = np.nan
cordex_var2[cordex_var2.mask] = np.nan
cordex_var3[cordex_var3.mask] = np.nan
cordex_var4[cordex_var4.mask] = np.nan

# convert from kg/(m^2*s) to mm/day
cordex_var1 = cordex_var1 * 86400 * 1000 / 997
cordex_var2 = cordex_var2 * 86400 * 1000 / 997
cordex_var3 = cordex_var3 * 86400 * 1000 / 997
cordex_var4 = cordex_var4 * 86400 * 1000 / 997

# print(cordex_var1[0, :, :])
# print(cordex_var1.shape)
# print(cordex_var4.shape)

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

model_var1[model_var1.mask] = np.nan
model_var2[model_var2.mask] = np.nan
model_var3[model_var3.mask] = np.nan
model_var4[model_var4.mask] = np.nan

# convert to mm/day
model_var1 = model_var1 * 86400 * 1000
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000

# print(model_var1[0, :, :])

# calculate percentiles for all grids
vars = [model_var1, model_var2, model_var3, model_var4,
        cordex_var1, cordex_var2, cordex_var3, cordex_var4, obs_var1]
percentiles_allgrids1 = get_percentiles(vars, percents)
# print(percentiles_allgrids1)

# calculate mean of each grid percentiles
vars = [model_var1, model_var2, model_var3, model_var4,
        cordex_var1, cordex_var2, cordex_var3, cordex_var4, obs_var1]
percentiles_mean1 = get_mean_percentiles(vars, percents)
# print(percentiles_mean1)

# calculate RMSE of percentiles between model and obs
vars = [model_var1, model_var2, model_var3, model_var4,
        cordex_var1, cordex_var2, cordex_var3, cordex_var4]
var_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
            cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
var_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
            cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]
ref_var = obs_var1
ref_lons = obs_lons1
ref_lats = obs_lats1
percentiles_rmse1 = get_percentiles_rmse(vars, ref_var, var_lons, var_lats, ref_lons, ref_lats, percents)
# print(percentiles_rmse1)

# calculate return levels
vars = [model_var1, model_var2, model_var3, model_var4,
        cordex_var1, cordex_var2, cordex_var3, cordex_var4, obs_var1]
extreme_returns1 = get_returns(vars, percentile)
# print(extreme_returns1)

# calculate return levels biaes
extreme_returns_biases1 = get_returns_biases(extreme_returns1)
# print(extreme_returns_biases1)


############################################################################
# Calculate the percentiles and return levels against TRMM
############################################################################
print('Calculating for TRMM...')

print('Reading CORDEX-SEA data...')

# read cordex
project = 'SEA-22'
varname = 'pr'
cordex_models = ['ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-LR', 'MPI-M-MPI-ESM-MR', 'MOHC-HadGEM2-ES']

modelname = 'ICHEC-EC-EARTH'
cordex_var1, cordex_time1, cordex_lats1, cordex_lons1 = readcordex(
    varname, 1998, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
modelname = 'IPSL-IPSL-CM5A-LR'
cordex_var2, cordex_time2, cordex_lats2, cordex_lons2 = readcordex(
    varname, 1998, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
modelname = 'MPI-M-MPI-ESM-MR'
cordex_var3, cordex_time3, cordex_lats3, cordex_lons3 = readcordex(
    varname, 1998, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
modelname = 'MOHC-HadGEM2-ES'
cordex_var4, cordex_time4, cordex_lats4, cordex_lons4 = readcordex(
    varname, 1998, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)

# convert from kg/(m^2*s) to mm/day
cordex_var1 = cordex_var1 * 86400 * 1000 / 997
cordex_var2 = cordex_var2 * 86400 * 1000 / 997
cordex_var3 = cordex_var3 * 86400 * 1000 / 997
cordex_var4 = cordex_var4 * 86400 * 1000 / 997

# print(cordex_var1[0, :, :])
# print(cordex_var1.shape)
# print(cordex_var4.shape)

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, 1998, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, 1998, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, 1998, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, 1998, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

# convert to mm/day
model_var1 = model_var1 * 86400 * 1000
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000

# print(model_var1[0, :, :])

# calculate percentiles for all grids
vars = [model_var1, model_var2, model_var3, model_var4,
        cordex_var1, cordex_var2, cordex_var3, cordex_var4, obs_var2]
percentiles_allgrids2 = get_percentiles(vars, percents)
# print(percentiles_allgrids2)

# calculate mean of each grid percentiles
vars = [model_var1, model_var2, model_var3, model_var4,
        cordex_var1, cordex_var2, cordex_var3, cordex_var4, obs_var2]
percentiles_mean2 = get_mean_percentiles(vars, percents)
# print(percentiles_mean2)

# calculate RMSE of percentiles between model and obs
vars = [model_var1, model_var2, model_var3, model_var4,
        cordex_var1, cordex_var2, cordex_var3, cordex_var4]
var_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
            cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
var_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
            cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]
ref_var = obs_var2
ref_lons = obs_lons2
ref_lats = obs_lats2
percentiles_rmse2 = get_percentiles_rmse(vars, ref_var, var_lons, var_lats, ref_lons, ref_lats, percents)
# print(percentiles_rmse2)

# calculate return levels
vars = [model_var1, model_var2, model_var3, model_var4,
        cordex_var1, cordex_var2, cordex_var3, cordex_var4, obs_var2]
extreme_returns2 = get_returns(vars, percentile)
# print(extreme_returns2)

# calculate return levels biaes
extreme_returns_biases2 = get_returns_biases(extreme_returns2)
# print(extreme_returns_biases2)

############################################################################
# Calculate the percentiles and return levels against SA-OBS
############################################################################
print('Calculating for SA-OBS...')

print('Reading CORDEX-SEA data...')

# read cordex
project = 'SEA-22'
varname = 'pr'
cordex_models = ['ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-LR', 'MPI-M-MPI-ESM-MR', 'MOHC-HadGEM2-ES']

modelname = 'ICHEC-EC-EARTH'
cordex_var1, cordex_time1, cordex_lats1, cordex_lons1 = readcordex(
    varname, 1990, endyear, project, modelname, frequency, reg_lats, reg_lons, oceanmask=0, ignore_years=[1999])
modelname = 'IPSL-IPSL-CM5A-LR'
cordex_var2, cordex_time2, cordex_lats2, cordex_lons2 = readcordex(
    varname, 1990, endyear, project, modelname, frequency, reg_lats, reg_lons, oceanmask=0, ignore_years=[1999])
modelname = 'MPI-M-MPI-ESM-MR'
cordex_var3, cordex_time3, cordex_lats3, cordex_lons3 = readcordex(
    varname, 1990, endyear, project, modelname, frequency, reg_lats, reg_lons, oceanmask=0, ignore_years=[1999])
modelname = 'MOHC-HadGEM2-ES'
cordex_var4, cordex_time4, cordex_lats4, cordex_lons4 = readcordex(
    varname, 1990, endyear, project, modelname, frequency, reg_lats, reg_lons, oceanmask=0, ignore_years=[1999])

# convert from kg/(m^2*s) to mm/day
cordex_var1 = cordex_var1 * 86400 * 1000 / 997
cordex_var2 = cordex_var2 * 86400 * 1000 / 997
cordex_var3 = cordex_var3 * 86400 * 1000 / 997
cordex_var4 = cordex_var4 * 86400 * 1000 / 997

# print(cordex_var1[0, :, :])
# print(cordex_var1.shape)
# print(cordex_var4.shape)

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, 1990, endyear, resolution, varfname, case, frequency, reg_lats, reg_lons, oceanmask=0, ignore_years=[1999])

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, 1990, endyear, resolution, varfname, case, frequency, reg_lats, reg_lons, oceanmask=0, ignore_years=[1999])

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, 1990, endyear, resolution, varfname, case, frequency, reg_lats, reg_lons, oceanmask=0, ignore_years=[1999])

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, 1990, endyear, resolution, varfname, case, frequency, reg_lats, reg_lons, oceanmask=0, ignore_years=[1999])

# convert to mm/day
model_var1 = model_var1 * 86400 * 1000
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000

# print(model_var1[0, :, :])

# find stations in gridded data
cordex_var1 = data_findstns(cordex_var1, cordex_time1, cordex_lats1, cordex_lons1, obs_var3, stnlats, stnlons, stnnames)
cordex_var2 = data_findstns(cordex_var2, cordex_time2, cordex_lats2, cordex_lons2, obs_var3, stnlats, stnlons, stnnames)
cordex_var3 = data_findstns(cordex_var3, cordex_time3, cordex_lats3, cordex_lons3, obs_var3, stnlats, stnlons, stnnames)
cordex_var4 = data_findstns(cordex_var4, cordex_time4, cordex_lats4, cordex_lons4, obs_var3, stnlats, stnlons, stnnames)

model_var1 = data_findstns(model_var1, model_time1, model_lats1, model_lons1, obs_var3, stnlats, stnlons, stnnames)
model_var2 = data_findstns(model_var2, model_time2, model_lats2, model_lons2, obs_var3, stnlats, stnlons, stnnames)
model_var3 = data_findstns(model_var3, model_time3, model_lats3, model_lons3, obs_var3, stnlats, stnlons, stnnames)
model_var4 = data_findstns(model_var4, model_time4, model_lats4, model_lons4, obs_var3, stnlats, stnlons, stnnames)


# calculate percentiles for all grids
vars = [model_var1.values, model_var2.values, model_var3.values, model_var4.values,
        cordex_var1.values, cordex_var2.values, cordex_var3.values, cordex_var4.values, dataset3.values]
percentiles_allgrids3 = get_percentiles(vars, percents)
# print(percentiles_allgrids3)

# calculate mean of each grid percentiles
vars = [model_var1.values, model_var2.values, model_var3.values, model_var4.values,
        cordex_var1.values, cordex_var2.values, cordex_var3.values, cordex_var4.values, dataset3.values]
percentiles_mean3 = get_mean_percentiles(vars, percents)
# print(percentiles_mean3)

# calculate RMSE of percentiles between model and obs
vars = [model_var1, model_var2, model_var3, model_var4,
        cordex_var1, cordex_var2, cordex_var3, cordex_var4]
ref_var = dataset3
percentiles_rmse3 = get_percentiles_rmse_SA(vars, ref_var, percents)
# print(percentiles_rmse3)

# calculate return levels
vars = [model_var1.values, model_var2.values, model_var3.values, model_var4.values,
        cordex_var1.values, cordex_var2.values, cordex_var3.values, cordex_var4.values, dataset3.values]
extreme_returns3 = get_returns(vars, percentile)
# print(extreme_returns3)

# calculate return levels biaes
extreme_returns_biases3 = get_returns_biases(extreme_returns3)
# print(extreme_returns_biases3)

############################################################################
# Plot RMSD
############################################################################
print('Plot for RMSD...')

# cesm and cordex
plot_data = [percentiles_rmse1, percentiles_rmse2, percentiles_rmse3]
plot_xdata = percents

subtitles = ['Biases relative to CPC', 'Biases relative to TRMM', 'Biases relative to SACA']
title = 'Biases of precip extemes relative to observations over'+reg_name
fname = 'vrseasia_prect_extremes_SEA_biases_RMSE_references_vs_cordex'
plot_scatters(plot_xdata, plot_data, colors, markers, linetypes, legends, subtitles,
              title, outdir+fname, xlabel='Percentiles', ylabel='RMSD (mm/day)')

# cesm only
plot_data = [percentiles_rmse1[:4], percentiles_rmse2[:4], percentiles_rmse3[:4]]
plot_xdata = percents

subtitles = ['Biases relative to CPC', 'Biases relative to TRMM', 'Biases relative to SACA']
title = 'Biases of precip extemes relative to observations over'+reg_name
fname = 'vrseasia_prect_extremes_SEA_biases_RMSE_references'
plot_scatters(plot_xdata, plot_data, cesm_colors, cesm_markers, cesm_linetypes, cesm_legends, subtitles,
              title, outdir+fname, xlabel='Percentiles', ylabel='RMSD (mm/day)')


############################################################################
# Plot Return levels
############################################################################
print('Plot for return levels...')

# cesm and cordex
plot_data = [extreme_returns_biases1, extreme_returns_biases2, extreme_returns_biases3]
plot_xdata = return_years

subtitles = ['Biases relative to CPC', 'Biases relative to TRMM', 'Biases relative to SACA']
title = 'Biases of precip return levels relative to observations over'+reg_name
fname = 'vrseasia_prect_extremes_SEA_return_levels_references_vs_cordex'
plot_scatters(plot_xdata, plot_data, colors, markers, linetypes, legends, subtitles,
              title, outdir+fname, xlabel='Return years', ylabel='Biases (mm/day)')

# cesm only
plot_data = [extreme_returns_biases1[:4], extreme_returns_biases2[:4], extreme_returns_biases3[:4]]
plot_xdata = return_years

subtitles = ['Biases relative to CPC', 'Biases relative to TRMM', 'Biases relative to SACA']
title = 'Biases of precip return levels relative to observations over'+reg_name
fname = 'vrseasia_prect_extremes_SEA_return_levels_references'
plot_scatters(plot_xdata, plot_data, cesm_colors, cesm_markers,  cesm_linetypes, cesm_legends, subtitles,
              title, outdir+fname, xlabel='Return years', ylabel='Biases (mm/day)')


############################################################################
# Plot percentiles
############################################################################
print('Plot for Percentiles...')

# cesm and cordex
plot_data = [percentiles_allgrids1, percentiles_allgrids2, percentiles_allgrids3]
plot_xdata = percents

colors.append('black')
markers.append('*')
legends.append('Obervation')

subtitles = ['CPC as reference', 'TRMM as reference', 'SACA as reference']
title = 'Total precip percentiles compared to observations over'+reg_name
fname = 'vrseasia_prect_extremes_SEA_percentiles_references_vs_cordex'
plot_scatters(plot_xdata, plot_data, colors, markers,  linetypes, legends, subtitles,
              title, outdir+fname, xlabel='Percentiles', ylabel='Precip (mm/day)')

# cesm only
select_data = [0, 1, 2, 3, 8]
plot_data = [list(percentiles_allgrids1[i] for i in select_data),
             list(percentiles_allgrids2[i] for i in select_data),
             list(percentiles_allgrids3[i] for i in select_data)]
plot_xdata = percents

cesm_colors.append('black')
cesm_markers.append('*')
cesm_legends.append('Obervation')

subtitles = ['CPC as reference', 'TRMM as reference', 'SACA as reference']
title = 'Total precip percentiles compared to observations over'+reg_name
fname = 'vrseasia_prect_extremes_SEA_percentiles_references'
plot_scatters(plot_xdata, plot_data, cesm_colors, cesm_markers,  cesm_linetypes, cesm_legends, subtitles,
              title, outdir+fname, xlabel='Percentiles', ylabel='Precip (mm/day)')

############################################################################
# output the results
############################################################################

result = {}
result['models'] = legends
result['CPC-all-grids-percentiles'] = percentiles_allgrids1
result['CPC-grids-mean-percentiles'] = percentiles_mean1
result['CPC-percentiles-RMSE'] = percentiles_rmse1
result['CPC-return-levels'] = extreme_returns1
result['CPC-return-levels-biases'] = extreme_returns_biases1

result['TRMM-all-grids-percentiles'] = percentiles_allgrids2
result['TRMM-grids-mean-percentiles'] = percentiles_mean2
result['TRMM-percentiles-RMSE'] = percentiles_rmse2
result['TRMM-return-levels'] = extreme_returns2
result['TRMM-return-levels-biases'] = extreme_returns_biases2

result['SAOBS-all-grids-percentiles'] = percentiles_allgrids3
result['SAOBS-grids-mean-percentiles'] = percentiles_mean3
result['SAOBS-percentiles-RMSE'] = percentiles_rmse3
result['SAOBS-return-levels'] = extreme_returns3
result['SAOBS-return-levels-biases'] = extreme_returns_biases3

pickle.dump(result, open(outdir+'vrseasia_prect_extremes_SEA_relatives_to_references_results_vs_cordex.p', "wb"))

# save to csv

# CPC
percentiles_rmse1.append([0., 0., 0., 0., 0., 0., 0., 0.])
extreme_returns_biases1.append([0., 0., 0., 0., 0., 0., 0., 0.])
result_percents = {}
result_percents['Percentiles'] = percents
legends[-1] = 'CPC'
for idx in range(len(legends)):
    result_percents[legends[idx]+'-all-grids'] = percentiles_allgrids1[idx]
    result_percents[legends[idx]+'-grids-mean'] = percentiles_mean1[idx]
    result_percents[legends[idx]+'-RMSE'] = percentiles_rmse1[idx]
result_percents = pd.DataFrame(result_percents)
result_percents = result_percents.set_index('Percentiles')
result_percents.to_csv(outdir+'CPC_'+str(iniyear)+'to'+str(endyear) +
                       '_prect_extremes_percentiles_results.csv', sep=',', index=True)

result_returns = {}
result_returns['Return years'] = percents
for idx in range(len(legends)):
    result_returns[legends[idx]+'-return-levels'] = extreme_returns1[idx]
    result_returns[legends[idx]+'-biases'] = extreme_returns_biases1[idx]
result_returns = pd.DataFrame(result_returns)
result_returns = result_returns.set_index('Return years')
result_returns.to_csv(outdir+'CPC_'+str(iniyear)+'to'+str(endyear) +
                      '_prect_extremes_return_levels_results.csv', sep=',', index=True)

# TRMM
percentiles_rmse2.append([0., 0., 0., 0., 0., 0., 0., 0.])
extreme_returns_biases2.append([0., 0., 0., 0., 0., 0., 0., 0.])
result_percents = {}
result_percents['Percentiles'] = percents
legends[-1] = 'TRMM'
for idx in range(len(legends)):
    result_percents[legends[idx]+'-all-grids'] = percentiles_allgrids2[idx]
    result_percents[legends[idx]+'-grids-mean'] = percentiles_mean2[idx]
    result_percents[legends[idx]+'-RMSE'] = percentiles_rmse2[idx]
result_percents = pd.DataFrame(result_percents)
result_percents = result_percents.set_index('Percentiles')
result_percents.to_csv(outdir+'TRMM_1998to'+str(endyear) +
                       '_prect_extremes_percentiles_results.csv', sep=',', index=True)

result_returns = {}
result_returns['Return years'] = percents
for idx in range(len(legends)):
    result_returns[legends[idx]+'-return-levels'] = extreme_returns2[idx]
    result_returns[legends[idx]+'-biases'] = extreme_returns_biases2[idx]
result_returns = pd.DataFrame(result_returns)
result_returns = result_returns.set_index('Return years')
result_returns.to_csv(outdir+'TRMM_1998to'+str(endyear) +
                      '_prect_extremes_return_levels_results.csv', sep=',', index=True)

# SA-OBS
percentiles_rmse3.append([0., 0., 0., 0., 0., 0., 0., 0.])
extreme_returns_biases3.append([0., 0., 0., 0., 0., 0., 0., 0.])
result_percents = {}
result_percents['Percentiles'] = percents
legends[-1] = 'SA-OBS'
for idx in range(len(legends)):
    result_percents[legends[idx]+'-all-grids'] = percentiles_allgrids3[idx]
    result_percents[legends[idx]+'-grids-mean'] = percentiles_mean3[idx]
    result_percents[legends[idx]+'-RMSE'] = percentiles_rmse3[idx]
result_percents = pd.DataFrame(result_percents)
result_percents = result_percents.set_index('Percentiles')
result_percents.to_csv(outdir+'SAOBS_1990to'+str(endyear) +
                       '_prect_extremes_percentiles_results.csv', sep=',', index=True)

result_returns = {}
result_returns['Return years'] = return_years
for idx in range(len(legends)):
    result_returns[legends[idx]+'-return-levels'] = extreme_returns3[idx]
    result_returns[legends[idx]+'-biases'] = extreme_returns_biases3[idx]
result_returns = pd.DataFrame(result_returns)
result_returns = result_returns.set_index('Return years')
result_returns.to_csv(outdir+'SAOBS_1990to'+str(endyear) +
                      '_prect_extremes_return_levels_results.csv', sep=',', index=True)
