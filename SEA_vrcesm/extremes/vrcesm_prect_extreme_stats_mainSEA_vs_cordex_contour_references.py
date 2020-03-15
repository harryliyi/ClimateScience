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
from modules.datareader.mod_dataread_obs_CPC import readobs_pre_CPC
from modules.datareader.mod_dataread_obs_TRMM import readobs_pre_TRMM_day
from modules.plot.mod_plt_regrid import data_regrid
from modules.plot.mod_plt_lines import plot_lines
from modules.plot.mod_plt_findstns import data_findstns
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import basemap
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
latbounds = [-10, 25]
lonbounds = [90, 130]

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'day'

# set percentile
percentile = 99
percents = [50, 70, 80, 90, 95, 97, 99, 99.5]

# return years
return_years = [2, 5, 10, 20, 50, 100, 150]

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# create seasons
seasons = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
seasnames = ['DJF', 'MAM', 'JJA', 'SON', 'Annual']

# plot legend
cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5']

cesm_colors = ['red', 'yellow', 'green', 'blue']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed']

############################################################################
# define functions
############################################################################


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


def get_percentiles(vars, ref_var, percents):
    res = []
    for idata in vars:
        temp = np.percentile(idata, percents)
        res.append(temp)

    return res

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
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
modelname = 'IPSL-IPSL-CM5A-LR'
cordex_var2, cordex_time2, cordex_lats2, cordex_lons2 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
modelname = 'MPI-M-MPI-ESM-MR'
cordex_var3, cordex_time3, cordex_lats3, cordex_lons3 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
modelname = 'MOHC-HadGEM2-ES'
cordex_var4, cordex_time4, cordex_lats4, cordex_lons4 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)

# convert from kg/(m^2*s) to mm/day
cordex_var1 = cordex_var1 * 86400 * 1000 / 997
cordex_var2 = cordex_var2 * 86400 * 1000 / 997
cordex_var3 = cordex_var3 * 86400 * 1000 / 997
cordex_var4 = cordex_var4 * 86400 * 1000 / 997

print(cordex_var1[0, :, :])
print(cordex_var4[0, :, :])
print(cordex_var1.shape)
print(cordex_var4.shape)

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

# convert to mm/day
model_var1 = model_var1 * 86400 * 1000
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000

print(model_var1[0, :, :])

# read Observations

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

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-EC-Earth-RegCM4',
           'CORDEX-IPSL-CM5A-RegCM4', 'CORDEX-MPI-ESM-RegCM4', 'CORDEX-HadGEM2-ES-RCA4',
           'CPC', 'TRMM', 'SA-OBS']

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CPC', 'TRMM', 'SA-OBS']

############################################################################
# plot extreme prect over specific region
############################################################################

for idx_seas in range(len(seasons)):
    print('Plot for '+seasnames[idx_seas]+':')
    cordex_var_sub1 = get_vars(cordex_time1, cordex_var1, seasons[idx_seas])
    cordex_var_sub2 = get_vars(cordex_time2, cordex_var2, seasons[idx_seas])
    cordex_var_sub3 = get_vars(cordex_time3, cordex_var3, seasons[idx_seas])
    cordex_var_sub4 = get_vars(cordex_time4, cordex_var4, seasons[idx_seas])

    model_var_sub1 = get_vars(model_time1, model_var1, seasons[idx_seas])
    model_var_sub2 = get_vars(model_time2, model_var2, seasons[idx_seas])
    model_var_sub3 = get_vars(model_time3, model_var3, seasons[idx_seas])
    model_var_sub4 = get_vars(model_time4, model_var4, seasons[idx_seas])

    obs_var_sub1 = get_vars(obs_time1, obs_var1, seasons[idx_seas])
    obs_var_sub2 = get_vars(obs_time2, obs_var2, seasons[idx_seas])

    cordex_percentile1 = np.nanpercentile(cordex_var_sub1, percentile, axis=0)
    cordex_percentile2 = np.nanpercentile(cordex_var_sub2, percentile, axis=0)
    cordex_percentile3 = np.nanpercentile(cordex_var_sub3, percentile, axis=0)
    cordex_percentile4 = np.nanpercentile(cordex_var_sub4, percentile, axis=0)

    model_percentile1 = np.percentile(model_var_sub1, percentile, axis=0)
    model_percentile2 = np.percentile(model_var_sub2, percentile, axis=0)
    model_percentile3 = np.percentile(model_var_sub3, percentile, axis=0)
    model_percentile4 = np.percentile(model_var_sub4, percentile, axis=0)

    obs_percentile1 = np.nanpercentile(obs_var_sub1, percentile, axis=0)
    obs_percentile2 = np.percentile(obs_var_sub2, percentile, axis=0)

    # print(cordex_percentile1)
    # print(cordex_percentile2)

    plot_data = [model_percentile1, model_percentile2, model_percentile3, model_percentile4,
                 cordex_percentile1, cordex_percentile2, cordex_percentile3, cordex_percentile4,
                 obs_percentile1, obs_percentile2]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4,
                 obs_lons1, obs_lons2]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4,
                 obs_lats1, obs_lats2]

    print('Plot the '+str(percentile)+'th precip extremes...')
    varname = 'Total Precip'
    var_unit = 'mm/day'

    clevs = np.arange(10., 82., 2.)
    colormap = cm.viridis

    ylabel = ['CESMs', 'CORDEXs', 'Observations']

    title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + ' ' + str(percentile)+'th total precip over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_'+str(percentile) + \
        'th_prect_SEA_'+seasnames[idx_seas]+'_mean_contour_vs_cordex'
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0, ylabel=ylabel)

    # cesm only
    plot_data = [model_percentile1, model_percentile2, model_percentile3, model_percentile4,
                 obs_percentile1, obs_percentile2]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 obs_lons1, obs_lons2]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 obs_lats1, obs_lats2]

    varname = 'Total Precip'
    var_unit = 'mm/day'

    clevs = np.arange(10., 82., 2.)
    colormap = cm.viridis

    ylabel = ['CESMs', 'Observations']

    title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + ' ' + str(percentile)+'th total precip over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_'+str(percentile) + \
        'th_prect_SEA_'+seasnames[idx_seas]+'_mean_contour'
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0, ylabel=ylabel)

    ############################################################################
    # plot extreme prect biases over specific region
    ############################################################################
    # CPC
    ref_data = obs_percentile1
    ref_lons = obs_lons1
    ref_lats = obs_lats1
    ref_name = 'CPC'

    vars = [model_percentile1, model_percentile2, model_percentile3, model_percentile4,
            cordex_percentile1, cordex_percentile2, cordex_percentile3, cordex_percentile4]

    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    plot_data = []
    for idx_data in range(len(vars)):
        res = get_bias(vars[idx_data], ref_data, ref_lons, ref_lats, plot_lons[idx_data], plot_lats[idx_data])
        plot_data.append(res)

    print('Plot the '+str(percentile)+'th precip extremes biases ref'+ref_name+'...')
    varname = 'Total Precip'
    var_unit = 'mm/day'

    clevs = np.arange(-40., 42., 2.)
    colormap = cm.BrBG

    ylabel = ['CESMs', 'CORDEXs']

    title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + ' ' + str(percentile)+'th precip extemes biases over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_'+str(percentile) + \
        'th_prect_SEA_'+seasnames[idx_seas]+'_biases_contour_vs_cordex_ref'+ref_name
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0, ylabel=ylabel)

    # cesm only
    vars = [model_percentile1, model_percentile2, model_percentile3, model_percentile4]

    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    plot_data = []
    for idx_data in range(len(vars)):
        res = get_bias(vars[idx_data], ref_data, ref_lons, ref_lats, plot_lons[idx_data], plot_lats[idx_data])
        plot_data.append(res)

    varname = 'Total Precip'
    var_unit = 'mm/day'

    clevs = np.arange(-40., 42., 2.)
    colormap = cm.BrBG

    ylabel = ['CESMs', 'CORDEXs']

    title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + ' ' + str(percentile)+'th precip extemes biases over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_'+str(percentile) + \
        'th_prect_SEA_'+seasnames[idx_seas]+'_biases_contour_ref'+ref_name
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0, ylabel=ylabel)

    ############################################################################
    # TRMM
    ref_data = obs_percentile2
    ref_lons = obs_lons2
    ref_lats = obs_lats2
    ref_name = 'TRMM'

    cordex_var_sub1 = get_vars(cordex_time1[(1998-iniyear)*365:], cordex_var1[(1998-iniyear)*365:], seasons[idx_seas])
    cordex_var_sub2 = get_vars(cordex_time2[(1998-iniyear)*365:], cordex_var2[(1998-iniyear)*365:], seasons[idx_seas])
    cordex_var_sub3 = get_vars(cordex_time3[(1998-iniyear)*365:], cordex_var3[(1998-iniyear)*365:], seasons[idx_seas])
    cordex_var_sub4 = get_vars(cordex_time4[(1998-iniyear)*360:], cordex_var4[(1998-iniyear)*360:], seasons[idx_seas])

    model_var_sub1 = get_vars(model_time1[(1998-iniyear)*365:], model_var1[(1998-iniyear)*365:], seasons[idx_seas])
    model_var_sub2 = get_vars(model_time2[(1998-iniyear)*365:], model_var2[(1998-iniyear)*365:], seasons[idx_seas])
    model_var_sub3 = get_vars(model_time3[(1998-iniyear)*365:], model_var3[(1998-iniyear)*365:], seasons[idx_seas])
    model_var_sub4 = get_vars(model_time4[(1998-iniyear)*365:], model_var4[(1998-iniyear)*365:], seasons[idx_seas])

    cordex_percentile1 = np.percentile(cordex_var_sub1, percentile, axis=0)
    cordex_percentile2 = np.percentile(cordex_var_sub2, percentile, axis=0)
    cordex_percentile3 = np.percentile(cordex_var_sub3, percentile, axis=0)
    cordex_percentile4 = np.percentile(cordex_var_sub4, percentile, axis=0)

    model_percentile1 = np.percentile(model_var_sub1, percentile, axis=0)
    model_percentile2 = np.percentile(model_var_sub2, percentile, axis=0)
    model_percentile3 = np.percentile(model_var_sub3, percentile, axis=0)
    model_percentile4 = np.percentile(model_var_sub4, percentile, axis=0)

    vars = [model_percentile1, model_percentile2, model_percentile3, model_percentile4,
            cordex_percentile1, cordex_percentile2, cordex_percentile3, cordex_percentile4]

    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4]

    plot_data = []
    for idx_data in range(len(vars)):
        res = get_bias(vars[idx_data], ref_data, ref_lons, ref_lats, plot_lons[idx_data], plot_lats[idx_data])
        plot_data.append(res)

    print('Plot the '+str(percentile)+'th precip extremes biases ref'+ref_name+'...')
    varname = 'Total Precip'
    var_unit = 'mm/day'

    clevs = np.arange(-40., 42., 2.)
    colormap = cm.BrBG

    ylabel = ['CESMs', 'CORDEXs']

    title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + ' ' + str(percentile)+'th precip extemes biases over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_'+str(percentile) + \
        'th_prect_SEA_'+seasnames[idx_seas]+'_biases_contour_vs_cordex_ref'+ref_name
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0, ylabel=ylabel)

    # cesm only
    vars = [model_percentile1, model_percentile2, model_percentile3, model_percentile4]

    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    plot_data = []
    for idx_data in range(len(vars)):
        res = get_bias(vars[idx_data], ref_data, ref_lons, ref_lats, plot_lons[idx_data], plot_lats[idx_data])
        plot_data.append(res)

    varname = 'Total Precip'
    var_unit = 'mm/day'

    clevs = np.arange(-40., 42., 2.)
    colormap = cm.BrBG

    ylabel = ['CESMs', 'CORDEXs']

    title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + ' ' + str(percentile)+'th precip extemes biases over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_'+str(percentile) + \
        'th_prect_SEA_'+seasnames[idx_seas]+'_biases_contour_ref'+ref_name
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0, ylabel=ylabel)
