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
from modules.stats.mod_stats_clim import mon2clim
from modules.plot.mod_plt_lines import plot_lines
from modules.datareader.mod_dataread_vrcesm import readvrcesm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')

############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/pre/prect/extremes/'

############################################################################
# set parameters
############################################################################
# variable info
var_longname = 'Large-scale Precip'
varstr = 'precl'
var_unit = 'mm/day'
varname = 'PRECL'

# time bounds
iniyear = 1980
endyear = 2005

# define regions
latbounds = [0, 30]
lonbounds = [90, 115]
reg_name = 'mainSEA'

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'day'

# set percentile
percentile = 99

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


def get_extrevars(prec, prect, thresholds):
    nlats = prect.shape[1]
    nlons = prect.shape[2]
    res = np.zeros((nlats, nlons))

    for ilat in range(nlats):
        for ilon in range(nlons):
            temp_prec = prec[:, ilat, ilon]
            temp_prect = prect[:, ilat, ilon]
            temp_threshold = thresholds[ilat, ilon]

            temp_prec = temp_prec[temp_prect > temp_threshold]
            temp_prect = temp_prect[temp_prect > temp_threshold]
            res[ilat, ilon] = np.mean(temp_prec)

    return res


def get_extrefrac(precl, prect, thresholds):
    nlats = prect.shape[1]
    nlons = prect.shape[2]
    res = np.zeros((nlats, nlons))

    for ilat in range(nlats):
        for ilon in range(nlons):
            temp_precl = precl[:, ilat, ilon]
            temp_prect = prect[:, ilat, ilon]
            temp_threshold = thresholds[ilat, ilon]

            temp_precl = temp_precl[temp_prect > temp_threshold]
            temp_prect = temp_prect[temp_prect > temp_threshold]
            res[ilat, ilon] = np.sum(temp_precl)/np.sum(temp_prect)*100

    return res


def get_vars(time, precc, precl, prect, months):
    if np.isscalar(months):
        res_precc = precc[time.month == months, :, :]
        res_precl = precl[time.month == months, :, :]
        res_prect = prect[time.month == months, :, :]
    else:
        # print(time[np.in1d(time.month, months)])
        res_precc = precc[np.in1d(time.month, months), :, :]
        res_precl = precl[np.in1d(time.month, months), :, :]
        res_prect = prect[np.in1d(time.month, months), :, :]

    return res_precc, res_precl, res_prect

############################################################################
# read data
############################################################################

# read vrcesm


print('Reading VRCESM data...')

resolution = 'fv02'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
precc_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    'PRECC', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
precl_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    'PRECL', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
prect_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    'PRECT', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
precc_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    'PRECC', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
precl_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    'PRECL', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
prect_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    'PRECT', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
precc_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    'PRECC', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
precl_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    'PRECL', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
prect_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    'PRECT', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
precc_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    'PRECC', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
precl_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    'PRECL', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
prect_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    'PRECT', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

precc_var1 = precc_var1 * 86400 * 1000
precc_var2 = precc_var2 * 86400 * 1000
precc_var3 = precc_var3 * 86400 * 1000
precc_var4 = precc_var4 * 86400 * 1000

precl_var1 = precl_var1 * 86400 * 1000
precl_var2 = precl_var2 * 86400 * 1000
precl_var3 = precl_var3 * 86400 * 1000
precl_var4 = precl_var4 * 86400 * 1000

prect_var1 = prect_var1 * 86400 * 1000
prect_var2 = prect_var2 * 86400 * 1000
prect_var3 = prect_var3 * 86400 * 1000
prect_var4 = prect_var4 * 86400 * 1000


############################################################################
# plot extreme prect over specific region
############################################################################

for idx_seas in range(len(seasons)):
    print('Plot for '+seasnames[idx_seas]+':')
    precc_var_sub1, precl_var_sub1, prect_var_sub1 = get_vars(
        model_time1, precc_var1, precl_var1, prect_var1, seasons[idx_seas])
    precc_var_sub2, precl_var_sub2, prect_var_sub2 = get_vars(
        model_time2, precc_var2, precl_var2, prect_var2, seasons[idx_seas])
    precc_var_sub3, precl_var_sub3, prect_var_sub3 = get_vars(
        model_time3, precc_var3, precl_var3, prect_var3, seasons[idx_seas])
    precc_var_sub4, precl_var_sub4, prect_var_sub4 = get_vars(
        model_time4, precc_var4, precl_var4, prect_var4, seasons[idx_seas])

    prect_percentile1 = np.percentile(prect_var_sub1, percentile, axis=0)
    prect_percentile2 = np.percentile(prect_var_sub2, percentile, axis=0)
    prect_percentile3 = np.percentile(prect_var_sub3, percentile, axis=0)
    prect_percentile4 = np.percentile(prect_var_sub4, percentile, axis=0)

    plot_data = [prect_percentile1, prect_percentile2, prect_percentile3, prect_percentile4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    print('Plot the '+str(percentile)+'th precip extremes...')
    varname = 'Total Precip'
    var_unit = 'mm/day'

    clevs = np.arange(20., 81., 1.)
    colormap = cm.viridis

    title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + \
        ' mean of '+str(percentile)+'th total precip over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_'+str(percentile) + \
        'th_prect_SEA_'+seasnames[idx_seas]+'_mean_contour'
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)

    ############################################################################
    # calculate the seasonal prect/precc/precl
    ############################################################################

    print('Plot seasonal PRECC/PRECL...')
    vars_precc = [precc_var_sub1, precc_var_sub2, precc_var_sub3, precc_var_sub4]
    vars_precl = [precl_var_sub1, precl_var_sub2, precl_var_sub3, precl_var_sub4]
    vars_prect = [prect_var_sub1, prect_var_sub2, prect_var_sub3, prect_var_sub4]
    extreme_thresholds = [prect_percentile1, prect_percentile2, prect_percentile3, prect_percentile4]

    plot_data = []
    for idx_data in range(len(vars_prect)):
        res = np.mean(vars_prect[idx_data], axis=0)
        plot_data.append(res)

    for idx_data in range(len(vars_precc)):
        res = np.mean(vars_precc[idx_data], axis=0)
        plot_data.append(res)

    for idx_data in range(len(vars_precl)):
        res = np.mean(vars_precl[idx_data], axis=0)
        plot_data.append(res)

    varname = 'Precip'
    var_unit = 'mm/day'

    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 model_lons1, model_lons2, model_lons3, model_lons4,
                 model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 model_lats1, model_lats2, model_lats3, model_lats4,
                 model_lats1, model_lats2, model_lats3, model_lats4]

    clevs = np.arange(0., 18., 1.)
    colormap = cm.plasma_r
    legends = ['a) CESM-vrseasia', 'b) CESM-ne30', 'c) CESM-fv0.9x1.25', 'd) CESM-fv1.9x2.5',
               'e) CESM-vrseasia', 'f) CESM-ne30', 'g) CESM-fv0.9x1.25', 'h) CESM-fv1.9x2.5',
               'i) CESM-vrseasia', 'j) CESM-ne30', 'k) CESM-fv0.9x1.25',  'l) CESM-fv1.9x2.5']

    ylabel = ['Total Precip', 'Convective Precip', 'Large-scale Precip']

    title = str(iniyear)+' to '+str(endyear)+' ' + \
        seasnames[idx_seas]+' precipitation over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_prec_SEA_'+seasnames[idx_seas]+'_mean_contour'
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0, ylabel=ylabel)

    ############################################################################
    # calculate the prect/precc/precl in extreme events
    ############################################################################

    print('Plot seasonal PRECC/PRECL in extremes...')
    vars_precc = [precc_var_sub1, precc_var_sub2, precc_var_sub3, precc_var_sub4]
    vars_precl = [precl_var_sub1, precl_var_sub2, precl_var_sub3, precl_var_sub4]
    vars_prect = [prect_var_sub1, prect_var_sub2, prect_var_sub3, prect_var_sub4]
    extreme_thresholds = [prect_percentile1, prect_percentile2, prect_percentile3, prect_percentile4]

    plot_data = []
    for idx_data in range(len(vars_prect)):
        res = get_extrevars(vars_prect[idx_data], vars_prect[idx_data], extreme_thresholds[idx_data])
        plot_data.append(res)

    for idx_data in range(len(vars_precc)):
        res = get_extrevars(vars_precc[idx_data], vars_prect[idx_data], extreme_thresholds[idx_data])
        plot_data.append(res)

    for idx_data in range(len(vars_precl)):
        res = get_extrevars(vars_precl[idx_data], vars_prect[idx_data], extreme_thresholds[idx_data])
        plot_data.append(res)

    varname = 'Precip'
    var_unit = 'mm/day'

    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 model_lons1, model_lons2, model_lons3, model_lons4,
                 model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 model_lats1, model_lats2, model_lats3, model_lats4,
                 model_lats1, model_lats2, model_lats3, model_lats4]

    clevs = np.arange(10., 81., 1.)
    colormap = cm.plasma_r
    legends = ['a) CESM-vrseasia', 'b) CESM-ne30', 'c) CESM-fv0.9x1.25', 'd) CESM-fv1.9x2.5',
               'e) CESM-vrseasia', 'f) CESM-ne30', 'g) CESM-fv0.9x1.25', 'h) CESM-fv1.9x2.5',
               'i) CESM-vrseasia', 'j) CESM-ne30', 'k) CESM-fv0.9x1.25',  'l) CESM-fv1.9x2.5']

    ylabel = ['Total Precip', 'Convective Precip', 'Large-scale Precip']

    title = str(iniyear)+' to '+str(endyear)+' ' + \
        seasnames[idx_seas]+' '+str(percentile)+'th precipitation extremes over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_prec_'+str(percentile) + \
        'th_tp_SEA_'+seasnames[idx_seas]+'_mean_contour'
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0, ylabel=ylabel)

    ############################################################################
    # calculate the precc/precl fraction in extreme events
    ############################################################################

    print('Calculate the percent of PRECC/PRECL in extremes...')
    vars_precc = [precc_var_sub1, precc_var_sub2, precc_var_sub3, precc_var_sub4]
    vars_precl = [precl_var_sub1, precl_var_sub2, precl_var_sub3, precl_var_sub4]
    vars_prect = [prect_var_sub1, prect_var_sub2, prect_var_sub3, prect_var_sub4]
    extreme_thresholds = [prect_percentile1, prect_percentile2, prect_percentile3, prect_percentile4]

    plot_data = []
    for idx_data in range(len(vars_precc)):
        res = get_extrefrac(vars_precc[idx_data], vars_prect[idx_data], extreme_thresholds[idx_data])
        plot_data.append(res)

    for idx_data in range(len(vars_precl)):
        res = get_extrefrac(vars_precl[idx_data], vars_prect[idx_data], extreme_thresholds[idx_data])
        plot_data.append(res)

    print('Plot the percent of PRECL in extremes...')
    varname = 'Percentage'
    var_unit = '%'

    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
                 model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
                 model_lats1, model_lats2, model_lats3, model_lats4]

    clevs = np.arange(0., 105., 5.)
    colormap = cm.plasma_r
    legends = ['a) CESM-vrseasia', 'b) CESM-ne30', 'c) CESM-fv0.9x1.25',
               'd) CESM-fv1.9x2.5', 'e) CESM-vrseasia', 'f) CESM-ne30', 'g) CESM-fv0.9x1.25',
               'h) CESM-fv1.9x2.5']

    ylabel = ['Convective Precip', 'Large-scale Precip']

    title = str(iniyear)+' to '+str(endyear)+' ' + \
        seasnames[idx_seas]+' ratio of convective/large-scale precip to total precipitation in ' + \
        str(percentile)+'th extremes over'+reg_name
    fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_'+str(percentile) + \
        'th_tp_SEA_'+seasnames[idx_seas]+'_mean_fraction_contour'
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0, ylabel=ylabel)
