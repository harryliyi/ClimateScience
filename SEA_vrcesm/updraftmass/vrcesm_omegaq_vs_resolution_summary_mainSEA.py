# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_vrcesm import readvrcesm, readvrcesm_3Dlevel
from modules.stats.mod_stats_clim import mon2clim
from modules.plot.mod_plt_lines import plot_lines
from modules.plot.mod_plt_regrid import data_regrid
from modules.plot.mod_plt_contour import plot_2Dcontour

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.cm as cm
plt.switch_backend('agg')


# set up data directories and filenames
case1 = "vrseasia_AMIP_1979_to_2005"
case2 = "ne30_ne30_AMIP_1979_to_2005"
case3 = "f19_f19_AMIP_1979_to_2005"
case4 = "f09_f09_AMIP_1979_to_2005"

expdir1 = "/scratch/d/dylan/harryli/cesm1/vrcesm/fields_archive/"+case1+"/atm/hist/"
expdir2 = "/scratch/d/dylan/harryli/cesm1/vrcesm/fields_archive/"+case2+"/atm/hist/"
expdir3 = "/scratch/d/dylan/harryli/cesm1/vrcesm/fields_archive/"+case3+"/atm/hist/"
expdir4 = "/scratch/d/dylan/harryli/cesm1/vrcesm/fields_archive/"+case4+"/atm/hist/"


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/updraftmass/'

# define pressure level
plevel = 850

# set up variable names and file name
varname = 'updraft mass'
var_longname = 'updraft mass'
varstr = str(plevel)+"hPa_updraftmass"
var_unit = 'mm/day'


# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-10, 25]
lonbounds = [90, 130]
# latbounds = [-12, 40]
# lonbounds = [65, 150]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# reg_lats = [-10, 25]
# reg_lons = [90, 130]

# set data frequency
frequency = 'day'

# select bins for histogram
nbins = 120

# month series
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

outdir = outdir+'850hPa_fake/'

############################################################################
# physics constant
############################################################################
oro = 1000  # water density (kg/m^3)
g = 9.8  # gravitational constant (N/kg)

############################################################################
# define functions
############################################################################


def get_scaling(var_day, var_mon_lev, var_mon_surf, var_day_time, var_mon_time):
    res = var_day

    for idx in range(len(var_mon_time)):
        # print(var_day_time[(var_day_time.year == var_mon_time[idx].year)
        #                    & (var_day_time.month == var_mon_time[idx].month)])
        temp = res[(var_day_time.year == var_mon_time[idx].year) & (var_day_time.month == var_mon_time[idx].month)]
        for ii in range(temp.shape[0]):
            temp[ii, :, :] = temp[ii, :, :] * var_mon_lev[idx, :, :]/var_mon_surf[idx, :, :]
        res[(var_day_time.year == var_mon_time[idx].year) & (var_day_time.month == var_mon_time[idx].month)] = temp

    return res


def get_vars(time, var, months):
    if np.isscalar(months):
        res_var = var[time.month == months, :, :]
    else:
        # print(time[np.in1d(time.month, months)])
        res_var = var[np.in1d(time.month, months), :, :]

    return res_var


def get_ranges(var1, var2, edges):
    res_up = np.zeros(len(edges)-1)
    res_bot = np.zeros(len(edges)-1)
    res_med = np.zeros(len(edges)-1)
    for idx in range(len(edges)-1):
        temp = var2[(var1 >= edges[idx]) & (var1 < edges[idx+1])]
        res_up[idx] = np.percentile(temp, 75)
        res_bot[idx] = np.percentile(temp, 25)
        res_med[idx] = np.percentile(temp, 50)

    return res_bot, res_med, res_up


def get_quantiles(var1, var2, edges):
    res_up = np.zeros(len(edges)-1)
    res_bot = np.zeros(len(edges)-1)
    res_med = np.zeros(len(edges)-1)
    for idx in range(len(edges)-1):
        percentl = 100*len(var1[var1 < edges[idx]])/len(var1)
        percentu = 100*len(var1[var1 < edges[idx+1]])/len(var1)
        percentl_val = np.percentile(var2, percentl)
        percentu_val = np.percentile(var2, percentu)
        temp = var2[(var2 >= percentl_val) & (var2 < percentu_val)]
        res_up[idx] = temp.max()
        res_bot[idx] = temp.min()
        res_med[idx] = np.median(temp)

    return res_bot, res_med, res_up


def get_rms(var):
    return np.sqrt(np.mean(var**2))

############################################################################
# read data
############################################################################


# read vrcesm
print('Reading VRCESM data...')

# read OMEGA
varname = 'OMEGA'

resolution = 'fv09'
varfname = 'OMEGA'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, plevel, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'OMEGA'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, plevel, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'OMEGA'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, plevel, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'OMEGA'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, plevel, latbounds, lonbounds)

print(model_var1.shape)
print(model_levs1)

# read Q
varname = 'QREFHT'

resolution = 'fv09'
varfname = 'QREFHT'
case = 'vrseasia_AMIP_1979_to_2005'
model_q_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'QREFHT'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_q_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'QREFHT'
case = 'f09_f09_AMIP_1979_to_2005'
model_q_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'QREFHT'
case = 'f19_f19_AMIP_1979_to_2005'
model_q_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)


# read monthly Q to scale
varname = 'Q'

resolution = 'fv09'
varfname = 'Q'
case = 'vrseasia_AMIP_1979_to_2005'
model_q850_var1, model_mon_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, 'mon', plevel, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'Q'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_q850_var2, model_mon_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, 'mon', plevel, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'Q'
case = 'f09_f09_AMIP_1979_to_2005'
model_q850_var3, model_mon_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, 'mon', plevel, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'Q'
case = 'f19_f19_AMIP_1979_to_2005'
model_q850_var4, model_mon_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, 'mon', plevel, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'Q'
case = 'vrseasia_AMIP_1979_to_2005'
model_q995_var1, model_mon_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, 'mon', 995, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'Q'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_q995_var2, model_mon_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, 'mon', 995, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'Q'
case = 'f09_f09_AMIP_1979_to_2005'
model_q995_var3, model_mon_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, 'mon', 995, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'Q'
case = 'f19_f19_AMIP_1979_to_2005'
model_q995_var4, model_mon_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, 'mon', 995, latbounds, lonbounds)

model_q_var1 = get_scaling(model_q_var1, model_q850_var1, model_q995_var1, model_time1, model_mon_time1)
model_q_var2 = get_scaling(model_q_var2, model_q850_var2, model_q995_var2, model_time2, model_mon_time2)
model_q_var3 = get_scaling(model_q_var3, model_q850_var3, model_q995_var3, model_time3, model_mon_time3)
model_q_var4 = get_scaling(model_q_var4, model_q850_var4, model_q995_var4, model_time4, model_mon_time4)


# model_q_var1[model_q_var1.mask] = np.nan
# model_q_var2[model_q_var2.mask] = np.nan
# model_q_var3[model_q_var3.mask] = np.nan
# model_q_var4[model_q_var4.mask] = np.nan

# read PRECT
varname = 'PRECT'

resolution = 'fv09'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
model_pre_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_pre_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_pre_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_pre_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

model_pre_var1 = model_pre_var1 * 86400 * 1000
model_pre_var2 = model_pre_var2 * 86400 * 1000
model_pre_var3 = model_pre_var3 * 86400 * 1000
model_pre_var4 = model_pre_var4 * 86400 * 1000

# model_pre_var1[model_pre_var1.mask] = np.nan
# model_pre_var2[model_pre_var2.mask] = np.nan
# model_pre_var3[model_pre_var3.mask] = np.nan
# model_pre_var4[model_pre_var4.mask] = np.nan
#
# model_var1[model_pre_var1.mask] = np.nan
# model_var2[model_pre_var2.mask] = np.nan
# model_var3[model_pre_var3.mask] = np.nan
# model_var4[model_pre_var4.mask] = np.nan

# # read PS
# varname = 'PS'
#
# resolution = 'fv09'
# varfname = 'PS'
# case = 'vrseasia_AMIP_1979_to_2005'
# model_ps1, model_time1,  model_lats1, model_lons1 = readvrcesm(
#     varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)
#
# resolution = 'fv09'
# varfname = 'PS'
# case = 'ne30_ne30_AMIP_1979_to_2005'
# model_ps2, model_time2,  model_lats2, model_lons2 = readvrcesm(
#     varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)
#
# resolution = 'fv09'
# varfname = 'PS'
# case = 'f09_f09_AMIP_1979_to_2005'
# model_ps3, model_time3,  model_lats3, model_lons3 = readvrcesm(
#     varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)
#
# resolution = 'fv19'
# varfname = 'PS'
# case = 'f19_f19_AMIP_1979_to_2005'
# model_ps4, model_time4, model_lats4, model_lons4 = readvrcesm(
#     varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)
#
# model_ps1 = model_ps1/100
# model_ps2 = model_ps2/100
# model_ps3 = model_ps3/100
# model_ps4 = model_ps4/100

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


# calculate the updraft mass, convert to mm/day
model_var1 = -1.0 * model_var1 * model_q_var1/oro/g*86400*1000
model_var2 = -1.0 * model_var2 * model_q_var2/oro/g*86400*1000
model_var3 = -1.0 * model_var3 * model_q_var3/oro/g*86400*1000
model_var4 = -1.0 * model_var4 * model_q_var4/oro/g*86400*1000

# # mask the region if ps<plevel
# model_var1 = np.ma.masked_where(model_ps1 < plevel, model_var1)
# model_var2 = np.ma.masked_where(model_ps2 < plevel, model_var2)
# model_var3 = np.ma.masked_where(model_ps3 < plevel, model_var3)
# model_var4 = np.ma.masked_where(model_ps4 < plevel, model_var4)

############################################################################
# calculate climatological mean and plot
############################################################################

# for idx_seas in range(len(seasons)):
#     print('Plot for '+seasnames[idx_seas]+':')
#
#     model_var_sub1 = get_vars(model_time1, model_var1, seasons[idx_seas])
#     model_var_sub2 = get_vars(model_time2, model_var2, seasons[idx_seas])
#     model_var_sub3 = get_vars(model_time3, model_var3, seasons[idx_seas])
#     model_var_sub4 = get_vars(model_time4, model_var4, seasons[idx_seas])
#
#     model_var_mean1 = np.mean(model_var_sub1, axis=0)
#     model_var_mean2 = np.mean(model_var_sub2, axis=0)
#     model_var_mean3 = np.mean(model_var_sub3, axis=0)
#     model_var_mean4 = np.mean(model_var_sub4, axis=0)
#
#     # cesm only
#     plot_data = [model_var_mean1, model_var_mean2, model_var_mean3, model_var_mean4]
#     plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
#     plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]
#
#     legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']
#
#     clevs = np.arange(-12, 13, 1)
#     colormap = cm.bwr
#
#     title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + \
#         ' mean '+str(plevel)+'hPa '+var_longname+' over '+reg_name
#     fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_' + varstr + '_SEA_'+seasnames[idx_seas]+'_mean_contour'
#     plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
#                    latbounds, varname, var_unit, title, outdir+fname, opt=0)
#
#     # plot for difference
#     print('plot for difference...')
#
#     lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
#     model_var_regrid1 = data_regrid(model_var_sub1, model_lons1, model_lats1, lonsout, latsout)
#     model_var_regrid2 = data_regrid(model_var_sub2, model_lons2, model_lats2, lonsout, latsout)
#     model_var_regrid3 = data_regrid(model_var_sub3, model_lons3, model_lats3, lonsout, latsout)
#     model_var_mean1 = np.mean(model_var_regrid1, axis=0)
#     model_var_mean2 = np.mean(model_var_regrid2, axis=0)
#     model_var_mean3 = np.mean(model_var_regrid3, axis=0)
#
#     model_var_mean1 = model_var_mean1 - model_var_mean2
#     model_var_mean3 = model_var_mean3 - model_var_mean4
#
#     plot_data = [model_var_mean1, model_var_mean3]
#     plot_lons = [model_lons4, model_lons4]
#     plot_lats = [model_lats4, model_lats4]
#
#     legends = ['CESM-vrseasia vs CESM-ne30', 'CESM-fv0.9x1.25 vs CESM-fv1.9x2.5']
#
#     clevs = np.arange(-6, 6.4, 0.4)
#     colormap = cm.RdBu_r
#
#     title = str(iniyear)+' to '+str(endyear)+' '+seasnames[idx_seas] + \
#         ' mean '+str(plevel)+'hPa '+var_longname+' differences over '+reg_name
#     fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_' + varstr + '_SEA_'+seasnames[idx_seas]+'_mean_contour_diff'
#
#     plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
#                    latbounds, varname, var_unit, title, outdir+fname, opt=0)

############################################################################
# plot histogram
############################################################################

print('plot for histogram...')

# lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
# model_pre_var1 = data_regrid(model_pre_var1, model_lons1, model_lats1, lonsout, latsout)
# model_pre_var2 = data_regrid(model_pre_var2, model_lons2, model_lats2, lonsout, latsout)
# model_pre_var3 = data_regrid(model_pre_var3, model_lons3, model_lats3, lonsout, latsout)
#
# model_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
# model_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
# model_var3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)

model_pre_sub1 = model_pre_var1[:, model_latl1: model_latu1+1, model_lonl1: model_lonr1+1]
model_pre_sub2 = model_pre_var2[:, model_latl2: model_latu2+1, model_lonl2: model_lonr2+1]
model_pre_sub3 = model_pre_var3[:, model_latl3: model_latu3+1, model_lonl3: model_lonr3+1]
model_pre_sub4 = model_pre_var4[:, model_latl4: model_latu4+1, model_lonl4: model_lonr4+1]

model_var_sub1 = model_var1[:, model_latl1: model_latu1+1, model_lonl1: model_lonr1+1]
model_var_sub2 = model_var2[:, model_latl2: model_latu2+1, model_lonl2: model_lonr2+1]
model_var_sub3 = model_var3[:, model_latl3: model_latu3+1, model_lonl3: model_lonr3+1]
model_var_sub4 = model_var4[:, model_latl4: model_latu4+1, model_lonl4: model_lonr4+1]

model_pre_sub1 = model_pre_sub1.flatten()
model_pre_sub2 = model_pre_sub2.flatten()
model_pre_sub3 = model_pre_sub3.flatten()
model_pre_sub4 = model_pre_sub4.flatten()

model_var_sub1 = model_var_sub1.flatten()
model_var_sub2 = model_var_sub2.flatten()
model_var_sub3 = model_var_sub3.flatten()
model_var_sub4 = model_var_sub4.flatten()

# model_pre_sub1 = model_pre_sub1[~np.isnan(model_pre_sub1)]
# model_pre_sub2 = model_pre_sub2[~np.isnan(model_pre_sub2)]
# model_pre_sub3 = model_pre_sub3[~np.isnan(model_pre_sub3)]
# model_pre_sub4 = model_pre_sub4[~np.isnan(model_pre_sub4)]
#
# model_var_sub1 = model_var_sub1[~np.isnan(model_var_sub1)]
# model_var_sub2 = model_var_sub2[~np.isnan(model_var_sub2)]
# model_var_sub3 = model_var_sub3[~np.isnan(model_var_sub3)]
# model_var_sub4 = model_var_sub4[~np.isnan(model_var_sub4)]

model_pre_sub1 = model_pre_sub1[model_var_sub1 > 0.]
model_pre_sub2 = model_pre_sub2[model_var_sub2 > 0.]
model_pre_sub3 = model_pre_sub3[model_var_sub3 > 0.]
model_pre_sub4 = model_pre_sub4[model_var_sub4 > 0.]

model_var_sub1 = model_var_sub1[model_var_sub1 > 0.]
model_var_sub2 = model_var_sub2[model_var_sub2 > 0.]
model_var_sub3 = model_var_sub3[model_var_sub3 > 0.]
model_var_sub4 = model_var_sub4[model_var_sub4 > 0.]


print('get rms')
model_var_rms = [get_rms(model_var_sub1), get_rms(model_var_sub2), get_rms(model_var_sub3), get_rms(model_var_sub4)]
model_pre_rms = [get_rms(model_pre_sub1), get_rms(model_pre_sub2), get_rms(model_pre_sub3), get_rms(model_pre_sub4)]
model_res = [28, 110, 110, 220]
model_res_steps = np.arange(20, 510, 10)

yval = [get_rms(model_var_sub1), get_rms(model_var_sub2), get_rms(model_var_sub3), get_rms(model_var_sub4),
        get_rms(model_pre_sub1), get_rms(model_pre_sub2), get_rms(model_pre_sub3), get_rms(model_pre_sub4)]
xval = [28, 110, 110, 220, 28, 110, 110, 220]

var_slope, intercept, r_value, p_value, var_std_err = ss.linregress(np.log10(model_res), np.log10(model_var_rms))
print(var_slope)
var_regress = 10**intercept*model_res_steps**var_slope
pre_slope, intercept, r_value, p_value, pre_std_err = ss.linregress(np.log10(model_res), np.log10(model_pre_rms))
print(pre_slope)
pre_regress = 10**intercept*model_res_steps**pre_slope
val_slope, intercept, r_value, p_value, val_std_err = ss.linregress(np.log10(xval), np.log10(yval))
print(val_slope)
val_regress = 10**intercept*model_res_steps**val_slope

print('plot updraft mass vs resolution...')
plt.clf()
fig = plt.figure()

ax = fig.add_subplot(111)
for idx in range(4):
    ax.scatter(model_res[idx], model_var_rms[idx], marker='o', c=cesm_colors[idx], alpha=0.7)
    ax.scatter(model_res[idx], model_pre_rms[idx], marker='s', c=cesm_colors[idx], alpha=0.7)

ax.plot(model_res_steps, var_regress, c='grey', linestyle='solid', linewidth=1.5, alpha=0.6, label='slope='+str(round(var_slope, 2))+r'$\pm$'+str(round(var_std_err, 2)))
ax.plot(model_res_steps, pre_regress, c='brown', linestyle='solid', linewidth=1.5, alpha=0.6, label='slope='+str(round(pre_slope, 2))+r'$\pm$'+str(round(pre_std_err, 2)))
ax.legend(fontsize=5)
ax.set_xscale('log')
ax.set_xticks([20, 100, 200])
ax.set_xticklabels([20, 100, 200], fontsize=5)
ax.set_xlim(20, 450)
ax.set_yscale('log')
ax.set_yticks([10, 15, 20])
ax.set_yticklabels([10, 15, 20], fontsize=5)
ax.set_ylim(8, 25)
ax.get_yaxis().set_major_formatter(tick.ScalarFormatter())

ax.set_xlabel('Res. [km]', fontsize=5)
ax.set_ylabel('RMS [mm/day]', fontsize=5)
ax.minorticks_off()

title = str(iniyear)+' to '+str(endyear)+' updraft mass and prect vs resolution over '+reg_name
fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_updraftmass_prect_vs_resolution_mainSEA'
plt.savefig(outdir+fname+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)


# model_var_sub1 = abs(model_var_sub1)
# model_var_sub2 = abs(model_var_sub2)
# model_var_sub3 = abs(model_var_sub3)
# model_var_sub4 = abs(model_var_sub4)

print('plot prect vs updraft mass...')
plt.clf()
fig = plt.figure()

plot_data1 = [model_var_sub1, model_var_sub2, model_var_sub3, model_var_sub4]
plot_data2 = [model_pre_sub1, model_pre_sub2, model_pre_sub3, model_pre_sub4]

ax = fig.add_subplot(211)
for idx in range(4):
    thres = np.percentile(plot_data1[idx], 99.9)
    ranges = np.arange(0., thres, thres/nbins)
    # ranges = np.append(ranges, plot_data1[idx].max())
    bins = ranges[:-1]+(ranges[1]-ranges[0])/2
    res_bot, res_med, res_up = get_ranges(plot_data1[idx], plot_data2[idx], ranges)
    ax.plot(bins, res_med, c=cesm_colors[idx], linestyle='solid', linewidth=1.5, label=cesm_legends[idx], alpha=0.8)
    ax.fill_between(bins, res_bot, res_up, facecolor=cesm_colors[idx], alpha=0.4)

ax.legend(handlelength=4, fontsize=5)
xticks = np.arange(0, 200, 50)
ax.plot(xticks, xticks, c='black', linestyle='dashed', linewidth=1.5, alpha=0.3)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, fontsize=5)
ax.set_xlim(0, xticks[-1])
# ax.set_xlabel(r'$-\omega q/g$'+' ['+var_unit+']', fontsize=6)
yticks = np.arange(0, 200, 50)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=5)
ax.set_ylabel('Precip R'+' ['+var_unit+']', fontsize=6)
ax.set_ylim(0, yticks[-1])

ax.text(0.95, 0.05, '(a)', transform=ax.transAxes, color='black', fontsize=7)

# model_pre_sub1 = model_pre_sub1[model_pre_sub1 > 1.]
# model_pre_sub2 = model_pre_sub2[model_pre_sub2 > 1.]
# model_pre_sub3 = model_pre_sub3[model_pre_sub3 > 1.]
# model_pre_sub4 = model_pre_sub4[model_pre_sub4 > 1.]

ax = fig.add_subplot(212)
for idx in range(4):

    thres = np.percentile(plot_data1[idx], 99.9)
    edges = np.arange(0., thres, thres/nbins)
    # edges = np.append(edges, plot_data1[idx].max())
    hist, edges = np.histogram(plot_data1[idx], bins=edges, density=True)
    bins = edges[:-1]+(edges[1]-edges[0])/2
    # print(edges)
    ax.plot(bins, hist, c=cesm_colors[idx], linestyle='solid', linewidth=1.5, alpha=0.8)

    thres = np.percentile(plot_data2[idx], 99.9)
    edges = np.arange(0., thres, thres/nbins)
    # edges = np.append(edges, plot_data2[idx].max())
    hist, edges = np.histogram(plot_data2[idx], bins=edges, density=True)
    bins = edges[:-1]+(edges[1]-edges[0])/2
    # print(edges)
    ax.plot(bins, hist, c=cesm_colors[idx], linestyle='dashed', linewidth=1.5, alpha=0.8)


ax.legend([r'$P(-\omega q/g)$', 'P(R)'], handlelength=5, fontsize=5, loc='lower left')
xticks = np.arange(0, 200, 50)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, fontsize=5)
ax.set_xlim(0, xticks[-1])
ax.yaxis.set_tick_params(labelsize=5)
ax.set_xlabel(r'$-\omega q/g$'+' ['+var_unit+']', fontsize=6)
ax.set_yscale('log')
ax.set_ylabel(r'$P(-\omega q/g), P(R)$', fontsize=6)

ax.text(0.95, 0.05, '(b)', transform=ax.transAxes, color='black', fontsize=7)

ax.minorticks_off()

# insert a resolution figure
ax2 = fig.add_axes([0.69, 0.25, 0.2, 0.2])

for idx in range(4):
    ax2.scatter(model_res[idx], model_var_rms[idx], marker='o', s=5, c=cesm_colors[idx], alpha=0.7)
    ax2.scatter(model_res[idx], model_pre_rms[idx], marker='s', s=5, c=cesm_colors[idx], alpha=0.7)

ax2.plot(model_res_steps, var_regress, c='grey', linestyle='solid', linewidth=1.5, alpha=0.6, label='slope='+str(round(var_slope, 2))+r'$\pm$'+str(round(var_std_err, 2)))
# ax2.plot(model_res_steps, pre_regress, c='brown', linestyle='solid', linewidth=1.5, alpha=0.6)
ax2.legend(fontsize=5)
ax2.set_xscale('log')
ax2.set_xticks([20, 100, 200])
ax2.set_xticklabels([20, 100, 200], fontsize=5)
ax2.set_xlim(20, 450)
ax2.set_yscale('log')
ax2.set_yticks([10, 15, 20])
ax2.set_yticklabels([10, 15, 20], fontsize=5)
ax2.set_ylim(8, 25)
ax2.get_yaxis().set_major_formatter(tick.ScalarFormatter())

ax2.set_xlabel('Res. [km]', fontsize=6)
ax2.set_ylabel('RMS [mm/day]', fontsize=6)
ax2.minorticks_off()


title = str(iniyear)+' to '+str(endyear)+' prect vs updraft mass over '+reg_name
fname = 'vrseasia_'+str(iniyear)+'to'+str(endyear)+'_prect_vs_updraftmass_mainSEA'

plt.savefig(outdir+fname+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
