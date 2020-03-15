# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_cesm import readcesm, readcesm_3D
from modules.plot.mod_plt_contour import plot_2Dcontour


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')


# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = "/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/"


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/omega/'

# set up variable names and file name
varname = 'OMEGA'
var_longname = r"$\omega 500$"
varstr = "500hPa_omega"
var_unit = r'$\times 10^{-3} Pa/s$'


# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'mon'

# define pressure level
plevel = 500

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# define ENSO yyears
years_elweak = [1980, 1983, 1987, 1988, 1992, 1995, 1998, 2003, 2005]
years_elmod = [1983, 1987, 1988, 1992, 1998, 2003]
years_laweak = [1984, 1985, 1989, 1996, 1999, 2000, 2001]
years_lamod = [1989, 1999, 2000]

years_elweakpre = [iyear-1 for iyear in years_elweak]
years_laweakpre = [iyear-1 for iyear in years_laweak]

############################################################################
# define functions
############################################################################


def cal_diff(var1, var2, std1, std2, n1, n2):
    res = var1-var2
    SE = np.sqrt((std1**2/n1) + (std2**2/n2))
    res_sig = res/SE

    return res, res_sig


############################################################################
# read data
############################################################################

# read vrcesm

print('Reading CESM data...')

# read PS
varname = 'PS'

resolution = 'fv19'
varfname = 'PS'
model_ps, model_time,  model_lats, model_lons = readcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)


model_ps = model_ps/100

varname = 'OMEGA'
varfname = 'OMEGA'
model_var, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

print(model_time)
print(model_var.shape)
print(model_levs)

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))


model_lev = np.argmin(np.abs(model_levs - plevel))
print(model_lev)

# select specific level and convert the unit to 10^-3
model_var = model_var[:, model_lev, :, :] * 1000


# mask the region if ps<plevel
model_var = np.ma.masked_where(model_ps < plevel, model_var)

print(model_var[0, :, :])


############################################################################
# calculate the monthly averaged values
############################################################################

for idx in range(12):
    select_el = np.in1d(model_time.year, years_elweak) & (model_time.month == months[idx])
    select_la = np.in1d(model_time.year, years_laweak) & (model_time.month == months[idx])
    time_temp = model_time[select_el]
    print(time_temp)
    var_el = model_var[select_el, :, :]
    var_la = model_var[select_la, :, :]

    var_el_mean = np.mean(var_el, axis=0)
    var_la_mean = np.mean(var_la, axis=0)
    var_el_std = np.std(var_el, axis=0)
    var_la_std = np.std(var_la, axis=0)

    var_diff, var_sig = cal_diff(var_el_mean, var_la_mean, var_el_std, var_la_std, len(years_elweak), len(years_laweak))
    var_elsig = var_sig
    var_lasig = var_sig
    var_elsig[:] = 0
    var_lasig[:] = 0

    plot_data = [var_el_mean, var_la_mean, var_diff]
    plot_lons = [model_lons, model_lons, model_lons]
    plot_lats = [model_lats, model_lats, model_lats]
    plot_test = [var_elsig, var_lasig, var_sig]

    legends = ['Under El Nino events', 'Under La Nina events', 'Differences (El Nino - La Nina)']

    clevs = np.arange(-30, 33, 3)
    colormap = cm.RdBu_r

    title = ' CESM monthly averaged '+var_longname+' in ENSO years: '+monnames[idx]
    fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_diff_'+str(idx+1)

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)

for idx in range(12):
    select_el = np.in1d(model_time.year, years_elweakpre) & (model_time.month == months[idx])
    select_la = np.in1d(model_time.year, years_laweakpre) & (model_time.month == months[idx])
    time_temp = model_time[select_el]
    print(time_temp)
    var_el = model_var[select_el, :, :]
    var_la = model_var[select_la, :, :]

    var_el_mean = np.mean(var_el, axis=0)
    var_la_mean = np.mean(var_la, axis=0)
    var_el_std = np.std(var_el, axis=0)
    var_la_std = np.std(var_la, axis=0)

    var_diff, var_sig = cal_diff(var_el_mean, var_la_mean, var_el_std, var_la_std, len(years_elweak), len(years_laweak))
    var_elsig = var_sig
    var_lasig = var_sig
    var_elsig[:] = 0
    var_lasig[:] = 0

    plot_data = [var_el_mean, var_la_mean, var_diff]
    plot_lons = [model_lons, model_lons, model_lons]
    plot_lats = [model_lats, model_lats, model_lats]
    plot_test = [var_elsig, var_lasig, var_sig]

    legends = ['Under El Nino events', 'Under La Nina events', 'Differences (El Nino - La Nina)']

    clevs = np.arange(-30, 33, 3)
    colormap = cm.RdBu_r

    title = ' CESM monthly averaged '+var_longname+' in ENSO years: (-1) '+monnames[idx]
    fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_diff_(-1)'+str(idx+1)

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)


#
# ############################################################################
# # calculate climatological mean and plot
# ############################################################################
#
# print('Plotting for seasonality...')
#
# model_mean1, model_std1 = mon2clim(
#     model_var1[:, model_latl1: model_latu1 + 1, model_lonl1: model_lonr1 + 1])
# model_mean2, model_std2 = mon2clim(
#     model_var2[:, model_latl2: model_latu2 + 1, model_lonl2: model_lonr2 + 1])
# model_mean3, model_std3 = mon2clim(
#     model_var3[:, model_latl3: model_latu3 + 1, model_lonl3: model_lonr3 + 1])
# model_mean4, model_std4 = mon2clim(
#     model_var4[:, model_latl4: model_latu4 + 1, model_lonl4: model_lonr4 + 1])
#
#
# ############################################################################
# # plot only for cesm
# plot_data = [model_mean1, model_mean2, model_mean3, model_mean4]
#
# plot_err = [model_std1, model_std2, model_std3, model_std4]
#
# cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
#                 'CESM-fv1.9x2.5']
#
# cesm_colors = ['red', 'yellow', 'green', 'blue']
# cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed']
#
# # line plot
# xlabel = 'Month'
# ylabel = var_longname + ' ' + var_unit
# title = str(iniyear) + ' to ' + str(endyear) + ' CESM ' + var_longname + ' climatology'
# fname = 'vrseasia_' + varstr + '_'+reg_name+'_clim_line_overland.pdf'
# plot_lines(months, plot_data, cesm_colors, cesm_line_types,
#            cesm_legends, xlabel, ylabel, title, outdir+fname, yerr=plot_err, xticks=months, xticknames=monnames)
#
#
# # bar plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# bar_width = 0.85/len(plot_data)
# index = months - bar_width*len(plot_data)/2
# opacity = 0.8
#
# for idx in range(len(plot_data)):
#     plt.bar(index+idx*bar_width, plot_data[idx], bar_width, alpha=opacity,
#             yerr=plot_err[idx], color=cesm_colors[idx], label=cesm_legends[idx])
#
# plt.legend(handlelength=4, fontsize=5)
#
# plt.xticks(months, monnames, fontsize=6)
# plt.yticks(fontsize=6)
# plt.xlabel('Month', fontsize=8)
# plt.ylabel(var_longname + ' ' + var_unit, fontsize=8)
#
#
# title = str(iniyear) + ' to ' + str(endyear) + ' CESM ' + var_longname + ' climatology'
# fname = 'vrseasia_' + varstr + '_'+reg_name+'_clim_bar_overland.pdf'
# plt.suptitle('CESM ' + var_longname + ' climatology', fontsize=9, y=0.95)
# plt.savefig(outdir+fname, bbox_inches='tight')
#
#
# ############################################################################
# # calculate and plot seasonal mean
# ############################################################################
#
# print('Plotting for seasonal mean...')
# # calculate seasonal mean
# plot_data = []
# plot_err = []
#
# plot_cesm_data = []
# plot_cesm_err = []
#
# seasons_list = ['DJF', 'MAM', 'JJA', 'SON']
# for idx in range(len(seasons_list)):
#
#     model_mean1, model_std1 = mon2clim(
#         model_var1[:, model_latl1: model_latu1 + 1, model_lonl1: model_lonr1 + 1], opt=4, season=seasons_list[idx])
#     model_mean2, model_std2 = mon2clim(
#         model_var2[:, model_latl2: model_latu2 + 1, model_lonl2: model_lonr2 + 1], opt=4, season=seasons_list[idx])
#     model_mean3, model_std3 = mon2clim(
#         model_var3[:, model_latl3: model_latu3 + 1, model_lonl3: model_lonr3 + 1], opt=4, season=seasons_list[idx])
#     model_mean4, model_std4 = mon2clim(
#         model_var4[:, model_latl4: model_latu4 + 1, model_lonl4: model_lonr4 + 1], opt=4, season=seasons_list[idx])
#
#     plot_cesm_data.extend([model_mean1, model_mean2, model_mean3, model_mean4])
#     plot_cesm_err.extend([model_std1, model_std2, model_std3, model_std4])
#
#
# # bar plot for cesm only
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ndatasets = 4
# bar_width = 0.85/ndatasets
# index = np.arange(1, 5) - bar_width*(ndatasets/2-0.5)
# opacity = 0.8
#
# cesm_shape_type = ['', '', '', '']
#
# for idx in range(ndatasets):
#     plt.bar(index+idx*bar_width, plot_cesm_data[idx::ndatasets], bar_width, alpha=opacity,
#             color=cesm_colors[idx], label=cesm_legends[idx], hatch=cesm_shape_type[idx])
#     plt.errorbar(index+idx*bar_width, plot_cesm_data[idx::ndatasets],
#                  yerr=plot_cesm_err[idx::ndatasets], elinewidth=0.5, ecolor='black', fmt='none', alpha=opacity)
#
# plt.legend(handlelength=4, fontsize=5)
#
# plt.xticks(np.arange(1, 5), seasons_list, fontsize=6)
# plt.yticks(fontsize=6)
# plt.xlabel('Season', fontsize=8)
# plt.ylabel(var_longname + ' ' + var_unit, fontsize=8)
#
#
# title = str(iniyear) + ' to ' + str(endyear) + ' CESM seasonal mean '+var_longname
# fname = 'vrseasia_' + varstr + '_'+reg_name+'_seasonal_mean_bar_overland.pdf'
# plt.suptitle('CESM seasonal mean '+var_longname, fontsize=9, y=0.95)
# plt.savefig(outdir+fname, bbox_inches='tight')
#
#
# ############################################################################
# # calculate and plot monthly mean contour
# ############################################################################
# # calculate monthly mean contour
# print('Plotting for monthly mean contour...')
#
#
# # regrid for contour plot
# # lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
# #
# #
# # model_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
# # model_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
# # model_var3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)
#
# # calculate monthly mean
#
# model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
# model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
# model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
# model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)
#
# plot_list = monnames
# plot_list.append('Annual')
#
# for idx, imonname in enumerate(plot_list):
#     print('plotting for '+imonname+' mean...')
#
#     # print(model_test1.shape)
#     plot_data = [model_mean1[idx], model_mean2[idx], model_mean3[idx], model_mean4[idx]]
#     plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
#     plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]
#
#     legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']
#
#     clevs = np.arange(-65, 70, 5)
#     colormap = cm.bwr
#
#     title = str(iniyear)+' to '+str(endyear)+' '+imonname+' mean '+var_longname
#     if idx != 12:
#         fname = 'vrseasia_' + varstr + '_SEA_monthly_mean_contour_'+str(idx+1)+'.pdf'
#     else:
#         fname = 'vrseasia_' + varstr + '_SEA_annual_mean_contour.pdf'
#
#     plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
#                    latbounds, varname, var_unit, title, outdir+fname, opt=0)
#
#
# print('Plotting for monthly mean contour difference...')
#
#
# # regrid for contour plot
# lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
#
# model_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
# model_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
# model_var3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)
#
# # calculate monthly mean
#
# model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
# model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
# model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
# model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)
#
# model_mean1 = model_mean1 - model_mean2
# model_mean3 = model_mean3 - model_mean4
#
# for idx, imonname in enumerate(plot_list):
#     print('plotting for '+imonname+' mean...')
#
#     # print(model_test1.shape)
#     plot_data = [model_mean1[idx], model_mean3[idx]]
#     plot_lons = [model_lons4, model_lons4]
#     plot_lats = [model_lats4, model_lats4]
#
#     legends = ['CESM-vrseasia vs CESM-ne30', 'CESM-fv0.9x1.25 vs CESM-fv1.9x2.5']
#
#     clevs = np.arange(-30, 33, 3)
#     colormap = cm.RdBu_r
#
#     title = str(iniyear)+' to '+str(endyear)+' '+imonname+' mean '+var_longname
#     if idx != 12:
#         fname = 'vrseasia_' + varstr + '_SEA_monthly_mean_contour_diff_'+str(idx+1)+'.pdf'
#     else:
#         fname = 'vrseasia_' + varstr + '_SEA_annual_mean_contour_diff.pdf'
#
#     plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
#                    latbounds, varname, var_unit, title, outdir+fname, opt=0)
