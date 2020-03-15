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
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/TT/'

# set up variable names and file name
varname = 'T'
var_longname = 'Tropospheric Temperature'
varstr = "TT"
var_unit = 'K'


# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-30, 50]
lonbounds = [40, 180]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'mon'

# define pressure level
phigh = 700
plow = 200

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plot_mons = [1, 5, 7, 9]

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

resolution = 'fv19'
varname = 'T'
varfname = 'T'
model_var, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

model_var = model_var - 273.16
print(model_time)
print(model_var.shape)
print(model_levs)

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))
model_levl = np.argmin(np.abs(model_levs - plow))
model_levh = np.argmin(np.abs(model_levs - phigh))


# select specific level and convert the unit to 10^-3
model_var = np.mean(model_var[:, model_levl:model_levh + 1, :, :], axis=1)

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

    plot_data = [var_el_mean, var_la_mean]
    plot_lons = [model_lons, model_lons]
    plot_lats = [model_lats, model_lats]
    plot_test = [var_elsig, var_lasig]

    legends = ['a) Under El Nino events', 'b) Under La Nina events']

    clevs = np.arange(-30, -10, 1)
    colormap = cm.RdBu_r

    title = ' CESM monthly averaged '+var_longname+' in ENSO years: '+monnames[idx]
    fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_'+str(idx+1)

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)

    plot_data = [var_diff]
    plot_lons = [model_lons]
    plot_lats = [model_lats]
    plot_test = [var_sig]

    legends = ['Differences (El Nino - La Nina)']

    clevs = np.arange(-2.5, 2.75, 0.25)
    colormap = cm.RdBu_r

    title = ' CESM monthly averaged '+var_longname+' in ENSO years: '+monnames[idx]
    fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_diff_'+str(idx+1)

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                   latbounds, varname, var_unit, title, outdir+fname, opt=0)

plot_data = []
plot_lons = []
plot_lats = []
plot_test = []
for idx in range(4):
    select_el = np.in1d(model_time.year, years_elweak) & (model_time.month == plot_mons[idx])
    select_la = np.in1d(model_time.year, years_laweak) & (model_time.month == plot_mons[idx])
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

    plot_data.append(var_diff)
    plot_lons.append(model_lons)
    plot_lats.append(model_lats)
    plot_test.append(var_sig)

legends = ['a) January', 'b) May', 'c) July', 'd) September']

clevs = np.arange(-2.5, 2.75, 0.25)
colormap = cm.RdBu_r

title = ' CESM differences in monthly averaged '+var_longname+' in ENSO years'
fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_diff_allinone'

plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
               latbounds, var_longname, var_unit, title, outdir+fname, opt=0)

# for idx in range(12):
#     select_el = np.in1d(model_time.year, years_elweakpre) & (model_time.month == months[idx])
#     select_la = np.in1d(model_time.year, years_laweakpre) & (model_time.month == months[idx])
#     time_temp = model_time[select_el]
#     print(time_temp)
#     var_el = model_var[select_el]
#     var_la = model_var[select_la]
#
#     var_el_mean = np.mean(var_el, axis=0)
#     var_la_mean = np.mean(var_la, axis=0)
#     var_el_std = np.std(var_el, axis=0)
#     var_la_std = np.std(var_la, axis=0)
#
#     var_diff, var_sig = cal_diff(var_el_mean, var_la_mean, var_el_std, var_la_std, len(years_elweak), len(years_laweak))
#     var_elsig = var_sig
#     var_lasig = var_sig
#     var_elsig[:] = 0
#     var_lasig[:] = 0
#
#     plot_data = [var_el_mean, var_la_mean, var_diff]
#     plot_lons = [model_lons, model_lons, model_lons]
#     plot_lats = [model_lats, model_lats, model_lats]
#     plot_test = [var_elsig, var_lasig, var_sig]
#
#     legends = ['Under El Nino events', 'Under La Nina events', 'Differences (El Nino - La Nina)']
#
#     clevs = np.arange(-3, 3.2, 0.2)
#     colormap = cm.RdBu_r
#
#     title = ' CESM monthly averaged '+var_longname+' in ENSO years: (-1) '+monnames[idx]
#     fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_diff_(-1)'+str(idx+1)
#
#     plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
#                    latbounds, varname, var_unit, title, outdir+fname, opt=0)
