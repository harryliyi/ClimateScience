# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_vrcesm import readvrcesm, readvrcesm_3Dlevel
from modules.datareader.mod_dataread_obs_ERAinterim import readobs_ERAinterim, readobs_ERAinterim_3Dlevel

from modules.stats.mod_stats_clim import mon2clim, getstats_2D_ttest
from modules.plot.mod_plt_lines import plot_lines
from modules.plot.mod_plt_regrid import data_regrid
from modules.plot.mod_plt_contour import plot_2Dcontour

import numpy as np
import matplotlib.pyplot as plt
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
outdircesm = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/mse/'

# define pressure level
plevel = 992

outdircesm = outdircesm + 'surface/'

# set up variable names and file name
varname = 'MSE'
var_longname = 'moist static energy'
varstr = str(plevel)+"hPa_mse"
var_unit = r'$m^2/s^2$'


# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# define Indian region
reg_lats = [8, 25]
reg_lons = [70, 90]
reg_name = 'India'

# set data frequency
frequency = 'mon'

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

############################################################################
# physics constant
############################################################################
oro = 1000  # water density (kg/m^3)
g = 9.8  # gravitational constant (N/kg)
Cpd = 1.00464e3  # dry air specific heat capacity
Cpl = 1.81e3  # water vapor specific heat capacity
Lv = 2.501e6  # latent heat of vaporization

############################################################################
# read data
############################################################################

# read vrcesm

print('Reading VRCESM data...')
level = 29

# read Temperature
varname = 'T'

resolution = 'fv02'
varfname = 'TEMP'
case = 'vrseasia_AMIP_1979_to_2005'
model_t_var1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

resolution = 'fv09'
varfname = 'TEMP'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_t_var2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

resolution = 'fv09'
varfname = 'TEMP'
case = 'f09_f09_AMIP_1979_to_2005'
model_t_var3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

resolution = 'fv19'
varfname = 'TEMP'
case = 'f19_f19_AMIP_1979_to_2005'
model_t_var4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

print(model_t_var1.shape)
print(model_levs1)

# read Specific humidity
varname = 'Q'

resolution = 'fv02'
varfname = 'Q'
case = 'vrseasia_AMIP_1979_to_2005'
model_q_var1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

resolution = 'fv09'
varfname = 'Q'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_q_var2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

resolution = 'fv09'
varfname = 'Q'
case = 'f09_f09_AMIP_1979_to_2005'
model_q_var3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

resolution = 'fv19'
varfname = 'Q'
case = 'f19_f19_AMIP_1979_to_2005'
model_q_var4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

# read Geopotential height
varname = 'Z3'

resolution = 'fv02'
varfname = 'Z3'
case = 'vrseasia_AMIP_1979_to_2005'
model_z3_var1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

resolution = 'fv09'
varfname = 'Z3'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_z3_var2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

resolution = 'fv09'
varfname = 'Z3'
case = 'f09_f09_AMIP_1979_to_2005'
model_z3_var3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)

resolution = 'fv19'
varfname = 'Z3'
case = 'f19_f19_AMIP_1979_to_2005'
model_z3_var4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3Dlevel(
    varname, iniyear, endyear, resolution, varfname, case, frequency, level, latbounds, lonbounds, model_level=True)


# # read PS
# varname = 'PS'
#
# resolution = 'fv02'
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
#
# print(model_ps1.shape)

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

# calculate MSE
model_var1 = ((Cpd + Cpl * model_q_var1) * model_t_var1 + Lv * model_q_var1 + g * model_z3_var1) / 1000.
model_var2 = ((Cpd + Cpl * model_q_var2) * model_t_var2 + Lv * model_q_var2 + g * model_z3_var2) / 1000.
model_var3 = ((Cpd + Cpl * model_q_var3) * model_t_var3 + Lv * model_q_var3 + g * model_z3_var3) / 1000.
model_var4 = ((Cpd + Cpl * model_q_var4) * model_t_var4 + Lv * model_q_var4 + g * model_z3_var4) / 1000.

# # mask the region if ps<plevel
# model_var1 = np.ma.masked_where(model_ps1 < plevel, model_var1)
# model_var2 = np.ma.masked_where(model_ps2 < plevel, model_var2)
# model_var3 = np.ma.masked_where(model_ps3 < plevel, model_var3)
# model_var4 = np.ma.masked_where(model_ps4 < plevel, model_var4)
#
# model_var1[model_var1.mask] = np.nan
# model_var2[model_var2.mask] = np.nan
# model_var3[model_var3.mask] = np.nan
# model_var4[model_var4.mask] = np.nan

# read ERA-interim

print('Reading ERA-interim data...')
level = 59

# read Temperature
varname = 't'
varfanme = 'tmp'
frequency = 'monthly'

obs_t_var, obs_time, obs_levs, obs_lats, obs_lons = readobs_ERAinterim_3Dlevel(
    varname, iniyear, endyear, varfanme, frequency, level, latbounds, lonbounds, oceanmask=0, model_level=True)

# read Specific humidity
varname = 'q'
varfanme = 'sh'

obs_q_var, obs_time, obs_levs, obs_lats, obs_lons = readobs_ERAinterim_3Dlevel(
    varname, iniyear, endyear, varfanme, frequency, level, latbounds, lonbounds, oceanmask=0, model_level=True)

# # read Geopotential
# varname = 'z'
# varfanme = 'z3'
#
# obs_z_var, obs_time, obs_levs, obs_lats, obs_lons = readobs_ERAinterim_3Dlevel(
#     varname, iniyear, endyear, varfanme, frequency, level, latbounds, lonbounds, oceanmask=0)

print(obs_levs)

# # read Geopotential
# varname = 'sp'
# varfanme = 'ps'
#
# obs_ps, obs_time, obs_lats, obs_lons = readobs_ERAinterim(
#     varname, iniyear, endyear, varfanme, frequency, latbounds, lonbounds, oceanmask=0)
#
# obs_ps = obs_ps/100

# find regional lat/lon boundaries
obs_latl = np.argmin(np.abs(obs_lats - reg_lats[0]))
obs_latu = np.argmin(np.abs(obs_lats - reg_lats[1]))
obs_lonl = np.argmin(np.abs(obs_lons - reg_lons[0]))
obs_lonr = np.argmin(np.abs(obs_lons - reg_lons[1]))

# calculate MSE
obs_var = ((Cpd + Cpl * obs_q_var) * obs_t_var + Lv * obs_q_var) / 1000.

# # mask the region if ps<plevel
# obs_var = np.ma.masked_where(obs_ps < plevel, obs_var)
# obs_var[obs_var.mask] = np.nan

############################################################################
# calculate and plot monthly mean contour
############################################################################
# calculate monthly mean contour
print('Plotting for monthly mean contour...')

plot_list = monnames

# variable info
varname = 'surface Moist Static Energy'
varstr = str(plevel)+'hPa_mse'
var_unit = r'$kJ/kg$'

# calculate monthly mean
model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

obs_mean, obs_std = mon2clim(obs_var[:, :, :], opt=2)

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean...')

    plot_data = [model_mean1[idx, :, :], model_mean2[idx, :, :], model_mean3[idx, :, :], model_mean4[idx, :, :]]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(320, 364, 4)
    colormap = cm.rainbow

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname

    # without significance level
    fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)

############################################################################
# calculate and plot monthly mean meridional gradient of MSE
############################################################################
# calculate monthly mean meridional gradient of MSE
print('Plotting for monthly meridional gradient of MSE...')

# variable info
varname = 'surface Moist Static Energy'
varstr = str(plevel)+'hPa_mse'
var_unit = r'$kJ/kg$'

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'ERA-interim']

cesm_colors = ['red', 'yellow', 'green', 'blue', 'black']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed', 'solid']
xlabel = 'Latitude'
ylabel = varname + '[' + var_unit + ']'

xticks = np.arange(reg_lats[0], reg_lats[1]+1, 5)

model_x_lats1 = model_lats1[model_latl1: model_latu1 + 1]
model_x_lats2 = model_lats2[model_latl2: model_latu2 + 1]
model_x_lats3 = model_lats3[model_latl3: model_latu3 + 1]
model_x_lats4 = model_lats4[model_latl4: model_latu4 + 1]
obs_x_lats = obs_lats[obs_latl: obs_latu + 1]

x_data = [model_x_lats1, model_x_lats2, model_x_lats3, model_x_lats4, obs_x_lats]

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean meridional gradient...')

    model_grad1 = np.nanmean(model_mean1[idx, model_latl1: model_latu1 + 1, model_lonl1: model_lonr1 + 1], axis=1)
    model_grad2 = np.nanmean(model_mean2[idx, model_latl2: model_latu2 + 1, model_lonl2: model_lonr2 + 1], axis=1)
    model_grad3 = np.nanmean(model_mean3[idx, model_latl3: model_latu3 + 1, model_lonl3: model_lonr3 + 1], axis=1)
    model_grad4 = np.nanmean(model_mean4[idx, model_latl4: model_latu4 + 1, model_lonl4: model_lonr4 + 1], axis=1)

    obs_grad = np.nanmean(obs_mean[idx, obs_latl: obs_latu + 1, obs_lonl: obs_lonr + 1], axis=1)

    plot_data = [model_grad1, model_grad2, model_grad3, model_grad4, obs_grad]

    # line plot
    title = str(iniyear)+'-'+str(endyear)+' Longitudinally Averaged '+imonname+' mean '+varname+' over '+reg_name
    fname = 'vrseasia_'+varstr+'_'+reg_name+'_monthly_zonal_mean_gradient_'+str(idx+1)+'.pdf'
    plot_lines(x_data, plot_data, cesm_colors, cesm_line_types,
               cesm_legends, xlabel, ylabel, title, outdircesm+fname, multix=True, xticks=xticks)

############################################################################
# calculate and plot monthly mean difference
############################################################################
# calculate monthly mean difference
print('Plotting for monthly mean difference...')

# variable info
varname = 'surface Moist Static Energy'
varstr = str(plevel)+'hPa_mse'
var_unit = r'$kJ/kg$'

############################################################################
# plot against ERA-interim
project = 'ERA-interim'
projectstr = 'erainterim'
print('Plotting against '+project+'...')

# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons1, model_lats1)
obs_var_model1 = data_regrid(obs_var, obs_lons, obs_lats, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons2, model_lats2)
obs_var_model2 = data_regrid(obs_var, obs_lons, obs_lats, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons3, model_lats3)
obs_var_model3 = data_regrid(obs_var, obs_lons, obs_lats, lonsout, latsout)
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
obs_var_model4 = data_regrid(obs_var, obs_lons, obs_lats, lonsout, latsout)

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

obs_mean_model1, obs_std_model1 = mon2clim(obs_var_model1[:, :, :], opt=2)
obs_mean_model2, obs_std_model2 = mon2clim(obs_var_model2[:, :, :], opt=2)
obs_mean_model3, obs_std_model3 = mon2clim(obs_var_model3[:, :, :], opt=2)
obs_mean_model4, obs_std_model4 = mon2clim(obs_var_model4[:, :, :], opt=2)

# calculate degree of freedom
expdf = (endyear-iniyear)
refdf = (endyear-iniyear)

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' differences...')

    # calculate the mean difference and t-test results
    model_diff1, model_test1 = getstats_2D_ttest(
        model_mean1[idx, :, :], obs_mean_model1[idx, :, :], model_std1[idx, :, :], obs_std_model1[idx, :, :], expdf, refdf)
    model_diff2, model_test2 = getstats_2D_ttest(
        model_mean2[idx, :, :], obs_mean_model2[idx, :, :], model_std2[idx, :, :], obs_std_model2[idx, :, :], expdf, refdf)
    model_diff3, model_test3 = getstats_2D_ttest(
        model_mean3[idx, :, :], obs_mean_model3[idx, :, :], model_std3[idx, :, :], obs_std_model3[idx, :, :], expdf, refdf)
    model_diff4, model_test4 = getstats_2D_ttest(
        model_mean4[idx, :, :], obs_mean_model4[idx, :, :], model_std4[idx, :, :], obs_std_model4[idx, :, :], expdf, refdf)

    plot_data = [model_diff1, model_diff2, model_diff3, model_diff4]
    plot_test = [model_test1, model_test2, model_test3, model_test4]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-30, 32, 2)
    colormap = cm.RdBu_r

    title = str(iniyear)+'-'+str(endyear)+' '+imonname+' mean '+varname+' differece (Ref as '+project+')'
    fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_wtsig_'+str(idx+1)+'.pdf'

    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=1, sig_test=plot_test)

    # without significance levels
    fname = 'vrseasia_'+varstr+'_SEA_monthly_mean_contour_ref'+projectstr+'_nosig_'+str(idx+1)+'.pdf'
    plot_2Dcontour(plot_data, plot_lons, plot_lats, colormap, clevs, cesm_legends, lonbounds,
                   latbounds, varname, var_unit, title, outdircesm+fname, opt=0)
