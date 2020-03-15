# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_vrcesm import readvrcesm, readvrcesm_3D
from modules.stats.mod_stats_clim import mon2clim
from modules.plot.mod_plt_regrid import data_regrid
from modules.plot.mod_plt_contour import plot_2Dvectorcontour

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
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/circulation/'

# define pressure level
plevel = 850

# set up variable names and file name
varname = 'Wind'
var_longname = str(plevel)+' wind'
varstr = str(plevel)+'hPa_wind'
var_unit = 'm/s'
vector_unit = r'$m/s$'

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

# define top layer
ptop = 200

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

############################################################################
# read data
############################################################################

# read vrcesm


print('Reading VRCESM data...')

# read U
varname = 'U'

resolution = 'fv09'
varfname = 'U_WIND'
case = 'vrseasia_AMIP_1979_to_2005'
model_u1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'U_WIND'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_u2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'U_WIND'
case = 'f09_f09_AMIP_1979_to_2005'
model_u3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'U_WIND'
case = 'f19_f19_AMIP_1979_to_2005'
model_u4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

# read V
varname = 'V'

resolution = 'fv09'
varfname = 'V_WIND'
case = 'vrseasia_AMIP_1979_to_2005'
model_v1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'V_WIND'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_v2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'V_WIND'
case = 'f09_f09_AMIP_1979_to_2005'
model_v3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'V_WIND'
case = 'f19_f19_AMIP_1979_to_2005'
model_v4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

print(model_u1.shape)
print(model_levs1)

# read PS
varname = 'PS'

resolution = 'fv09'
varfname = 'PS'
case = 'vrseasia_AMIP_1979_to_2005'
model_ps1, model_time1,  model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'PS'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_ps2, model_time2,  model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'PS'
case = 'f09_f09_AMIP_1979_to_2005'
model_ps3, model_time3,  model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'PS'
case = 'f19_f19_AMIP_1979_to_2005'
model_ps4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

model_ps1 = model_ps1/100
model_ps2 = model_ps2/100
model_ps3 = model_ps3/100
model_ps4 = model_ps4/100

# find the level
model_lev1 = np.argmin(np.abs(model_levs1 - plevel))
model_lev2 = np.argmin(np.abs(model_levs2 - plevel))
model_lev3 = np.argmin(np.abs(model_levs3 - plevel))
model_lev4 = np.argmin(np.abs(model_levs4 - plevel))

# select specific level
model_u1 = model_u1[:, model_lev1, :, :]
model_u2 = model_u2[:, model_lev2, :, :]
model_u3 = model_u3[:, model_lev3, :, :]
model_u4 = model_u4[:, model_lev4, :, :]

model_v1 = model_v1[:, model_lev1, :, :]
model_v2 = model_v2[:, model_lev2, :, :]
model_v3 = model_v3[:, model_lev3, :, :]
model_v4 = model_v4[:, model_lev4, :, :]

# mask the region if ps<plevel
model_u1 = np.ma.masked_where(model_ps1 < plevel, model_u1)
model_u2 = np.ma.masked_where(model_ps2 < plevel, model_u2)
model_u3 = np.ma.masked_where(model_ps3 < plevel, model_u3)
model_u4 = np.ma.masked_where(model_ps4 < plevel, model_u4)

model_v1 = np.ma.masked_where(model_ps1 < plevel, model_v1)
model_v2 = np.ma.masked_where(model_ps2 < plevel, model_v2)
model_v3 = np.ma.masked_where(model_ps3 < plevel, model_v3)
model_v4 = np.ma.masked_where(model_ps4 < plevel, model_v4)

############################################################################
# plot only for cesm and set up legends
cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5']

cesm_colors = ['red', 'yellow', 'green', 'blue']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed']

varname = 'Wind'

############################################################################
# calculate and plot monthly mean contour
############################################################################

# calculate monthly mean contour
print('Plotting for monthly mean contour...')


# regrid for contour plot
# lonsout, latsout = np.meshgrid(model_lons4, model_lats4)
#
#
# model_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
# model_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
# model_var3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)

# calculate monthly mean

model_mean_u1, model_u_std1 = mon2clim(model_u1[:, :, :], opt=2)
model_mean_u2, model_u_std2 = mon2clim(model_u2[:, :, :], opt=2)
model_mean_u3, model_u_std3 = mon2clim(model_u3[:, :, :], opt=2)
model_mean_u4, model_u_std4 = mon2clim(model_u4[:, :, :], opt=2)

model_mean_v1, model_v_std1 = mon2clim(model_v1[:, :, :], opt=2)
model_mean_v2, model_v_std2 = mon2clim(model_v2[:, :, :], opt=2)
model_mean_v3, model_v_std3 = mon2clim(model_v3[:, :, :], opt=2)
model_mean_v4, model_v_std4 = mon2clim(model_v4[:, :, :], opt=2)

model_mean_ps1, model_ps_std1 = mon2clim(model_v1[:, :, :], opt=2)
model_mean_ps2, model_ps_std2 = mon2clim(model_v2[:, :, :], opt=2)
model_mean_ps3, model_ps_std3 = mon2clim(model_v3[:, :, :], opt=2)
model_mean_ps4, model_ps_std4 = mon2clim(model_v4[:, :, :], opt=2)

vector_length = 4. * np.mean(np.sqrt(np.power(model_mean_u1, 2)+np.power(model_mean_v1, 2)))
plot_list = monnames
plot_list.append('Annual')

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean...')

    # print(model_test1.shape)
    plot_data = [model_u1[idx], model_u2[idx], model_u3[idx], model_u4[idx]]
    plot_u = [model_mean_u1[idx], model_mean_u2[idx], model_mean_u3[idx], model_mean_u4[idx]]
    plot_v = [model_mean_v1[idx], model_mean_v2[idx], model_mean_v3[idx], model_mean_v4[idx]]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-20, 22, 2)
    colormap = cm.PuOr

    title = str(iniyear)+' to '+str(endyear)+' '+imonname+' mean '+var_longname
    if idx != 12:
        fname = 'vrseasia_' + varstr + '_SEA_monthly_mean_contour_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_' + varstr + '_SEA_annual_mean_contour.pdf'

    plot_2Dvectorcontour(plot_data, plot_u, plot_v, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                         latbounds, varname, var_unit, vector_unit, vector_length, title, outdir+fname, opt=0)


print('Plotting for monthly mean contour difference...')


# regrid for contour plot
lonsout, latsout = np.meshgrid(model_lons4, model_lats4)

# model_qu1 = data_regrid(model_qu1, model_lons1, model_lats1, lonsout, latsout)
# model_qv1 = data_regrid(model_qv1, model_lons1, model_lats1, lonsout, latsout)
#
# model_qu2 = data_regrid(model_qu2, model_lons2, model_lats2, lonsout, latsout)
# model_qv2 = data_regrid(model_qv2, model_lons2, model_lats2, lonsout, latsout)

model_u3 = data_regrid(model_u3, model_lons3, model_lats3, lonsout, latsout)
model_v3 = data_regrid(model_v3, model_lons3, model_lats3, lonsout, latsout)

# calculate monthly mean
# model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
# model_mean_qu1, model_qu_std1 = mon2clim(model_qu1[:, :, :], opt=2)
# model_mean_qv1, model_qv_std1 = mon2clim(model_qv1[:, :, :], opt=2)
#
# model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
# model_mean_qu2, model_qu_std2 = mon2clim(model_qu2[:, :, :], opt=2)
# model_mean_qv2, model_qv_std2 = mon2clim(model_qv2[:, :, :], opt=2)

model_mean_u3, model_u_std3 = mon2clim(model_u3[:, :, :], opt=2)
model_mean_v3, model_v_std3 = mon2clim(model_v3[:, :, :], opt=2)

model_mean_u1 = model_mean_u1 - model_mean_u2
model_mean_u3 = model_mean_u3 - model_mean_u4

model_mean_v1 = model_mean_v1 - model_mean_v2
model_mean_v3 = model_mean_v3 - model_mean_v4

vector_length = 4. * np.mean(np.sqrt(np.power(model_mean_u1, 2)+np.power(model_mean_v1, 2)))

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean...')

    # print(model_test1.shape)
    plot_data = [model_mean_u1[idx], model_mean_u3[idx]]
    plot_u = [model_mean_u1[idx], model_mean_u3[idx]]
    plot_v = [model_mean_v1[idx], model_mean_v3[idx]]
    plot_lons = [model_lons2, model_lons4]
    plot_lats = [model_lats2, model_lats4]

    legends = ['CESM-vrseasia vs CESM-ne30', 'CESM-fv0.9x1.25 vs CESM-fv1.9x2.5']

    clevs = np.arange(-3, 3.5, 0.5)
    colormap = cm.PuOr

    title = str(iniyear)+' to '+str(endyear)+' '+imonname+' mean '+var_longname
    if idx != 12:
        fname = 'vrseasia_' + varstr + '_SEA_monthly_mean_contour_diff_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_' + varstr + '_SEA_annual_mean_contour_diff.pdf'

    plot_2Dvectorcontour(plot_data, plot_u, plot_v, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                         latbounds, varname, var_unit, vector_unit, vector_length, title, outdir+fname, opt=0)
