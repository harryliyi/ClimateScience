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
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/mfc/'

# set up variable names and file name
varname = 'MFC'
var_longname = 'vertically integrated moisture flux convergence'
varstr = "mfc"
var_unit = 'mm/day'
vector_unit = r'$m^2/s$'

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
plevel = 850

# define top layer
ptop = 200

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# contants
oro_water = 997
g = 9.8
r_earth = 6371000

############################################################################
# define functions
############################################################################


def dpres_plevel(levs, ps, ptop):
    # calculate layer thickness
    dlevs = np.gradient(levs)
    dlevs[0] = dlevs[0]/2
    dlevs[-1] = dlevs[-1]/2

    # get dimensions
    ntime = ps.shape[0]
    nlev = len(levs)
    nlat = ps.shape[1]
    nlon = ps.shape[2]
    levli = np.abs(levs - ptop).argmin()

    layer_thickness = np.zeros((ntime, nlev, nlat, nlon))
    for ilev in range(levli, nlev, 1):
        temp = np.zeros((ntime, nlat, nlon))
        temp[ps > levs[ilev]] = dlevs[ilev]
        layer_thickness[:, ilev, :, :] = temp

    layer_thickness = layer_thickness * 100  # convert from hPa to Pa

    return layer_thickness


def getdiv1(lats, lons, u, v):
    # calculat divergence
    dlats = lats
    dlons = lons
    # print(dlats)

    dtemp = np.zeros((len(dlats), len(dlons)))
    for ilat in range(len(dlats)):
        dtemp[ilat, :] = v[ilat, :]/r_earth * np.tan(np.deg2rad(dlats[ilat]))
    diverge = np.gradient(u, dlons, axis=1)/np.pi*180/r_earth + np.gradient(v, dlats, axis=0)/np.pi*180/r_earth - dtemp

    return diverge


def getmfc(lats, lons, levs, u, v, q, ps, ptop):
    # calculate moisture flux convergence

    qu = q * u
    qv = q * v

#    print('calculating thickness...')
    layer_thickness = dpres_plevel(levs, ps, ptop)
#    print(ps[5,lattest,lontest])
#    print(levs)
#    print(layer_thickness[5,:,lattest,lontest])
    qu_int = np.sum(qu*layer_thickness, axis=1)/oro_water/g
    qv_int = np.sum(qv*layer_thickness, axis=1)/oro_water/g

#    print(qu[5,:,lattest,lontest])
#    print(qu_int[5,lattest,lontest])
#    print(sum(qu[5,:,lattest,lontest]*layer_thickness[5,:,lattest,lontest]))

    ntime = qu_int.shape[0]
    nlat = qu_int.shape[1]
    nlon = qu_int.shape[2]

    div = np.zeros((ntime, nlat, nlon))
    for itime in range(ntime):
        div[itime, :, :] = getdiv1(lats, lons, qu_int[itime, :, :], qv_int[itime, :, :])
#        print(itime)

    return -div, qu_int, qv_int


def getvint(lats, lons, levs, var, ps, ptop):
    # vertical integration
    layer_thickness = dpres_plevel(levs, ps, ptop)
    var_int = np.sum(var*layer_thickness, axis=1)/oro_water/g

    return var_int

############################################################################
# read data
############################################################################

# read vrcesm


print('Reading VRCESM data...')

# read Q
varname = 'Q'

resolution = 'fv09'
varfname = 'Q'
case = 'vrseasia_AMIP_1979_to_2005'
model_q1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'Q'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_q2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'Q'
case = 'f09_f09_AMIP_1979_to_2005'
model_q3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'Q'
case = 'f19_f19_AMIP_1979_to_2005'
model_q4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

print(model_q1.shape)
print(model_levs1)

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

############################################################################
# plot only for cesm and set up legends
cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5']

cesm_colors = ['red', 'yellow', 'green', 'blue']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed']

varname = 'Moisture Flux Convergence'


############################################################################
# calculate and plot monthly mean contour
############################################################################
print('calculating mfc...')
# calculate mfc
model_var1, model_qu1, model_qv1 = getmfc(model_lats1, model_lons1, model_levs1,
                                          model_u1, model_v1, model_q1, model_ps1, ptop)
model_var2, model_qu2, model_qv2 = getmfc(model_lats2, model_lons2, model_levs2,
                                          model_u2, model_v2, model_q2, model_ps2, ptop)
model_var3, model_qu3, model_qv3 = getmfc(model_lats3, model_lons3, model_levs3,
                                          model_u3, model_v3, model_q3, model_ps3, ptop)
model_var4, model_qu4, model_qv4 = getmfc(model_lats4, model_lons4, model_levs4,
                                          model_u4, model_v4, model_q4, model_ps4, ptop)

model_var1 = model_var1 * 86400 * 1000  # convert to mm/day
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000

print(model_var1.shape)

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

model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean4, model_std4 = mon2clim(model_var4[:, :, :], opt=2)

model_mean_qu1, model_qu_std1 = mon2clim(model_qu1[:, :, :], opt=2)
model_mean_qu2, model_qu_std2 = mon2clim(model_qu2[:, :, :], opt=2)
model_mean_qu3, model_qu_std3 = mon2clim(model_qu3[:, :, :], opt=2)
model_mean_qu4, model_qu_std4 = mon2clim(model_qu4[:, :, :], opt=2)

model_mean_qv1, model_qv_std1 = mon2clim(model_qv1[:, :, :], opt=2)
model_mean_qv2, model_qv_std2 = mon2clim(model_qv2[:, :, :], opt=2)
model_mean_qv3, model_qv_std3 = mon2clim(model_qv3[:, :, :], opt=2)
model_mean_qv4, model_qv_std4 = mon2clim(model_qv4[:, :, :], opt=2)

vector_length = 4. * np.mean(np.sqrt(np.power(model_mean_qu1, 2)+np.power(model_mean_qv1, 2)))
plot_list = monnames
plot_list.append('Annual')

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean...')

    # print(model_test1.shape)
    plot_data = [model_mean1[idx], model_mean2[idx], model_mean3[idx], model_mean4[idx]]
    plot_u = [model_mean_qu1[idx], model_mean_qu2[idx], model_mean_qu3[idx], model_mean_qu4[idx]]
    plot_v = [model_mean_qv1[idx], model_mean_qv2[idx], model_mean_qv3[idx], model_mean_qv4[idx]]
    plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4]
    plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

    clevs = np.arange(-15, 16, 1)
    colormap = cm.BrBG

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

# model_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
# model_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
# model_var1 = data_regrid(model_var1, model_lons1, model_lats1, lonsout, latsout)
# model_qu1 = data_regrid(model_qu1, model_lons1, model_lats1, lonsout, latsout)
# model_qv1 = data_regrid(model_qv1, model_lons1, model_lats1, lonsout, latsout)
#
# model_var2 = data_regrid(model_var2, model_lons2, model_lats2, lonsout, latsout)
# model_qu2 = data_regrid(model_qu2, model_lons2, model_lats2, lonsout, latsout)
# model_qv2 = data_regrid(model_qv2, model_lons2, model_lats2, lonsout, latsout)

model_var3 = data_regrid(model_var3, model_lons3, model_lats3, lonsout, latsout)
model_qu3 = data_regrid(model_qu3, model_lons3, model_lats3, lonsout, latsout)
model_qv3 = data_regrid(model_qv3, model_lons3, model_lats3, lonsout, latsout)

# calculate monthly mean
# model_mean1, model_std1 = mon2clim(model_var1[:, :, :], opt=2)
# model_mean_qu1, model_qu_std1 = mon2clim(model_qu1[:, :, :], opt=2)
# model_mean_qv1, model_qv_std1 = mon2clim(model_qv1[:, :, :], opt=2)
#
# model_mean2, model_std2 = mon2clim(model_var2[:, :, :], opt=2)
# model_mean_qu2, model_qu_std2 = mon2clim(model_qu2[:, :, :], opt=2)
# model_mean_qv2, model_qv_std2 = mon2clim(model_qv2[:, :, :], opt=2)

model_mean3, model_std3 = mon2clim(model_var3[:, :, :], opt=2)
model_mean_qu3, model_qu_std3 = mon2clim(model_qu3[:, :, :], opt=2)
model_mean_qv3, model_qv_std3 = mon2clim(model_qv3[:, :, :], opt=2)

model_mean1 = model_mean1 - model_mean2
model_mean3 = model_mean3 - model_mean4

model_mean_qu1 = model_mean_qu1 - model_mean_qu2
model_mean_qu3 = model_mean_qu3 - model_mean_qu4

model_mean_qv1 = model_mean_qv1 - model_mean_qv2
model_mean_qv3 = model_mean_qv3 - model_mean_qv4

vector_length = 4. * np.mean(np.sqrt(np.power(model_mean_qu1, 2)+np.power(model_mean_qv1, 2)))

for idx, imonname in enumerate(plot_list):
    print('plotting for '+imonname+' mean...')

    # print(model_test1.shape)
    plot_data = [model_mean1[idx], model_mean3[idx]]
    plot_u = [model_mean_qu1[idx], model_mean_qu3[idx]]
    plot_v = [model_mean_qv1[idx], model_mean_qv3[idx]]
    plot_lons = [model_lons2, model_lons4]
    plot_lats = [model_lats2, model_lats4]

    legends = ['CESM-vrseasia vs CESM-ne30', 'CESM-fv0.9x1.25 vs CESM-fv1.9x2.5']

    clevs = np.arange(-10, 10, 1)
    colormap = cm.BrBG

    title = str(iniyear)+' to '+str(endyear)+' '+imonname+' mean '+var_longname
    if idx != 12:
        fname = 'vrseasia_' + varstr + '_SEA_monthly_mean_contour_diff_'+str(idx+1)+'.pdf'
    else:
        fname = 'vrseasia_' + varstr + '_SEA_annual_mean_contour_diff.pdf'

    plot_2Dvectorcontour(plot_data, plot_u, plot_v, plot_lons, plot_lats, colormap, clevs, legends, lonbounds,
                         latbounds, varname, var_unit, vector_unit, vector_length, title, outdir+fname, opt=0)
