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
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/pre/'

############################################################################
# set parameters
############################################################################
# variable info
var_longname = 'Convective Precip'
varstr = 'precc'
var_unit = 'mm/day'


# time bounds
iniyear = 1980
endyear = 2005

# define regions
latbounds = [10, 25]
lonbounds = [100, 110]
reg_name = 'mainSEA'

# # mainland Southeast Asia
# reg_lats = [0, 20]
# reg_lons = [60, 130]

# set data frequency
frequency = 'day'

# set prect threshold
premin = 17
premax = 30

# 17mm is 95th percentile for all resolution, 39-30mm is the 99th percentile

# set percentile
percentile = 99

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# plot legend
cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5']

cesm_colors = ['red', 'yellow', 'green', 'blue']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed']


############################################################################
# read data
############################################################################

# read vrcesm

print('Reading VRCESM data...')

resolution = 'fv09'
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

print('scatter plot for precc vs precl..')
# scatter plot for precc vs precl
plt.clf()
fig = plt.figure()

precc_var1 = precc_var1.flatten() * 86400 * 1000
precl_var1 = precl_var1.flatten() * 86400 * 1000
prect_var1 = prect_var1.flatten() * 86400 * 1000

# print(np.mean((prect_var1 - precc_var1 - precl_var1)**2))
print(np.percentile(prect_var1, percentile))

precc_v1 = precc_var1[((prect_var1 > premin) & (prect_var1 < premax))]
precl_v1 = precl_var1[((prect_var1 > premin) & (prect_var1 < premax))]
prect_v1 = prect_var1[((prect_var1 > premin) & (prect_var1 < premax))]

precc_var2 = precc_var2.flatten() * 86400 * 1000
precl_var2 = precl_var2.flatten() * 86400 * 1000
prect_var2 = prect_var2.flatten() * 86400 * 1000

print(np.percentile(prect_var2, percentile))

precc_v2 = precc_var2[((prect_var2 > premin) & (prect_var2 < premax))]
precl_v2 = precl_var2[((prect_var2 > premin) & (prect_var2 < premax))]
prect_v2 = prect_var2[((prect_var2 > premin) & (prect_var2 < premax))]

precc_var3 = precc_var3.flatten() * 86400 * 1000
precl_var3 = precl_var3.flatten() * 86400 * 1000
prect_var3 = prect_var3.flatten() * 86400 * 1000

print(np.percentile(prect_var3, percentile))

precc_v3 = precc_var3[((prect_var3 > premin) & (prect_var3 < premax))]
precl_v3 = precl_var3[((prect_var3 > premin) & (prect_var3 < premax))]
prect_v3 = prect_var3[((prect_var3 > premin) & (prect_var3 < premax))]

precc_var4 = precc_var4.flatten() * 86400 * 1000
precl_var4 = precl_var4.flatten() * 86400 * 1000
prect_var4 = prect_var4.flatten() * 86400 * 1000

print(np.percentile(prect_var4, percentile))

precc_v4 = precc_var4[((prect_var4 > premin) & (prect_var4 < premax))]
precl_v4 = precl_var4[((prect_var4 > premin) & (prect_var4 < premax))]
prect_v4 = prect_var4[((prect_var4 > premin) & (prect_var4 < premax))]

print(precc_v1.shape)
plot_datax = [precc_v1, precc_v2, precc_v3, precc_v4]
plot_datay = [precl_v1, precl_v2, precl_v3, precl_v4]

for idx in range(len(plot_datax)):
    ax = fig.add_subplot(2, 2, idx+1)
    ax.set_title(cesm_legends[idx], fontsize=5, pad=2)

    plt.plot(plot_datax[idx], plot_datay[idx], 'o', color=cesm_colors[idx], markersize=2,)
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.set_xlim(0, max(np.amax(plot_datax[idx]), np.amax(plot_datay[idx])))
    ax.set_ylim(0, max(np.amax(plot_datax[idx]), np.amax(plot_datay[idx])))
    ax.set_xlabel('Convective Precip', fontsize=5, labelpad=0.7)
    ax.set_ylabel('Large-scale Precip', fontsize=5, labelpad=0.7)

title = str(iniyear)+' to '+str(endyear)+' R'+str(premin)+'-'+str(premax)+' PRECC vs PRECL over '+reg_name
fname = 'vrseasia_precc_vs_precl_R'+str(premin)+'-'+str(premax)+'_'+reg_name+'_allgrids.png'

plt.suptitle(title, fontsize=7, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight', dpi=600)
plt.close(fig)

print('scatter plot for precl vs prect..')
# scatter plot for precl vs prect
plt.clf()
fig = plt.figure()

plot_datax = [prect_v1, prect_v2, prect_v3, prect_v4]
plot_datay = [precl_v1, precl_v2, precl_v3, precl_v4]

for idx in range(len(plot_datax)):
    ax = fig.add_subplot(2, 2, idx+1)
    ax.set_title(cesm_legends[idx], fontsize=5, pad=2)

    plt.plot(plot_datax[idx], plot_datay[idx], 'o', color=cesm_colors[idx], markersize=2)
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.set_xlim(0, max(np.amax(plot_datax[idx]), np.amax(plot_datay[idx])))
    ax.set_ylim(0, max(np.amax(plot_datax[idx]), np.amax(plot_datay[idx])))
    ax.set_xlabel('Total Precip', fontsize=5, labelpad=0.7)
    ax.set_ylabel('Large-scale Precip', fontsize=5, labelpad=0.7)

title = str(iniyear)+' to '+str(endyear)+' R'+str(premin)+'-'+str(premax)+' PRECT vs PRECL over '+reg_name
fname = 'vrseasia_prect_vs_precl_R'+str(premin)+'-'+str(premax)+'_'+reg_name+'_allgrids.png'

plt.suptitle(title, fontsize=7, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight', dpi=600)
plt.close(fig)

print('scatter plot for precc vs prect..')
# scatter plot for precc vs prect
plt.clf()
fig = plt.figure()

plot_datax = [prect_v1, prect_v2, prect_v3, prect_v4]
plot_datay = [precc_v1, precc_v2, precc_v3, precc_v4]

for idx in range(len(plot_datax)):
    ax = fig.add_subplot(2, 2, idx+1)
    ax.set_title(cesm_legends[idx], fontsize=5, pad=2)

    plt.plot(plot_datax[idx], plot_datay[idx], 'o', color=cesm_colors[idx], markersize=2)
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.set_xlim(0, max(np.amax(plot_datax[idx]), np.amax(plot_datay[idx])))
    ax.set_ylim(0, max(np.amax(plot_datax[idx]), np.amax(plot_datay[idx])))
    ax.set_xlabel('Total Precip', fontsize=5, labelpad=0.7)
    ax.set_ylabel('Convective Precip', fontsize=5, labelpad=0.7)

title = str(iniyear)+' to '+str(endyear)+' R'+str(premin)+'-'+str(premax)+' PRECT vs PRECC over '+reg_name
fname = 'vrseasia_prect_vs_precc_R'+str(premin)+'-'+str(premax)+'_'+reg_name+'_allgrids.png'

plt.suptitle(title, fontsize=7, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight', dpi=600)
plt.close(fig)
