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
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/pre/prect/'

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
reg_name = 'mainSEA'

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]

# set data frequency
frequency = 'day'

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
varfname = 'prec'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
model_mask_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
model_mask_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
model_mask_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)
model_mask_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)

model_var1 = model_var1 * 86400 * 1000
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000

model_mask_var1 = model_mask_var1 * 86400 * 1000
model_mask_var2 = model_mask_var2 * 86400 * 1000
model_mask_var3 = model_mask_var3 * 86400 * 1000
model_mask_var4 = model_mask_var4 * 86400 * 1000

# plot the histogram for all grids
fig = plt.figure()
ax = fig.add_subplot(111)

tempdata = model_mask_var4.flatten()
tempdata = tempdata[~np.isnan(tempdata)]
binmax = np.amax(tempdata)*3./4.
binarray = np.arange(0, binmax, binmax/30)

plot_data = [model_var1, model_var2, model_var3, model_var4]
for ii in range(4):
    tempdata = plot_data[ii].flatten()
    tempdata = tempdata[~np.isnan(tempdata)]
    y, binEdges = np.histogram(tempdata, bins=binarray, density=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    # print(bincenters)
    plt.plot(bincenters, y, c=cesm_colors[ii], linestyle=cesm_line_types[ii], linewidth=1.5, label=cesm_legends[ii])

# legends
plt.yscale('log')
plt.legend(handlelength=4, fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel("Frequency")
plt.xlabel("Precip(mm/day)")

title = str(iniyear)+' to '+str(endyear)+' '+var_longname+' distribution over '+reg_name
fname = 'vrseasia_'+varstr+'_'+reg_name+'_hist_allgrids.pdf'

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)

# plot the histogram for over land
fig = plt.figure()
ax = fig.add_subplot(111)

tempdata = model_mask_var4.flatten()
tempdata = tempdata[~np.isnan(tempdata)]
binmax = np.amax(tempdata)*3./4.
binarray = np.arange(0, binmax, binmax/30)

plot_data = [model_mask_var1, model_mask_var2, model_mask_var3, model_mask_var4]
for ii in range(4):
    tempdata = plot_data[ii].flatten()
    tempdata = tempdata[~np.isnan(tempdata)]
    y, binEdges = np.histogram(tempdata, bins=binarray, density=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    # print(bincenters)
    plt.plot(bincenters, y, c=cesm_colors[ii], linestyle=cesm_line_types[ii], linewidth=1.5, label=cesm_legends[ii])

# legends
plt.yscale('log')
plt.legend(handlelength=4, fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel("Frequency")
plt.xlabel("Precip(mm/day)")

title = str(iniyear)+' to '+str(endyear)+' '+var_longname+' distribution over '+reg_name
fname = 'vrseasia_'+varstr+'_'+reg_name+'_hist_overland.pdf'

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)
