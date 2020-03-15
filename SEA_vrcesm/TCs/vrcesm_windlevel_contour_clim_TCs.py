# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_vrcesm import readvrcesm, readvrcesm_3D
from modules.plot.mod_plt_contour import plot_2Dcontour
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.1


plt.switch_backend('agg')


# set up output directory and output log
outdircesm = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/TCs/clim/'

############################################################################
# set parameters
############################################################################
# set up variable names and file name
varname = 'Vorticity'
var_longname = 'Relative vorticity'
varstr = "vorticity"
var_unit = r'$\times 10^{-6}$'


# define inital year and end year
iniyear = 1980
endyear = 2005
yearts = np.arange(iniyear, endyear+1)
nyears = endyear - iniyear + 1

# define the contour plot region
latbounds = [-20, 40]
lonbounds = [80, 140]

vc_latbounds = [10, 22]
vc_lonbounds = [100, 110]

# # mainland Southeast Asia
reg_lats = [10, 30]
reg_lons = [100, 140]
reg_name = 'Western North Pacific'
reg_str = 'WNPacific'

# reg_lats = [10, 20]
# reg_lons = [100, 110]
# reg_name = 'mainSEA'
# reg_str = 'mainSEA'

# set data frequency
frequency = 'mon'

# define top layer
ptop = 200
plevel = 925
plevs = [200, 300, 400, 500, 700, 850, 925]

# contants
oro_water = 997
g = 9.8
r_earth = 6371000


# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monsoon_period = [7, 8, 9]
monsoon_period_str = 'JAS'
outdircesm = outdircesm+monsoon_period_str+'/'


# define the legends
cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5']

# select bins for histogram
nbins = 120

################################################################################################
# S0-Define functions
################################################################################################


def getvc(lats, lons, u, v):
    dlats = lats
    dlons = lons
    # dtemp = np.zeros((u.shape[0], u.shape[1], u.shape[2], u.shape[3]))
    #
    # for ilat in range(len(dlats)):
    #     dtemp[:, :, ilat, :] = v[:, :, ilat, :]/r_earth * np.tan(np.deg2rad(dlats[ilat]))

    vorticity = -1*np.gradient(u, dlats, axis=1)/np.pi*180/r_earth + np.gradient(v, dlons, axis=2)/np.pi*180/r_earth

    return vorticity


def cal_diff(var1, var2, std1, std2, n1, n2):
    res = var1-var2
    SE = np.sqrt((std1**2/n1) + (std2**2/n2))
    res_sig = res/SE
    res_sig = np.abs(res_sig)

    return res, res_sig


############################################################################
# read data
############################################################################


# set data frequency
frequency = 'day'

# read U
varname = 'U'

resolution = 'fv09'
varfname = 'U'
case = 'vrseasia_AMIP_1979_to_2005'
model_u1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'U'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_u2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'U'
case = 'f09_f09_AMIP_1979_to_2005'
model_u3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'U'
case = 'f19_f19_AMIP_1979_to_2005'
model_u4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

# read V
varname = 'V'

resolution = 'fv09'
varfname = 'V'
case = 'vrseasia_AMIP_1979_to_2005'
model_v1, model_time1, model_levs1, model_lats1, model_lons1 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'V'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_v2, model_time2, model_levs2, model_lats2, model_lons2 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv09'
varfname = 'V'
case = 'f09_f09_AMIP_1979_to_2005'
model_v3, model_time3, model_levs3, model_lats3, model_lons3 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

resolution = 'fv19'
varfname = 'V'
case = 'f19_f19_AMIP_1979_to_2005'
model_v4, model_time4, model_levs4, model_lats4, model_lons4 = readvrcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

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


################################################################################################
# calculate Hovmöller diagram of vorticity at 700 hPa based on Takahashi_etal_2006
model_lev = np.argmin(np.abs(model_levs1 - plevel))

model_u_lev1 = model_u1[:, model_lev, :, :]
model_u_lev2 = model_u2[:, model_lev, :, :]
model_u_lev3 = model_u3[:, model_lev, :, :]
model_u_lev4 = model_u4[:, model_lev, :, :]

model_v_lev1 = model_v1[:, model_lev, :, :]
model_v_lev2 = model_v2[:, model_lev, :, :]
model_v_lev3 = model_v3[:, model_lev, :, :]
model_v_lev4 = model_v4[:, model_lev, :, :]

time_index = np.in1d(model_time1.month, monsoon_period)

select_time = model_time1[time_index]


model_u_mon1 = model_u_lev1[time_index, :, :]
model_v_mon1 = model_v_lev1[time_index, :, :]

model_u_mon2 = model_u_lev2[time_index, :, :]
model_v_mon2 = model_v_lev2[time_index, :, :]

model_u_mon3 = model_u_lev3[time_index, :, :]
model_v_mon3 = model_v_lev3[time_index, :, :]

model_u_mon4 = model_u_lev4[time_index, :, :]
model_v_mon4 = model_v_lev4[time_index, :, :]

vc1 = getvc(model_lats1, model_lons1, model_u_mon1, model_v_mon1)
vc2 = getvc(model_lats2, model_lons2, model_u_mon2, model_v_mon2)
vc3 = getvc(model_lats3, model_lons3, model_u_mon3, model_v_mon3)
vc4 = getvc(model_lats4, model_lons4, model_u_mon4, model_v_mon4)

vc1 = vc1[:, model_latl1:model_latu1+1, model_lonl1:model_lonr1+1] * 1000000
vc2 = vc2[:, model_latl2:model_latu2+1, model_lonl2:model_lonr2+1] * 1000000
vc3 = vc3[:, model_latl3:model_latu3+1, model_lonl3:model_lonr3+1] * 1000000
vc4 = vc4[:, model_latl4:model_latu4+1, model_lonl4:model_lonr4+1] * 1000000

title = ' CESM distribution of vorticity in '+monsoon_period_str+' over '+reg_name
fname = 'vrcesm_'+str(plevel)+'hPa_vorticity_'+reg_str+'_pdf_'+monsoon_period_str+'_allmodels'

plt.clf()
fig = plt.figure()

plot_data = [vc1, vc2, vc3, vc4]
cesm_colors = ['red', 'yellow', 'green', 'blue']

ax = fig.add_subplot(111)
for idx in range(4):
    thres = np.percentile(plot_data[idx], 99.9)
    ranges = np.arange(0., thres, thres/nbins)
    # ranges = np.append(ranges, plot_data1[idx].max())
    temp = plot_data[idx]
    hist, bin_edges = np.histogram(temp.flatten(), bins=ranges, density=True)
    bins = ranges[:-1]+(ranges[1]-ranges[0])/2
    ax.plot(bins, hist, c=cesm_colors[idx], linestyle='solid', linewidth=1.5, label=cesm_legends[idx], alpha=0.8)

ax.legend(handlelength=4, fontsize=5)
xticks = np.arange(0, 101, 20)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, fontsize=5)
ax.set_xlim(xticks[0], xticks[-1])
ax.set_xlabel(var_longname+' ['+var_unit+']', fontsize=6)
yticks = np.arange(0, .1, 0.02)
yticknames = [str(np.round(ii, 2)) for ii in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticknames, fontsize=5)
ax.set_ylabel('Frequency', fontsize=6)
ax.set_ylim(0, yticks[-1])
# plt.yscale('log')

plt.savefig(outdircesm+fname+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdircesm+fname+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
