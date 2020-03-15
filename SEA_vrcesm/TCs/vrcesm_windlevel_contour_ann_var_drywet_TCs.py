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
outdircesm = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/TCs/drywet/'

############################################################################
# set parameters
############################################################################
# set up variable names and file name
varname = 'PRECT'
var_longname = 'Precipitation'
varstr = "prect"
var_unit = 'mm/day'


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

# mainland Southeast Asia
reg_lats = [10, 20]
reg_lons = [100, 110]
reg_name = 'mainSEA'

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

# read prect

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0)

# convert from m/s to mm/day
model_var1 = model_var1 * 86400 * 1000
model_var2 = model_var2 * 86400 * 1000
model_var3 = model_var3 * 86400 * 1000
model_var4 = model_var4 * 86400 * 1000
print(model_time1)
print(model_var1.shape)

model_var_seas1 = np.zeros((nyears, len(model_lats1), len(model_lons1)))
model_var_seas2 = np.zeros((nyears, len(model_lats2), len(model_lons2)))
model_var_seas3 = np.zeros((nyears, len(model_lats3), len(model_lons3)))
model_var_seas4 = np.zeros((nyears, len(model_lats4), len(model_lons4)))

for iyear in range(nyears):
    model_var_seas1[iyear, :, :] = np.mean(
        model_var1[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)
    model_var_seas2[iyear, :, :] = np.mean(
        model_var2[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)
    model_var_seas3[iyear, :, :] = np.mean(
        model_var3[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)
    model_var_seas4[iyear, :, :] = np.mean(
        model_var4[12*iyear+monsoon_period[0]-1: 12*iyear+monsoon_period[-1], :, :], axis=0)


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


# calculate regional mean time series
model_var_ann_ts1 = np.sum(
    np.sum(model_var_seas1[:, model_latl1: model_latu1+1, model_lonl1: model_lonr1+1], axis=1), axis=1)
model_var_ann_ts2 = np.sum(
    np.sum(model_var_seas2[:, model_latl2: model_latu2+1, model_lonl2: model_lonr2+1], axis=1), axis=1)
model_var_ann_ts3 = np.sum(
    np.sum(model_var_seas3[:, model_latl3: model_latu3+1, model_lonl3: model_lonr3+1], axis=1), axis=1)
model_var_ann_ts4 = np.sum(
    np.sum(model_var_seas4[:, model_latl4: model_latu4+1, model_lonl4: model_lonr4+1], axis=1), axis=1)


model_var_ann_ts1 = model_var_ann_ts1/((model_latu1-model_latl1+1)*(model_lonr1-model_lonl1+1))
model_var_ann_ts2 = model_var_ann_ts2/((model_latu2-model_latl2+1)*(model_lonr2-model_lonl2+1))
model_var_ann_ts3 = model_var_ann_ts3/((model_latu3-model_latl3+1)*(model_lonr3-model_lonl3+1))
model_var_ann_ts4 = model_var_ann_ts4/((model_latu4-model_latl4+1)*(model_lonr4-model_lonl4+1))

print(model_var_seas1[0, :, :])
print(model_var_ann_ts1)
print(model_var_ann_ts1.shape)

############################################################################
# calculate the dry-wet years for selected period
############################################################################

# calculate std
monsoon_std1 = np.std(model_var_ann_ts1)
monsoon_std2 = np.std(model_var_ann_ts2)
monsoon_std3 = np.std(model_var_ann_ts3)
monsoon_std4 = np.std(model_var_ann_ts4)

# calculate mean
monsoon_mean1 = np.mean(model_var_ann_ts1)
monsoon_mean2 = np.mean(model_var_ann_ts2)
monsoon_mean3 = np.mean(model_var_ann_ts3)
monsoon_mean4 = np.mean(model_var_ann_ts4)

# define dty-wet years
years_dry1 = []
years_dry2 = []
years_dry3 = []
years_dry4 = []

years_wet1 = []
years_wet2 = []
years_wet3 = []
years_wet4 = []

for iyear in range(nyears):
    if (model_var_ann_ts1[iyear] >= (monsoon_mean1 + monsoon_std1)):
        years_wet1.append(iniyear+iyear)
    if (model_var_ann_ts2[iyear] >= (monsoon_mean2 + monsoon_std2)):
        years_wet2.append(iniyear+iyear)
    if (model_var_ann_ts3[iyear] >= (monsoon_mean3 + monsoon_std3)):
        years_wet3.append(iniyear+iyear)
    if (model_var_ann_ts4[iyear] >= (monsoon_mean4 + monsoon_std4)):
        years_wet4.append(iniyear+iyear)

    if (model_var_ann_ts1[iyear] <= (monsoon_mean1 - monsoon_std1)):
        years_dry1.append(iniyear+iyear)
    if (model_var_ann_ts2[iyear] <= (monsoon_mean2 - monsoon_std2)):
        years_dry2.append(iniyear+iyear)
    if (model_var_ann_ts3[iyear] <= (monsoon_mean3 - monsoon_std3)):
        years_dry3.append(iniyear+iyear)
    if (model_var_ann_ts4[iyear] <= (monsoon_mean4 - monsoon_std4)):
        years_dry4.append(iniyear+iyear)


print('Wet years are:')
print(cesm_legends[0]+':')
print(years_wet1)
print(cesm_legends[1]+':')
print(years_wet2)
print(cesm_legends[2]+':')
print(years_wet3)
print(cesm_legends[3]+':')
print(years_wet4)

print('Dry years are:')
print(cesm_legends[0]+':')
print(years_dry1)
print(cesm_legends[1]+':')
print(years_dry2)
print(cesm_legends[2]+':')
print(years_dry3)
print(cesm_legends[3]+':')
print(years_dry4)

############################################################################
# read wind data
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

################################################################################################
# calculate Hovmöller diagram of vorticity at 700 hPa based on Takahashi_etal_2006
model_lev = np.argmin(np.abs(model_levs1 - 700))

model_u_lev1 = model_u1[:, model_lev, :, :]
model_u_lev2 = model_u2[:, model_lev, :, :]
model_u_lev3 = model_u3[:, model_lev, :, :]
model_u_lev4 = model_u4[:, model_lev, :, :]

model_v_lev1 = model_v1[:, model_lev, :, :]
model_v_lev2 = model_v2[:, model_lev, :, :]
model_v_lev3 = model_v3[:, model_lev, :, :]
model_v_lev4 = model_v4[:, model_lev, :, :]

wet_index1 = np.in1d(model_time1.year, years_wet1) & np.in1d(model_time1.month, monsoon_period)
dry_index1 = np.in1d(model_time1.year, years_dry1) & np.in1d(model_time1.month, monsoon_period)
wet_time1 = model_time1[wet_index1]
dry_time1 = model_time1[dry_index1]

wet_index2 = np.in1d(model_time2.year, years_wet2) & np.in1d(model_time2.month, monsoon_period)
dry_index2 = np.in1d(model_time2.year, years_dry2) & np.in1d(model_time2.month, monsoon_period)
wet_time2 = model_time2[wet_index2]
dry_time2 = model_time2[dry_index2]

wet_index3 = np.in1d(model_time3.year, years_wet3) & np.in1d(model_time3.month, monsoon_period)
dry_index3 = np.in1d(model_time3.year, years_dry3) & np.in1d(model_time3.month, monsoon_period)
wet_time3 = model_time3[wet_index3]
dry_time3 = model_time3[dry_index3]

wet_index4 = np.in1d(model_time4.year, years_wet4) & np.in1d(model_time4.month, monsoon_period)
dry_index4 = np.in1d(model_time4.year, years_dry4) & np.in1d(model_time4.month, monsoon_period)
wet_time4 = model_time4[wet_index4]
dry_time4 = model_time4[dry_index4]

print(model_time1[wet_index1])

model_u_wet1 = model_u_lev1[wet_index1, :, :]
model_u_dry1 = model_u_lev1[dry_index1, :, :]
model_v_wet1 = model_v_lev1[wet_index1, :, :]
model_v_dry1 = model_v_lev1[dry_index1, :, :]

model_u_wet2 = model_u_lev2[wet_index2, :, :]
model_u_dry2 = model_u_lev2[dry_index2, :, :]
model_v_wet2 = model_v_lev2[wet_index2, :, :]
model_v_dry2 = model_v_lev2[dry_index2, :, :]

model_u_wet3 = model_u_lev3[wet_index3, :, :]
model_u_dry3 = model_u_lev3[dry_index3, :, :]
model_v_wet3 = model_v_lev3[wet_index3, :, :]
model_v_dry3 = model_v_lev3[dry_index3, :, :]

model_u_wet4 = model_u_lev4[wet_index4, :, :]
model_u_dry4 = model_u_lev4[dry_index4, :, :]
model_v_wet4 = model_v_lev4[wet_index4, :, :]
model_v_dry4 = model_v_lev4[dry_index4, :, :]

vc_wet1 = getvc(model_lats1, model_lons1, model_u_wet1, model_v_wet1)
vc_dry1 = getvc(model_lats1, model_lons1, model_u_dry1, model_v_dry1)

vc_wet2 = getvc(model_lats2, model_lons2, model_u_wet2, model_v_wet2)
vc_dry2 = getvc(model_lats2, model_lons2, model_u_dry2, model_v_dry2)

vc_wet3 = getvc(model_lats3, model_lons3, model_u_wet3, model_v_wet3)
vc_dry3 = getvc(model_lats3, model_lons3, model_u_dry3, model_v_dry3)

vc_wet4 = getvc(model_lats4, model_lons4, model_u_wet4, model_v_wet4)
vc_dry4 = getvc(model_lats4, model_lons4, model_u_dry4, model_v_dry4)

model_latl1 = np.argmin(np.abs(model_lats1 - vc_latbounds[0]))
model_latu1 = np.argmin(np.abs(model_lats1 - vc_latbounds[1]))

model_latl2 = np.argmin(np.abs(model_lats2 - vc_latbounds[0]))
model_latu2 = np.argmin(np.abs(model_lats2 - vc_latbounds[1]))

model_latl3 = np.argmin(np.abs(model_lats3 - vc_latbounds[0]))
model_latu3 = np.argmin(np.abs(model_lats3 - vc_latbounds[1]))

model_latl4 = np.argmin(np.abs(model_lats4 - vc_latbounds[0]))
model_latu4 = np.argmin(np.abs(model_lats4 - vc_latbounds[1]))

vc_wet1 = np.mean(vc_wet1[:, model_latl1:model_latu1+1, :], axis=1)
vc_dry1 = np.mean(vc_dry1[:, model_latl1:model_latu1+1, :], axis=1)

vc_wet2 = np.mean(vc_wet2[:, model_latl2:model_latu2+1, :], axis=1)
vc_dry2 = np.mean(vc_dry2[:, model_latl2:model_latu2+1, :], axis=1)

vc_wet3 = np.mean(vc_wet3[:, model_latl3:model_latu3+1, :], axis=1)
vc_dry3 = np.mean(vc_dry3[:, model_latl3:model_latu3+1, :], axis=1)

vc_wet4 = np.mean(vc_wet4[:, model_latl4:model_latu4+1, :], axis=1)
vc_dry4 = np.mean(vc_dry4[:, model_latl4:model_latu4+1, :], axis=1)

ndays = int(len(wet_time1)/len(years_wet1))

vc_wet_mean1 = np.zeros((ndays, len(model_lons1)))
vc_wet_mean2 = np.zeros((ndays, len(model_lons2)))
vc_wet_mean3 = np.zeros((ndays, len(model_lons3)))
vc_wet_mean4 = np.zeros((ndays, len(model_lons4)))

vc_dry_mean1 = np.zeros((ndays, len(model_lons1)))
vc_dry_mean2 = np.zeros((ndays, len(model_lons2)))
vc_dry_mean3 = np.zeros((ndays, len(model_lons3)))
vc_dry_mean4 = np.zeros((ndays, len(model_lons4)))

vc_wet_std1 = np.zeros((ndays, len(model_lons1)))
vc_wet_std2 = np.zeros((ndays, len(model_lons2)))
vc_wet_std3 = np.zeros((ndays, len(model_lons3)))
vc_wet_std4 = np.zeros((ndays, len(model_lons4)))

vc_dry_std1 = np.zeros((ndays, len(model_lons1)))
vc_dry_std2 = np.zeros((ndays, len(model_lons2)))
vc_dry_std3 = np.zeros((ndays, len(model_lons3)))
vc_dry_std4 = np.zeros((ndays, len(model_lons4)))


for idx in range(ndays):
    vc_wet_mean1[idx, :] = np.mean(vc_wet1[idx::ndays, :], axis=0)
    vc_wet_mean2[idx, :] = np.mean(vc_wet2[idx::ndays, :], axis=0)
    vc_wet_mean3[idx, :] = np.mean(vc_wet3[idx::ndays, :], axis=0)
    vc_wet_mean4[idx, :] = np.mean(vc_wet4[idx::ndays, :], axis=0)

    vc_wet_std1[idx, :] = np.std(vc_wet1[idx::ndays, :], axis=0)
    vc_wet_std2[idx, :] = np.std(vc_wet2[idx::ndays, :], axis=0)
    vc_wet_std3[idx, :] = np.std(vc_wet3[idx::ndays, :], axis=0)
    vc_wet_std4[idx, :] = np.std(vc_wet4[idx::ndays, :], axis=0)

    vc_dry_mean1[idx, :] = np.mean(vc_dry1[idx::ndays, :], axis=0)
    vc_dry_mean2[idx, :] = np.mean(vc_dry2[idx::ndays, :], axis=0)
    vc_dry_mean3[idx, :] = np.mean(vc_dry3[idx::ndays, :], axis=0)
    vc_dry_mean4[idx, :] = np.mean(vc_dry4[idx::ndays, :], axis=0)

    vc_dry_std1[idx, :] = np.std(vc_dry1[idx::ndays, :], axis=0)
    vc_dry_std2[idx, :] = np.std(vc_dry2[idx::ndays, :], axis=0)
    vc_dry_std3[idx, :] = np.std(vc_dry3[idx::ndays, :], axis=0)
    vc_dry_std4[idx, :] = np.std(vc_dry4[idx::ndays, :], axis=0)

res_diff1, res_sig1 = cal_diff(vc_wet_mean1, vc_dry_mean1, vc_wet_std1, vc_dry_std1, len(years_wet1), len(years_dry1))
res_diff2, res_sig2 = cal_diff(vc_wet_mean2, vc_dry_mean2, vc_wet_std2, vc_dry_std2, len(years_wet2), len(years_dry2))
res_diff3, res_sig3 = cal_diff(vc_wet_mean3, vc_dry_mean3, vc_wet_std3, vc_dry_std3, len(years_wet3), len(years_dry3))
res_diff4, res_sig4 = cal_diff(vc_wet_mean4, vc_dry_mean4, vc_wet_std4, vc_dry_std4, len(years_wet4), len(years_dry4))

vc_wet_mean1 = vc_wet_mean1 * 1000000
vc_wet_mean2 = vc_wet_mean2 * 1000000
vc_wet_mean3 = vc_wet_mean3 * 1000000
vc_wet_mean4 = vc_wet_mean4 * 1000000

vc_dry_mean1 = vc_dry_mean1 * 1000000
vc_dry_mean2 = vc_dry_mean2 * 1000000
vc_dry_mean3 = vc_dry_mean3 * 1000000
vc_dry_mean4 = vc_dry_mean4 * 1000000

res_diff1 = res_diff1 * 1000000
res_diff2 = res_diff2 * 1000000
res_diff3 = res_diff3 * 1000000
res_diff4 = res_diff4 * 1000000

############################################################################
# calculate the seasonal mean contour for dey and wet years
############################################################################

years_wet = [years_wet1, years_wet2, years_wet3, years_wet4]
years_dry = [years_dry1, years_dry2, years_dry3, years_dry4]

model_lats = [model_lats1, model_lats2, model_lats3, model_lats4]
model_lons = [model_lons1, model_lons2, model_lons3, model_lons4]

vc_wet_mean = [vc_wet_mean1, vc_wet_mean2, vc_wet_mean3, vc_wet_mean4]
vc_dry_mean = [vc_dry_mean1, vc_dry_mean2, vc_dry_mean3, vc_dry_mean4]
res_diff = [res_diff1, res_diff2, res_diff3, res_diff4]
res_sig = [res_sig1, res_sig2, res_sig3, res_sig4]

legends = ['Wet composites', 'Dry composites', 'Wet-Dry']

for idx in range(4):
    days = range(ndays)
    model_sig = np.zeros((ndays, len(model_lons[idx])))
    model_sig[:, :] = 0.5

    plot_data = [vc_wet_mean[idx], vc_dry_mean[idx], res_diff[idx]]
    plot_days = [days, days, days]
    plot_lons = [model_lons[idx], model_lons[idx], model_lons[idx]]
    plot_test = [model_sig, model_sig, res_sig[idx]]

    print(len(plot_data))

    title = ' CESM differences in zonal averaged vorticity between Wet and Dry years in '+cesm_legends[idx]
    fname = 'vrcesm_vorticity_SEA_daily_contour_diff_drywet_'+monsoon_period_str+'_'+str(idx+1)

    # plot for wet-dry contours with sig
    plt.clf()
    fig, axes = plt.subplots(3, 1)
    axes = axes.flatten()

    for ss in range(3):
        axes[ss].set_title(legends[ss], fontsize=7, pad=-0.3)

        x, y = np.meshgrid(model_lons[idx], days)
        if ss < 2:
            clevs = np.arange(0, 15, 3)
            cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.YlGn, alpha=0.9, extend="both")
            axes[ss].axvline(x=105, linewidth=1.5, color='black')
            axes[ss].set_ylabel('Days', fontsize=7, labelpad=0.7)
            axes[ss].set_xticks([])
            axes[ss].yaxis.set_tick_params(labelsize=5)
            cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
            ticks = clevs
            ticks = np.round(ticks, 2)
            ticklabels = [str(itick) for itick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
        else:
            clevs = np.arange(-12, 13, 3)
            cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
            axes[ss].axvline(x=105, linewidth=1.5, color='black')
            axes[ss].set_xlabel('Longitude', fontsize=7, labelpad=0.7)
            axes[ss].set_ylabel('Days', fontsize=7, labelpad=0.7)
            axes[ss].xaxis.set_tick_params(labelsize=5)
            axes[ss].yaxis.set_tick_params(labelsize=5)
            temptest = plot_test[ss]
            levels = [0., 2.01, temptest.max()]
            axes[ss].contourf(x, y, plot_test[ss], levels=levels, colors='none', hatches=['', '//////'], alpha=0)
            cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
            ticks = clevs
            ticks = np.round(ticks, 2)
            ticklabels = [str(itick) for itick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=3)
        cbar.ax.set_title(r'$\times 10^{-6}$', fontsize=4)

    plt.savefig(outdircesm+fname+'_wtsig.png', bbox_inches='tight', dpi=600)
    plt.suptitle(title, fontsize=7, y=0.95)
    plt.savefig(outdircesm+fname+'_wtsig.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)

    # plot for wet and dry contours
    plt.clf()
    fig, axes = plt.subplots(2, 1)
    axes = axes.flatten()

    for ss in range(2):
        axes[ss].set_title(legends[ss], fontsize=7, pad=-0.3)

        x, y = np.meshgrid(model_lons[idx], days)
        if ss < 1:
            clevs = np.arange(0, 18, 3)
            cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.YlGn, alpha=0.9, extend="both")
            axes[ss].axvline(x=105, linewidth=1.5, color='black')
            axes[ss].set_ylabel('Days', fontsize=7, labelpad=0.7)
            axes[ss].set_xticks([])
            axes[ss].yaxis.set_tick_params(labelsize=5)
            cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
            ticks = clevs
            ticks = np.round(ticks, 2)
            ticklabels = [str(itick) for itick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
        else:
            clevs = np.arange(0, 18, 3)
            cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.YlGn, alpha=0.9, extend="both")
            axes[ss].axvline(x=105, linewidth=1.5, color='black')
            axes[ss].set_xlabel('Longitude', fontsize=7, labelpad=0.7)
            axes[ss].set_ylabel('Days', fontsize=7, labelpad=0.7)
            axes[ss].xaxis.set_tick_params(labelsize=5)
            axes[ss].yaxis.set_tick_params(labelsize=5)
            cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
            ticks = clevs
            ticks = np.round(ticks, 2)
            ticklabels = [str(itick) for itick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=3)
        cbar.ax.set_title(r'$\times 10^{-6}$', fontsize=4)

    plt.savefig(outdircesm+fname+'_nodiff.png', bbox_inches='tight', dpi=600)
    plt.suptitle(title, fontsize=7, y=0.95)
    plt.savefig(outdircesm+fname+'_nodiff.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)

title = ' CESM differences in zonal averaged vorticity between Wet and Dry years'
fname = 'vrcesm_vorticity_SEA_daily_contour_diff_drywet_'+monsoon_period_str+'_allmodels'


# # plot all difference together
# # plot for wet-dry contours with sig
# plt.clf()
# fig = plt.figure()
#
# for idx in range(4):
#     ax = fig.add_subplot(2, 2, idx+1)
#
#     ax.set_title(cesm_legends[idx], fontsize=5, pad=-0.3)
#     map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
#                   llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l', ax=ax)
#
#     map.drawcoastlines(linewidth=0.3)
#     map.drawcountries()
#     parallels = np.arange(latbounds[0], latbounds[1], 20)
#     meridians = np.arange(lonbounds[0], lonbounds[1], 20)
#     map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
#     map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
#
#     x, y = np.meshgrid(plot_lons[idx], plot_lats[idx])
#
#     clevs = np.arange(-6, 6.1, 0.5)
#     cs = map.contourf(x, y, plot_data[idx], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
#     temptest = plot_test[ss]
#     levels = [0., 2.01, temptest.max()]
#     csm = map.contourf(x, y, plot_test[idx], levels=levels, colors='none', hatches=['', '//////'], alpha=0)
#
# fig.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
# cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
# cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
# ticks = clevs[::4]
# ticks = np.round(ticks, 2)
# ticklabels = [str(itick) for itick in ticks]
# cbar.set_ticks(ticks)
# cbar.set_ticklabels(ticklabels)
# cbar.ax.tick_params(labelsize=4)
# cbar.set_label('mm/day', fontsize=5, labelpad=0.5)
#
# plt.savefig(outdircesm+fname+'.png', bbox_inches='tight', dpi=600)
# plt.suptitle(title, fontsize=5, y=0.95)
# plt.savefig(outdircesm+fname+'.pdf', bbox_inches='tight', dpi=600)
# plt.close(fig)
