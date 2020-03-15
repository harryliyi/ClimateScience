# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_cesm import readcesm, readcesm_3D

import matplotlib.cm as cm
import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = "/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/"


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/climatology/mf_atBC/'

# set up variable names and file name
varname = 'MFC'
varfname = "mfc"
varstr = "Moisture Flux Convergence"
var_res = "fv19"
var_unit = 'mm/day'

# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-40, 60]
lonbounds = [40, 180]

# latbounds = [ -40 , 40 ]
# lonbounds = [ 10 , 160 ]

# define top layer
ptop = 200

# contants
oro_water = 997
g = 9.8
r_earth = 6371000

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

# set data frequency
frequency = 'mon'

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

################################################################################################
# S0-Define functions
################################################################################################


# calculate layer thickness
def dpres_plevel(levs, ps, ptop):
    # calculate thickness
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

# calculat divergence


def getdiv1(lats, lons, u, v):
    dlats = lats
    dlons = lons
#    print(dlats)

    dtemp = np.zeros((len(dlats), len(dlons)))
    for ilat in range(len(dlats)):
        dtemp[ilat, :] = v[ilat, :]/r_earth * np.tan(np.deg2rad(dlats[ilat]))
    diverge = np.gradient(u, dlons, axis=1)/np.pi*180/r_earth + np.gradient(v, dlats, axis=0)/np.pi*180/r_earth - dtemp
    # diverge = np.gradient(u, dlons, axis=1)/np.pi*180/r_earth + np.gradient(v, dlats, axis=0)/np.pi*180/r_earth

    return diverge


# def getdiv2(lats, lons, u, v):
#
#     dlats = r_earth * np.deg2rad(np.gradient(lats))
#     dlons = r_earth * np.deg2rad(np.gradient(lons))
#
#     div = np.zeros((len(dlats), len(dlons)))
#     for ilat in range(len(dlats)):
#         for ilon in range(len(dlons)):
#             ilatu = ilat + 1
#             ilatl = ilat - 1
#             ilonu = ilon + 1
#             ilonl = ilon - 1
#             if (ilat == 0):
#                 ilatl = 0
#             if (ilat == len(dlats)-1):
#                 ilatu = len(dlats) - 1
#             if (ilon == 0):
#                 ilonl = 0
#             if (ilon == len(dlons)-1):
#                 ilonu = len(dlons) - 1
#
#             div[ilat, ilon] = (v[ilatu, ilon]-v[ilatl, ilon])/dlats[ilat] + (u[ilat, ilonu]-u[ilat, ilonl])/dlons[ilon] - v[ilat, ilon]/r_earth * np.tan(np.deg2rad(lats))
#
# #    print(np.sum(abs(div-diverge)))
# #    print(np.sum(abs(div)))
#     return div

# calculate moisture flux convergence


def getmfc(lats, lons, levs, u, v, q, ps, ptop):

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

    return -div


# plot for climatology
def plotclim(lons, lats, var, title, fname):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines()
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-15., 15.1, 1.)
    cs = map.contourf(x, y, var, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")

    # add colorbar.
    cbar = map.colorbar(cs, location='bottom', pad="5%")
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+' ['+var_unit+']')

    # add title
    plt.savefig(fname+'.png', bbox_inches='tight', dpi=600)
    plt.suptitle(title, fontsize=7, y=0.95)

    # save figure
    plt.savefig(fname+'.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)


################################################################################################
# S1-read climatological data
################################################################################################
# read vrcesm
print('Reading CESM data...')

# read PS
varname = 'PS'

resolution = 'fv19'
varfname = 'PS'
model_ps, model_time,  model_lats, model_lons = readcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)


model_ps = model_ps/100

varname = 'U'
varfname = 'U'
model_u, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

varname = 'V'
varfname = 'V'
model_v, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

varname = 'Q'
varfname = 'Q'
model_q, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

print('finished reading...')

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))

print('calculating mfc...')
# calculate mfc
model_mfc = getmfc(model_lats, model_lons, model_levs, model_u, model_v, model_q, model_ps, ptop)


#################################################################################################
# plot climatology
#################################################################################################
print('plotting climatology...')

for idx in range(12):
    select_time = (model_time.month == months[idx])
    time_temp = model_time[select_time]
    print(time_temp)

    model_mfc_select = model_mfc[select_time, :, :] * 86400 * 1000
    print(np.mean(model_mfc_select[:, model_latl:model_latu+1, model_lonl:model_lonr+1]))
    title = ' icam monthly averaged vertical integrated moisture flux convergence in '+monnames[idx]
    fname = 'icam5_integrated_mfc_SEA_monthly_mean_contour_'+str(idx+1)
    plotclim(model_lons, model_lats, np.mean(model_mfc_select, axis=0), title, outdir+fname)
