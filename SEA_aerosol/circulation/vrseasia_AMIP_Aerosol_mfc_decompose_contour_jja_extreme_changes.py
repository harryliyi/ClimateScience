# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import math as math
import pandas as pd
import datetime
import matplotlib.cm as cm
import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# set up data directories and filenames
case1 = "vrseasia_19501959_OBS"
case2 = "vrseasia_20002010_OBS"
case3 = "vrseasia_20002009_OBS_SUBAERSST_CESM1CAM5_SST"
case4 = "vrseasia_20002009_OBS_AEREMIS1950"
case5 = "vrseasia_20002009_OBS_AEREMIS1950_SUBAERSST_CESM1CAM5_SST"

pre_expdir1 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case1+"/atm/"
pre_expdir2 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case2+"/atm/"
pre_expdir3 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case3+"/atm/"
pre_expdir4 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case4+"/atm/"
pre_expdir5 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case5+"/atm/"

expdir1 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/daily/"+case1+"/"
expdir2 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/daily/"+case2+"/"
expdir3 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/daily/"+case3+"/"
expdir4 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/daily/"+case4+"/"
expdir5 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/daily/"+case5+"/"

# set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/mfc/extreme/"

# set up variable names and file name
varname = 'MFC'
varfname = "mfc"
varstr = "Moisture Flux Convergence"
var_res = "fv09"
var_unit = 'mm/day'

# define inital year and end year
iniyear = 2
endyear = 15

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# define regions
reg_names = ['mainland SEA', 'Central Indian']
reg_str = ['mainSEA', 'ctInd']
reg_lats = [[10, 20], [16.5, 26.5]]
reg_lons = [[100, 110], [74.5, 86.5]]

# define top layer
ptop = 200
p0 = 100000

# contants
oro_water = 997
g = 9.8
r_earth = 6371000

# set up moderate thresholds
mod_thres = [[5, 30], [5, 60]]


# month series
month = np.arange(1, 13, 1)
mname = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# define the season period
inimonth = 6
endmonth = 8

iniday = np.sum(mdays[0:inimonth-1])
endday = np.sum(mdays[0:endmonth])
# print(iniday)
# print(endday)
ndays = endday-iniday

# set up percentile
# percentile_ranges = [0, 10, 50, 90, 95, 97, 99]
# percentile_ranges = [99]
percentile_ranges = [90, 95, 99]
percentile_tops = [93, 97, 99.5]
percentile_bots = [85, 93, 97]

# percentile_ranges = [86, 96, 99]
# percentile_tops = [92.80, 98.07, 99.48]
# percentile_bots = [73.17, 92.80, 98.07]


# terms
terms = [r'$\Delta P^e$', r'$-\Delta (\partial q/\partial t)^e$', r'$-\Delta (q^e {\bigtriangledown\cdot u}^e)$',
         r'$-\Delta (u^e\cdot{\bigtriangledown q}^e)$', r'$-\Delta {\bigtriangledown\cdot(qu)}^e$']

################################################################################################
# S0-Define functions
################################################################################################
# calculate difference of mean and significance level


def getstats_diff(var1mean, var2mean, var1std, var2std, n1, n2):

    vardiff = var1mean - var2mean
    varttest = vardiff/np.sqrt(var1std**2/n1+var2std**2/n2)

    return vardiff, abs(varttest)

# calculate hypothesis test of mean


def getstats_mean(var):
    n = var.shape[0]
    varmean = np.mean(var, axis=0)
    varstd = np.std(var, axis=0)

    varttest = varmean/(varstd/n)

    return varmean, abs(varttest)


# calculate layer thickness for each model level


def dpres_plevel(levs, ps, ptop):

    # get dimensions
    ntime = ps.shape[0]
    nlev = len(levs)
    nlat = ps.shape[1]
    nlon = ps.shape[2]
    levli = np.abs(levs - ptop).argmin()

    layer_thickness = np.zeros((ntime, nlev, nlat, nlon))
    layer_mid = np.zeros((ntime, nlev, nlat, nlon))
    for ilev in range(0, nlev, 1):
        temp = p0*hyam[ilev]+ps*hybm[ilev]
        layer_mid[:, ilev, :, :] = temp
        temp = (p0*hyai[ilev+1]+ps*hybi[ilev+1]) - (p0*hyai[ilev]+ps*hybi[ilev])
        layer_thickness[:, ilev, :, :] = temp

    layer_thickness[layer_mid < ptop*100] = 0.

    # print(layer_thickness[0, :, lattest, lontest])
    # print(layer_mid[0, :, lattest, lontest])
    # print(ps[0, lattest, lontest])

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

    return diverge

# calculat divergence for all layers


def getdiv3(lats, lons, u, v):
    dlats = lats
    dlons = lons
#    print(dlats)

    dtemp = np.zeros((u.shape[0], u.shape[1], len(dlats), len(dlons)))
    for ilat in range(len(dlats)):
        dtemp[:, :, ilat, :] = v[:, :, ilat, :]/r_earth * np.tan(np.deg2rad(dlats[ilat]))
    diverge = np.gradient(u, dlons, axis=3)/np.pi*180/r_earth + np.gradient(v, dlats, axis=2)/np.pi*180/r_earth - dtemp

    return diverge

# back up test for divergence calculation


def getdiv2(lats, lons, u, v):

    dlats = r_earth * np.deg2rad(np.gradient(lats))
    dlons = r_earth * np.deg2rad(np.gradient(lons))

    div = np.zeros((len(dlats), len(dlons)))
    for ilat in range(len(dlats)):
        for ilon in range(len(dlons)):
            ilatu = ilat + 1
            ilatl = ilat - 1
            ilonu = ilon + 1
            ilonl = ilon - 1
            if (ilat == 0):
                ilatl = 0
            if (ilat == len(dlats)-1):
                ilatu = len(dlats) - 1
            if (ilon == 0):
                ilonl = 0
            if (ilon == len(dlons)-1):
                ilonu = len(dlons) - 1

            div[ilat, ilon] = (v[ilatu, ilon]-v[ilatl, ilon])/dlats[ilat] + (u[ilat, ilonu]-u[ilat, ilonl]
                                                                             )/dlons[ilon] - v[ilat, ilon]/r_earth * np.tan(np.deg2rad(lats[ilat+latli]))

#    print(np.sum(abs(div-diverge)))
#    print(np.sum(abs(div)))
    return div

# calculate convergence for each time step and each layer


def getconv(lats, lons, u, v):

    ntime = u.shape[0]
    nlev = u.shape[1]
    nlat = u.shape[2]
    nlon = u.shape[3]

    div = getdiv3(lats, lons, u[:, :, :, :], v[:, :, :, :])

    return -div


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

# calculate moisture flux convergence by getting dry-air mass convergence first


def getmfc2(lats, lons, levs, u, v, q, ps, ptop):

    qu = q * u
    qv = q * v

    conv = getconv(lats, lons, qu, qv)

#    print('calculating thickness...')
    layer_thickness = dpres_plevel(levs, ps, ptop)

    conv_int = np.sum(conv*layer_thickness, axis=1)/oro_water/g

    return conv_int


def getmassconv(lats, lons, levs, u, v, q, ps, ptop):

    conv = getconv(lats, lons, u, v)
    conv = conv * q

#    print('calculating thickness...')
    layer_thickness = dpres_plevel(levs, ps, ptop)

    conv_int = np.sum(conv*layer_thickness, axis=1)/oro_water/g

    return conv_int


def getmoistconv(lats, lons, levs, u, v, q, ps, ptop):

    dlats = lats
    dlons = lons

    conv = u * np.gradient(q, dlons, axis=3)/np.pi*180/r_earth + v * np.gradient(q, dlats, axis=2)/np.pi*180/r_earth

#    print('calculating thickness...')
    layer_thickness = dpres_plevel(levs, ps, ptop)

    conv_int = np.sum(conv*layer_thickness, axis=1)/oro_water/g

    return -conv_int


# vertical integration
def getvint(lats, lons, levs, var, ps, ptop):
    layer_thickness = dpres_plevel(levs, ps, ptop)
    var_int = np.sum(var*layer_thickness, axis=1)/oro_water/g

    return var_int


# plot for differences
def plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, label, forcingstr, fname, opt):
    fig = plt.figure()
    # Res1
    ax1 = fig.add_subplot(221)
    ax1.set_title('a) '+label[0], fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-18, 18.1, 3)
    cs = map.contourf(x, y, res1, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt1.max()]
        csm = plt.contourf(x, y, tt1, levels=levels, colors='none', hatches=["", "....."], alpha=0)
#    print(tt_total)

    # Res2
    ax2 = fig.add_subplot(222)
    ax2.set_title('b) '+label[1], fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    cs = map.contourf(x, y, res2, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt2.max()]
        csm = plt.contourf(x, y, tt2, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    # Res3
    ax3 = fig.add_subplot(223)
    ax3.set_title('c) '+label[2], fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    cs = map.contourf(x, y, res3, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt3.max()]
        csm = plt.contourf(x, y, tt3, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    # Res4
    ax3 = fig.add_subplot(224)
    ax3.set_title('d) '+label[3], fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    cs = map.contourf(x, y, res4, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt4.max()]
        csm = plt.contourf(x, y, tt4, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    fig.subplots_adjust(bottom=0.2, wspace=0.15, hspace=0.1)
    cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
#    cbar = fig.colorbar(cs,orientation='horizontal',fraction=0.15, aspect= 25,shrink = 0.8)
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('['+var_unit+']', fontsize=6, labelpad=-0.3)

    # add title
    if (opt == 1):
        plt.savefig(fname+"_with_siglev.png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fname+".png", dpi=600, bbox_inches='tight')
    plt.suptitle(forcingstr+" "+varstr+" changes", fontsize=7, y=1.08)
#    plt.tight_layout()
    if (opt == 1):
        plt.savefig(fname+"_with_siglev.pdf", bbox_inches='tight')
    else:
        plt.savefig(fname+".pdf", bbox_inches='tight')

    plt.close(fig)


################################################################################################
# S1-read daily data
################################################################################################
# read lats,lons,levs,
fname1 = var_res+"_Q_"+case1+".cam.h1.0001-01-01-00000.nc"
fdata1 = Dataset(expdir1+fname1)
lats = fdata1.variables['lat'][:]
lons = fdata1.variables['lon'][:]
levs = fdata1.variables['lev'][:]
hyam = fdata1.variables['hyam'][:]
hybm = fdata1.variables['hybm'][:]
hyai = fdata1.variables['hyai'][:]
hybi = fdata1.variables['hybi'][:]

# latitude/longitude  lower and upper contour index
latli = np.abs(lats - latbounds[0]).argmin()
latui = np.abs(lats - latbounds[1]).argmin()

lonli = np.abs(lons - lonbounds[0]).argmin()
lonui = np.abs(lons - lonbounds[1]).argmin()

lats = lats[latli:latui+1]
lons = lons[lonli:lonui+1]

lattest = np.abs(lats - 25).argmin()
lontest = np.abs(lons - 98).argmin()

nlats = latui - latli + 1
nlons = lonui - lonli + 1
nlevs = len(levs)
ntime = (endyear-iniyear+1)*ndays

print(ntime)
print(nlats)
print(nlons)
print(levs)


# read land mask
dir_lndfrc = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
flndfrc = 'USGS-gtopo30_0.9x1.25_remap_c051027.nc'
dataset_lndfrc = Dataset(dir_lndfrc+flndfrc)
lndfrc = dataset_lndfrc.variables['LANDFRAC'][latli:latui+1, lonli:lonui+1]

print('reading the data...')
starttime = datetime.datetime.now()
print(starttime)

# read precip
print('Reading precip...')
pre1 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
pre2 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
pre3 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
pre4 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
pre5 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))

for iyear in np.arange(iniyear, endyear+1, 1):
    if (iyear < 10):
        yearno = '000'+str(iyear)
    else:
        yearno = '00'+str(iyear)
    print('Current year is: '+yearno)

    fname1 = var_res+'_prect_'+case1+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname2 = var_res+'_prect_'+case2+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname3 = var_res+'_prect_'+case3+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname4 = var_res+'_prect_'+case4+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname5 = var_res+'_prect_'+case5+'.cam.h1.'+yearno+'-01-01-00000.nc'

    fdata1 = Dataset(pre_expdir1+fname1)
    fdata2 = Dataset(pre_expdir2+fname2)
    fdata3 = Dataset(pre_expdir3+fname3)
    fdata4 = Dataset(pre_expdir4+fname4)
    fdata5 = Dataset(pre_expdir5+fname5)

    temp1 = fdata1.variables['PRECT'][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000
    temp2 = fdata2.variables['PRECT'][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000
    temp3 = fdata3.variables['PRECT'][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000
    temp4 = fdata4.variables['PRECT'][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000
    temp5 = fdata5.variables['PRECT'][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000

    pre1[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp1.copy()
    pre2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp2.copy()
    pre3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp3.copy()
    pre4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp4.copy()
    pre5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp5.copy()


# read ps
print('reading ps...')
ps1 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
ps2 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
ps3 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
ps4 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
ps5 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))

for iyear in np.arange(iniyear, endyear+1, 1):
    if (iyear < 10):
        yearno = '000'+str(iyear)
    else:
        yearno = '00'+str(iyear)
    print('Current year is: '+yearno)

    fname1 = var_res+'_ps_'+case1+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname2 = var_res+'_ps_'+case2+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname3 = var_res+'_ps_'+case3+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname4 = var_res+'_ps_'+case4+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname5 = var_res+'_ps_'+case5+'.cam.h1.'+yearno+'-01-01-00000.nc'

    fdata1 = Dataset(expdir1+fname1)
    fdata2 = Dataset(expdir2+fname2)
    fdata3 = Dataset(expdir3+fname3)
    fdata4 = Dataset(expdir4+fname4)
    fdata5 = Dataset(expdir5+fname5)

    temp1 = fdata1.variables['PS'][iniday: endday, latli:latui+1, lonli:lonui+1]
    temp2 = fdata2.variables['PS'][iniday: endday, latli:latui+1, lonli:lonui+1]
    temp3 = fdata3.variables['PS'][iniday: endday, latli:latui+1, lonli:lonui+1]
    temp4 = fdata4.variables['PS'][iniday: endday, latli:latui+1, lonli:lonui+1]
    temp5 = fdata5.variables['PS'][iniday: endday, latli:latui+1, lonli:lonui+1]

    ps1[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp1.copy()
    ps2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp2.copy()
    ps3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp3.copy()
    ps4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp4.copy()
    ps5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp5.copy()

# read humidity
print('reading Q...')
q1 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
q2 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
q3 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
q4 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
q5 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))

dq1 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
dq2 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
dq3 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
dq4 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
dq5 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))

for iyear in np.arange(iniyear, endyear+1, 1):
    if (iyear < 10):
        yearno = '000'+str(iyear)
    else:
        yearno = '00'+str(iyear)
    print('Current year is: '+yearno)

    fname1 = var_res+'_Q_'+case1+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname2 = var_res+'_Q_'+case2+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname3 = var_res+'_Q_'+case3+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname4 = var_res+'_Q_'+case4+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname5 = var_res+'_Q_'+case5+'.cam.h1.'+yearno+'-01-01-00000.nc'

    fdata1 = Dataset(expdir1+fname1)
    fdata2 = Dataset(expdir2+fname2)
    fdata3 = Dataset(expdir3+fname3)
    fdata4 = Dataset(expdir4+fname4)
    fdata5 = Dataset(expdir5+fname5)

    temp1 = fdata1.variables['Q'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp2 = fdata2.variables['Q'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp3 = fdata3.variables['Q'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp4 = fdata4.variables['Q'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp5 = fdata5.variables['Q'][iniday: endday, :, latli:latui+1, lonli:lonui+1]

    q1[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp1.copy()
    q2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp2.copy()
    q3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp3.copy()
    q4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp4.copy()
    q5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp5.copy()

    dtemp1 = fdata1.variables['Q'][iniday+1: endday+1, :, latli:latui+1, lonli:lonui+1]
    dtemp2 = fdata2.variables['Q'][iniday+1: endday+1, :, latli:latui+1, lonli:lonui+1]
    dtemp3 = fdata3.variables['Q'][iniday+1: endday+1, :, latli:latui+1, lonli:lonui+1]
    dtemp4 = fdata4.variables['Q'][iniday+1: endday+1, :, latli:latui+1, lonli:lonui+1]
    dtemp5 = fdata5.variables['Q'][iniday+1: endday+1, :, latli:latui+1, lonli:lonui+1]

    dq1[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = (dtemp1.copy() - temp1.copy())/24/60/60
    dq2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = (dtemp2.copy() - temp2.copy())/24/60/60
    dq3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = (dtemp3.copy() - temp3.copy())/24/60/60
    dq4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = (dtemp4.copy() - temp4.copy())/24/60/60
    dq5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = (dtemp5.copy() - temp5.copy())/24/60/60

# read U
print('reading U...')
u1 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
u2 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
u3 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
u4 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
u5 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))

for iyear in np.arange(iniyear, endyear+1, 1):
    if (iyear < 10):
        yearno = '000'+str(iyear)
    else:
        yearno = '00'+str(iyear)
    print('Current year is: '+yearno)

    fname1 = var_res+'_uwind_'+case1+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname2 = var_res+'_uwind_'+case2+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname3 = var_res+'_uwind_'+case3+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname4 = var_res+'_uwind_'+case4+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname5 = var_res+'_uwind_'+case5+'.cam.h1.'+yearno+'-01-01-00000.nc'

    fdata1 = Dataset(expdir1+fname1)
    fdata2 = Dataset(expdir2+fname2)
    fdata3 = Dataset(expdir3+fname3)
    fdata4 = Dataset(expdir4+fname4)
    fdata5 = Dataset(expdir5+fname5)

    temp1 = fdata1.variables['U'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp2 = fdata2.variables['U'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp3 = fdata3.variables['U'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp4 = fdata4.variables['U'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp5 = fdata5.variables['U'][iniday: endday, :, latli:latui+1, lonli:lonui+1]

    u1[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp1.copy()
    u2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp2.copy()
    u3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp3.copy()
    u4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp4.copy()
    u5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp5.copy()

# read V
print('reading V...')
v1 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
v2 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
v3 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
v4 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))
v5 = np.zeros(((endyear-iniyear+1)*ndays, nlevs, nlats, nlons))

for iyear in np.arange(iniyear, endyear+1, 1):
    if (iyear < 10):
        yearno = '000'+str(iyear)
    else:
        yearno = '00'+str(iyear)
    print('Current year is: '+yearno)

    fname1 = var_res+'_vwind_'+case1+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname2 = var_res+'_vwind_'+case2+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname3 = var_res+'_vwind_'+case3+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname4 = var_res+'_vwind_'+case4+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname5 = var_res+'_vwind_'+case5+'.cam.h1.'+yearno+'-01-01-00000.nc'

    fdata1 = Dataset(expdir1+fname1)
    fdata2 = Dataset(expdir2+fname2)
    fdata3 = Dataset(expdir3+fname3)
    fdata4 = Dataset(expdir4+fname4)
    fdata5 = Dataset(expdir5+fname5)

    temp1 = fdata1.variables['V'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp2 = fdata2.variables['V'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp3 = fdata3.variables['V'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp4 = fdata4.variables['V'][iniday: endday, :, latli:latui+1, lonli:lonui+1]
    temp5 = fdata5.variables['V'][iniday: endday, :, latli:latui+1, lonli:lonui+1]

    v1[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp1.copy()
    v2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp2.copy()
    v3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp3.copy()
    v4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp4.copy()
    v5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :, :] = temp5.copy()

endtime = datetime.datetime.now()
print(endtime)
print('finished reading...')
print('Time consumed: ')
print(endtime-starttime)

#################################################################################################
# calculate mfc and decomposition
#################################################################################################
# calculate mfc (div(qu))
print('Calculating mfc...')
starttime = datetime.datetime.now()

mfc1 = getmfc2(lats, lons, levs, u1, v1, q1, ps1, ptop) * 86400 * 1000
mfc2 = getmfc2(lats, lons, levs, u2, v2, q2, ps2, ptop) * 86400 * 1000
mfc3 = getmfc2(lats, lons, levs, u3, v3, q3, ps3, ptop) * 86400 * 1000
mfc4 = getmfc2(lats, lons, levs, u4, v4, q4, ps4, ptop) * 86400 * 1000
mfc5 = getmfc2(lats, lons, levs, u5, v5, q5, ps5, ptop) * 86400 * 1000

endtime = datetime.datetime.now()
print('Time consumed: ')
print(endtime-starttime)

# calculate product of moisture and mass convergence (q div(u))
print('Calculating product of moisture and mass convergence...')
starttime = datetime.datetime.now()

msc1 = getmassconv(lats, lons, levs, u1, v1, q1, ps1, ptop) * 86400 * 1000
msc2 = getmassconv(lats, lons, levs, u2, v2, q2, ps2, ptop) * 86400 * 1000
msc3 = getmassconv(lats, lons, levs, u3, v3, q3, ps3, ptop) * 86400 * 1000
msc4 = getmassconv(lats, lons, levs, u4, v4, q4, ps4, ptop) * 86400 * 1000
msc5 = getmassconv(lats, lons, levs, u5, v5, q5, ps5, ptop) * 86400 * 1000

endtime = datetime.datetime.now()
print('Time consumed: ')
print(endtime-starttime)

# calculate advection of moisture (u div(q))
print('Calculating advection of moisture...')
starttime = datetime.datetime.now()

mtc1 = getmoistconv(lats, lons, levs, u1, v1, q1, ps1, ptop) * 86400 * 1000
mtc2 = getmoistconv(lats, lons, levs, u2, v2, q2, ps2, ptop) * 86400 * 1000
mtc3 = getmoistconv(lats, lons, levs, u3, v3, q3, ps3, ptop) * 86400 * 1000
mtc4 = getmoistconv(lats, lons, levs, u4, v4, q4, ps4, ptop) * 86400 * 1000
mtc5 = getmoistconv(lats, lons, levs, u5, v5, q5, ps5, ptop) * 86400 * 1000

endtime = datetime.datetime.now()
print('Time consumed: ')
print(endtime-starttime)

# calculate vertically integrated moisture-storage (-dq/dt) and convet from m/s to mm/day
print('Calculating vertically integrated -dq/dt...')

dq_int1 = getvint(lats, lons, levs, dq1, ps1, ptop) * -1. * 86400 * 1000
dq_int2 = getvint(lats, lons, levs, dq2, ps2, ptop) * -1. * 86400 * 1000
dq_int3 = getvint(lats, lons, levs, dq3, ps3, ptop) * -1. * 86400 * 1000
dq_int4 = getvint(lats, lons, levs, dq4, ps4, ptop) * -1. * 86400 * 1000
dq_int5 = getvint(lats, lons, levs, dq5, ps5, ptop) * -1. * 86400 * 1000


#################################################################################################
# plot climatology
#################################################################################################

# select precip extreme precentile
for idx_percent, percentile in enumerate(percentile_ranges):

    outdir_percent = outdir+str(percentile)+'th/'
    percentile_top = percentile_tops[idx_percent]
    percentile_bot = percentile_bots[idx_percent]

    print('Current percentile is '+str(percentile)+'th...')

    pre_percent_top1 = np.percentile(pre1, percentile_top, axis=0)
    pre_percent_top2 = np.percentile(pre2, percentile_top, axis=0)
    pre_percent_top3 = np.percentile(pre3, percentile_top, axis=0)
    pre_percent_top4 = np.percentile(pre4, percentile_top, axis=0)
    pre_percent_top5 = np.percentile(pre5, percentile_top, axis=0)

    pre_percent_bot1 = np.percentile(pre1, percentile_bot, axis=0)
    pre_percent_bot2 = np.percentile(pre2, percentile_bot, axis=0)
    pre_percent_bot3 = np.percentile(pre3, percentile_bot, axis=0)
    pre_percent_bot4 = np.percentile(pre4, percentile_bot, axis=0)
    pre_percent_bot5 = np.percentile(pre5, percentile_bot, axis=0)

    pre_ext1 = np.zeros((nlats, nlons))
    pre_ext2 = np.zeros((nlats, nlons))
    pre_ext3 = np.zeros((nlats, nlons))
    pre_ext4 = np.zeros((nlats, nlons))
    pre_ext5 = np.zeros((nlats, nlons))

    pre_std1 = np.zeros((nlats, nlons))
    pre_std2 = np.zeros((nlats, nlons))
    pre_std3 = np.zeros((nlats, nlons))
    pre_std4 = np.zeros((nlats, nlons))
    pre_std5 = np.zeros((nlats, nlons))

    mfc_ext1 = np.zeros((nlats, nlons))
    mfc_ext2 = np.zeros((nlats, nlons))
    mfc_ext3 = np.zeros((nlats, nlons))
    mfc_ext4 = np.zeros((nlats, nlons))
    mfc_ext5 = np.zeros((nlats, nlons))

    mfc_std1 = np.zeros((nlats, nlons))
    mfc_std2 = np.zeros((nlats, nlons))
    mfc_std3 = np.zeros((nlats, nlons))
    mfc_std4 = np.zeros((nlats, nlons))
    mfc_std5 = np.zeros((nlats, nlons))

    msc_ext1 = np.zeros((nlats, nlons))
    msc_ext2 = np.zeros((nlats, nlons))
    msc_ext3 = np.zeros((nlats, nlons))
    msc_ext4 = np.zeros((nlats, nlons))
    msc_ext5 = np.zeros((nlats, nlons))

    msc_std1 = np.zeros((nlats, nlons))
    msc_std2 = np.zeros((nlats, nlons))
    msc_std3 = np.zeros((nlats, nlons))
    msc_std4 = np.zeros((nlats, nlons))
    msc_std5 = np.zeros((nlats, nlons))

    mtc_ext1 = np.zeros((nlats, nlons))
    mtc_ext2 = np.zeros((nlats, nlons))
    mtc_ext3 = np.zeros((nlats, nlons))
    mtc_ext4 = np.zeros((nlats, nlons))
    mtc_ext5 = np.zeros((nlats, nlons))

    mtc_std1 = np.zeros((nlats, nlons))
    mtc_std2 = np.zeros((nlats, nlons))
    mtc_std3 = np.zeros((nlats, nlons))
    mtc_std4 = np.zeros((nlats, nlons))
    mtc_std5 = np.zeros((nlats, nlons))

    dq_ext1 = np.zeros((nlats, nlons))
    dq_ext2 = np.zeros((nlats, nlons))
    dq_ext3 = np.zeros((nlats, nlons))
    dq_ext4 = np.zeros((nlats, nlons))
    dq_ext5 = np.zeros((nlats, nlons))

    dq_std1 = np.zeros((nlats, nlons))
    dq_std2 = np.zeros((nlats, nlons))
    dq_std3 = np.zeros((nlats, nlons))
    dq_std4 = np.zeros((nlats, nlons))
    dq_std5 = np.zeros((nlats, nlons))

    for ilat in range(nlats):
        for ilon in range(nlons):
            pre_bool1 = (pre1[:, ilat, ilon] >= pre_percent_bot1[ilat, ilon]) & (
                pre1[:, ilat, ilon] < pre_percent_top1[ilat, ilon])
            pre_bool2 = (pre2[:, ilat, ilon] >= pre_percent_bot2[ilat, ilon]) & (
                pre2[:, ilat, ilon] < pre_percent_top2[ilat, ilon])
            pre_bool3 = (pre3[:, ilat, ilon] >= pre_percent_bot3[ilat, ilon]) & (
                pre3[:, ilat, ilon] < pre_percent_top3[ilat, ilon])
            pre_bool4 = (pre4[:, ilat, ilon] >= pre_percent_bot4[ilat, ilon]) & (
                pre4[:, ilat, ilon] < pre_percent_top4[ilat, ilon])
            pre_bool5 = (pre5[:, ilat, ilon] >= pre_percent_bot5[ilat, ilon]) & (
                pre5[:, ilat, ilon] < pre_percent_top5[ilat, ilon])

            df1 = np.sum(pre_bool1, axis=0)
            df2 = np.sum(pre_bool2, axis=0)
            df3 = np.sum(pre_bool3, axis=0)
            df4 = np.sum(pre_bool4, axis=0)
            df5 = np.sum(pre_bool5, axis=0)

            pre_ext1[ilat, ilon] = np.mean(pre1[pre_bool1, ilat, ilon])
            pre_ext2[ilat, ilon] = np.mean(pre2[pre_bool2, ilat, ilon])
            pre_ext3[ilat, ilon] = np.mean(pre3[pre_bool3, ilat, ilon])
            pre_ext4[ilat, ilon] = np.mean(pre4[pre_bool4, ilat, ilon])
            pre_ext5[ilat, ilon] = np.mean(pre5[pre_bool5, ilat, ilon])

            pre_std1[ilat, ilon] = np.std(pre1[pre_bool1, ilat, ilon])
            pre_std2[ilat, ilon] = np.std(pre2[pre_bool2, ilat, ilon])
            pre_std3[ilat, ilon] = np.std(pre3[pre_bool3, ilat, ilon])
            pre_std4[ilat, ilon] = np.std(pre4[pre_bool4, ilat, ilon])
            pre_std5[ilat, ilon] = np.std(pre5[pre_bool5, ilat, ilon])

            mfc_ext1[ilat, ilon] = np.mean(mfc1[pre_bool1, ilat, ilon])
            mfc_ext2[ilat, ilon] = np.mean(mfc2[pre_bool2, ilat, ilon])
            mfc_ext3[ilat, ilon] = np.mean(mfc3[pre_bool3, ilat, ilon])
            mfc_ext4[ilat, ilon] = np.mean(mfc4[pre_bool4, ilat, ilon])
            mfc_ext5[ilat, ilon] = np.mean(mfc5[pre_bool5, ilat, ilon])

            mfc_std1[ilat, ilon] = np.std(mfc1[pre_bool1, ilat, ilon])
            mfc_std2[ilat, ilon] = np.std(mfc2[pre_bool2, ilat, ilon])
            mfc_std3[ilat, ilon] = np.std(mfc3[pre_bool3, ilat, ilon])
            mfc_std4[ilat, ilon] = np.std(mfc4[pre_bool4, ilat, ilon])
            mfc_std5[ilat, ilon] = np.std(mfc5[pre_bool5, ilat, ilon])

            msc_ext1[ilat, ilon] = np.mean(msc1[pre_bool1, ilat, ilon])
            msc_ext2[ilat, ilon] = np.mean(msc2[pre_bool2, ilat, ilon])
            msc_ext3[ilat, ilon] = np.mean(msc3[pre_bool3, ilat, ilon])
            msc_ext4[ilat, ilon] = np.mean(msc4[pre_bool4, ilat, ilon])
            msc_ext5[ilat, ilon] = np.mean(msc5[pre_bool5, ilat, ilon])

            msc_std1[ilat, ilon] = np.std(msc1[pre_bool1, ilat, ilon])
            msc_std2[ilat, ilon] = np.std(msc2[pre_bool2, ilat, ilon])
            msc_std3[ilat, ilon] = np.std(msc3[pre_bool3, ilat, ilon])
            msc_std4[ilat, ilon] = np.std(msc4[pre_bool4, ilat, ilon])
            msc_std5[ilat, ilon] = np.std(msc5[pre_bool5, ilat, ilon])

            mtc_ext1[ilat, ilon] = np.mean(mtc1[pre_bool1, ilat, ilon])
            mtc_ext2[ilat, ilon] = np.mean(mtc2[pre_bool2, ilat, ilon])
            mtc_ext3[ilat, ilon] = np.mean(mtc3[pre_bool3, ilat, ilon])
            mtc_ext4[ilat, ilon] = np.mean(mtc4[pre_bool4, ilat, ilon])
            mtc_ext5[ilat, ilon] = np.mean(mtc5[pre_bool5, ilat, ilon])

            mtc_std1[ilat, ilon] = np.std(mtc1[pre_bool1, ilat, ilon])
            mtc_std2[ilat, ilon] = np.std(mtc2[pre_bool2, ilat, ilon])
            mtc_std3[ilat, ilon] = np.std(mtc3[pre_bool3, ilat, ilon])
            mtc_std4[ilat, ilon] = np.std(mtc4[pre_bool4, ilat, ilon])
            mtc_std5[ilat, ilon] = np.std(mtc5[pre_bool5, ilat, ilon])

            dq_ext1[ilat, ilon] = np.mean(dq_int1[pre_bool1, ilat, ilon])
            dq_ext2[ilat, ilon] = np.mean(dq_int2[pre_bool2, ilat, ilon])
            dq_ext3[ilat, ilon] = np.mean(dq_int3[pre_bool3, ilat, ilon])
            dq_ext4[ilat, ilon] = np.mean(dq_int4[pre_bool4, ilat, ilon])
            dq_ext5[ilat, ilon] = np.mean(dq_int5[pre_bool5, ilat, ilon])

            dq_std1[ilat, ilon] = np.std(dq_int1[pre_bool1, ilat, ilon])
            dq_std2[ilat, ilon] = np.std(dq_int2[pre_bool2, ilat, ilon])
            dq_std3[ilat, ilon] = np.std(dq_int3[pre_bool3, ilat, ilon])
            dq_std4[ilat, ilon] = np.std(dq_int4[pre_bool4, ilat, ilon])
            dq_std5[ilat, ilon] = np.std(dq_int5[pre_bool5, ilat, ilon])

    print('Plotting Moisture budget changes for extreme events...')
    labels = [r'$\Delta P^e$', r'$-\Delta (\partial q/\partial t)^e$',
              r'$-\Delta (q^e \{\bigtriangledown\cdot u\}^e)$', r'$-\Delta (u^e\cdot\{\bigtriangledown q\}^e)$']

    # all aerosol foring
    forcingstr = "All aerosol forcings"
    forcingfname = "allaerosols"
    res1, tt1 = getstats_diff(pre_ext2, pre_ext5, pre_std2, pre_std5, df2, df5)
    res2, tt2 = getstats_diff(dq_ext2, dq_ext5, dq_std2, dq_std5, df2, df5)
    res3, tt3 = getstats_diff(msc_ext2, msc_ext5, msc_std2, msc_std5, df2, df5)
    res4, tt4 = getstats_diff(mtc_ext2, mtc_ext5, mtc_std2, mtc_std5, df2, df5)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_moisture_budget_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # aerosol fast response1
    forcingstr = "Aerosol fast response"
    forcingfname = "fastaerosol1"
    res1, tt1 = getstats_diff(pre_ext2, pre_ext4, pre_std2, pre_std4, df2, df4)
    res2, tt2 = getstats_diff(dq_ext2, dq_ext4, dq_std2, dq_std4, df2, df4)
    res3, tt3 = getstats_diff(msc_ext2, msc_ext4, msc_std2, msc_std4, df2, df4)
    res4, tt4 = getstats_diff(mtc_ext2, mtc_ext4, mtc_std2, mtc_std4, df2, df4)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_moisture_budget_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # aerosol fast response1
    forcingstr = "Aerosol fast response"
    forcingfname = "fastaerosol2"
    res1, tt1 = getstats_diff(pre_ext3, pre_ext5, pre_std3, pre_std5, df3, df5)
    res2, tt2 = getstats_diff(dq_ext3, dq_ext5, dq_std3, dq_std5, df3, df5)
    res3, tt3 = getstats_diff(msc_ext3, msc_ext5, msc_std3, msc_std5, df3, df5)
    res4, tt4 = getstats_diff(mtc_ext3, mtc_ext5, mtc_std3, mtc_std5, df3, df5)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_moisture_budget_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # aerosol slow response1
    forcingstr = "Aerosol slow response"
    forcingfname = "slowaerosol1"
    res1, tt1 = getstats_diff(pre_ext4, pre_ext5, pre_std4, pre_std5, df4, df5)
    res2, tt2 = getstats_diff(dq_ext4, dq_ext5, dq_std4, dq_std5, df4, df5)
    res3, tt3 = getstats_diff(msc_ext4, msc_ext5, msc_std4, msc_std5, df4, df5)
    res4, tt4 = getstats_diff(mtc_ext4, mtc_ext5, mtc_std4, mtc_std5, df4, df5)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_moisture_budget_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # aerosol slow response2
    forcingstr = "Aerosol slow response"
    forcingfname = "slowaerosol2"
    res1, tt1 = getstats_diff(pre_ext2, pre_ext3, pre_std2, pre_std3, df2, df3)
    res2, tt2 = getstats_diff(dq_ext2, dq_ext3, dq_std2, dq_std3, df2, df3)
    res3, tt3 = getstats_diff(msc_ext2, msc_ext3, msc_std2, msc_std3, df2, df3)
    res4, tt4 = getstats_diff(mtc_ext2, mtc_ext3, mtc_std2, mtc_std3, df2, df3)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_moisture_budget_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # GHG and natural forcing
    forcingstr = "GHG and natural forcings"
    forcingfname = "GHGforcings"
    res1, tt1 = getstats_diff(pre_ext5, pre_ext1, pre_std5, pre_std1, df5, df1)
    res2, tt2 = getstats_diff(dq_ext5, dq_ext1, dq_std5, dq_std1, df5, df1)
    res3, tt3 = getstats_diff(msc_ext5, msc_ext1, msc_std5, msc_std1, df5, df1)
    res4, tt4 = getstats_diff(mtc_ext5, mtc_ext1, mtc_std5, mtc_std1, df5, df1)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_moisture_budget_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # All forcings
    forcingstr = "All forcings"
    forcingfname = "allforcings"
    res1, tt1 = getstats_diff(pre_ext2, pre_ext1, pre_std2, pre_std1, df2, df1)
    res2, tt2 = getstats_diff(dq_ext2, dq_ext1, dq_std2, dq_std1, df2, df1)
    res3, tt3 = getstats_diff(msc_ext2, msc_ext1, msc_std2, msc_std1, df2, df1)
    res4, tt4 = getstats_diff(mtc_ext2, mtc_ext1, mtc_std2, mtc_std1, df2, df1)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_moisture_budget_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    ######################################################################################
    print('Plotting MFC decomposition changes for extreme events...')
    labels = [r'$-\Delta \{\bigtriangledown\cdot(qu)\}^e$', r'$-\Delta (\partial q/\partial t)^e$',
              r'$-\Delta (q^e \{\bigtriangledown\cdot u\}^e)$', r'$-\Delta (u^e\cdot\{\bigtriangledown q\}^e)$']

    # all aerosol foring
    forcingstr = "All aerosol forcings"
    forcingfname = "allaerosols"
    res1, tt1 = getstats_diff(mfc_ext2, mfc_ext5, mfc_std2, mfc_std5, df2, df5)
    res2, tt2 = getstats_diff(dq_ext2, dq_ext5, dq_std2, dq_std5, df2, df5)
    res3, tt3 = getstats_diff(msc_ext2, msc_ext5, msc_std2, msc_std5, df2, df5)
    res4, tt4 = getstats_diff(mtc_ext2, mtc_ext5, mtc_std2, mtc_std5, df2, df5)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_mfc_decompose_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # aerosol fast response1
    forcingstr = "Aerosol fast response"
    forcingfname = "fastaerosol1"
    res1, tt1 = getstats_diff(mfc_ext2, mfc_ext4, mfc_std2, mfc_std4, df2, df4)
    res2, tt2 = getstats_diff(dq_ext2, dq_ext4, dq_std2, dq_std4, df2, df4)
    res3, tt3 = getstats_diff(msc_ext2, msc_ext4, msc_std2, msc_std4, df2, df4)
    res4, tt4 = getstats_diff(mtc_ext2, mtc_ext4, mtc_std2, mtc_std4, df2, df4)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_mfc_decompose_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # aerosol fast response1
    forcingstr = "Aerosol fast response"
    forcingfname = "fastaerosol2"
    res1, tt1 = getstats_diff(mfc_ext3, mfc_ext5, mfc_std3, mfc_std5, df3, df5)
    res2, tt2 = getstats_diff(dq_ext3, dq_ext5, dq_std3, dq_std5, df3, df5)
    res3, tt3 = getstats_diff(msc_ext3, msc_ext5, msc_std3, msc_std5, df3, df5)
    res4, tt4 = getstats_diff(mtc_ext3, mtc_ext5, mtc_std3, mtc_std5, df3, df5)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_mfc_decompose_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # aerosol slow response1
    forcingstr = "Aerosol slow response"
    forcingfname = "slowaerosol1"
    res1, tt1 = getstats_diff(mfc_ext4, mfc_ext5, mfc_std4, mfc_std5, df4, df5)
    res2, tt2 = getstats_diff(dq_ext4, dq_ext5, dq_std4, dq_std5, df4, df5)
    res3, tt3 = getstats_diff(msc_ext4, msc_ext5, msc_std4, msc_std5, df4, df5)
    res4, tt4 = getstats_diff(mtc_ext4, mtc_ext5, mtc_std4, mtc_std5, df4, df5)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_mfc_decompose_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # aerosol slow response2
    forcingstr = "Aerosol slow response"
    forcingfname = "slowaerosol2"
    res1, tt1 = getstats_diff(mfc_ext2, mfc_ext3, mfc_std2, mfc_std3, df2, df3)
    res2, tt2 = getstats_diff(dq_ext2, dq_ext3, dq_std2, dq_std3, df2, df3)
    res3, tt3 = getstats_diff(msc_ext2, msc_ext3, msc_std2, msc_std3, df2, df3)
    res4, tt4 = getstats_diff(mtc_ext2, mtc_ext3, mtc_std2, mtc_std3, df2, df3)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_mfc_decompose_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # GHG and natural forcing
    forcingstr = "GHG and natural forcings"
    forcingfname = "GHGforcings"
    res1, tt1 = getstats_diff(mfc_ext5, mfc_ext1, mfc_std5, mfc_std1, df5, df1)
    res2, tt2 = getstats_diff(dq_ext5, dq_ext1, dq_std5, dq_std1, df5, df1)
    res3, tt3 = getstats_diff(msc_ext5, msc_ext1, msc_std5, msc_std1, df5, df1)
    res4, tt4 = getstats_diff(mtc_ext5, mtc_ext1, mtc_std5, mtc_std1, df5, df1)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_mfc_decompose_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

    # All forcings
    forcingstr = "All forcings"
    forcingfname = "allforcings"
    res1, tt1 = getstats_diff(mfc_ext2, mfc_ext1, mfc_std2, mfc_std1, df2, df1)
    res2, tt2 = getstats_diff(dq_ext2, dq_ext1, dq_std2, dq_std1, df2, df1)
    res3, tt3 = getstats_diff(msc_ext2, msc_ext1, msc_std2, msc_std1, df2, df1)
    res4, tt4 = getstats_diff(mtc_ext2, mtc_ext1, mtc_std2, mtc_std1, df2, df1)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_mfc_decompose_"+forcingfname
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 0)
    plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, res4, tt4, labels, forcingstr, fname, 1)

# #################################################################################################
# # plot for different forcings
# #################################################################################################
# print('plotting responses...')
#
# # all aerosol foring
# forcingstr = "All aerosol forcings"
# forcingfname = "allaerosols"
#
# res_total, tt_total = getstats_mean((mfc2-mfc5) * 86400 * 1000)
# res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd5, vwnd5, q2-q5, ps5, ptop) * 86400 * 1000)
# res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd5, vwnd2-vwnd5, q5, ps5, ptop) * 86400 * 1000)
#
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
# print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)
#
#
# # aerosol fast response1
# forcingstr = "Aerosol fast response"
# forcingfname = "fastaerosol1"
#
# res_total, tt_total = getstats_mean((mfc2-mfc4) * 86400 * 1000)
# res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd4, vwnd4, q2-q4, ps4, ptop) * 86400 * 1000)
# res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd4, vwnd2-vwnd4, q4, ps4, ptop) * 86400 * 1000)
#
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
# print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)
#
# # aerosol slow response1
# forcingstr = "Aerosol slow response"
# forcingfname = "slowaerosol1"
#
# res_total, tt_total = getstats_mean((mfc4-mfc5) * 86400 * 1000)
# res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd5, vwnd5, q4-q5, ps5, ptop) * 86400 * 1000)
# res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd4-uwnd5, vwnd4-vwnd5, q5, ps5, ptop) * 86400 * 1000)
#
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
# print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)
#
# # aerosol fast response2
# forcingstr = "Aerosol fast response"
# forcingfname = "fastaerosol2"
#
# res_total, tt_total = getstats_mean((mfc3-mfc5) * 86400 * 1000)
# res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd5, vwnd5, q3-q5, ps5, ptop) * 86400 * 1000)
# res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd3-uwnd5, vwnd3-vwnd5, q5, ps5, ptop) * 86400 * 1000)
#
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
# print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)
#
# # aerosol slow response2
# forcingstr = "Aerosol slow response"
# forcingfname = "slowaerosol2"
#
# res_total, tt_total = getstats_mean((mfc2-mfc3) * 86400 * 1000)
# res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd3, vwnd3, q2-q3, ps3, ptop) * 86400 * 1000)
# res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd3, vwnd2-vwnd3, q3, ps3, ptop) * 86400 * 1000)
#
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
# print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)
#
# # GHG and natural forcing
# forcingstr = "GHG and natural forcings"
# forcingfname = "GHGforcings"
#
# res_total, tt_total = getstats_mean((mfc5-mfc1) * 86400 * 1000)
# res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd1, vwnd1, q5-q1, ps1, ptop) * 86400 * 1000)
# res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd5-uwnd1, vwnd5-vwnd1, q1, ps1, ptop) * 86400 * 1000)
#
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
# print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)
#
# # All forcings
# forcingstr = "All forcings"
# forcingfname = "allforcings"
#
# res_total, tt_total = getstats_mean((mfc2-mfc1) * 86400 * 1000)
# res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd1, vwnd1, q2-q1, ps1, ptop) * 86400 * 1000)
# res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd1, vwnd2-vwnd1, q1, ps1, ptop) * 86400 * 1000)
#
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
# plotdiff(lons, lats, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
# print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)
#
#
# # plot all in one
# # total
# res1_total, tt1_total = getstats_mean((mfc2-mfc5) * 86400 * 1000)
# res1_thermo, tt1_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd5, vwnd5, q2-q5, ps5, ptop) * 86400 * 1000)
# res1_dyn, tt1_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd5, vwnd2-vwnd5, q5, ps5, ptop) * 86400 * 1000)
#
# # fast
# res2_total, tt2_total = getstats_mean((mfc2 + mfc3 - mfc4 - mfc5)/2 * 86400 * 1000)
# res2_thermo, tt2_thermo = getstats_mean(getmfc(lats, lons, levs, (uwnd4+uwnd5)/2, (vwnd4+vwnd5)/2, (q2+q3-q4-q5)/2, (ps4+ps5)/2, ptop) * 86400 * 1000)
# res2_dyn, tt2_dyn = getstats_mean(getmfc(lats, lons, levs, (uwnd2+uwnd3-uwnd4-uwnd5)/2, (vwnd2+vwnd3-vwnd4-vwnd5)/2, (q4+q5)/2, (ps4+ps5)/2, ptop) * 86400 * 1000)
#
#
# res3_total, tt3_total = getstats_mean((mfc4 + mfc2 - mfc5 - mfc3)/2 * 86400 * 1000)
# res3_thermo, tt3_thermo = getstats_mean(getmfc(lats, lons, levs, (uwnd5+uwnd3)/2, (vwnd5+vwnd3)/2, (q4+q2-q5-q3)/2, (ps5+ps3)/2, ptop) * 86400 * 1000)
# res3_dyn, tt3_dyn = getstats_mean(getmfc(lats, lons, levs, (uwnd4+uwnd2-uwnd5-uwnd3)/2, (vwnd4+vwnd2-vwnd5-vwnd3)/2, (q5+q3)/2, (ps5+ps3)/2, ptop) * 86400 * 1000)
#
#
# plotalldiff(lons, lats, res1_total, tt1_total, res1_thermo, tt1_thermo, res1_dyn, tt1_dyn, res2_total, tt2_total,
#             res2_thermo, tt2_thermo, res2_dyn, tt2_dyn, res3_total, tt3_total, res3_thermo, tt3_thermo, res3_dyn, tt3_dyn, 0)
#
# plotalldiff(lons, lats, res1_total, tt1_total, res1_thermo, tt1_thermo, res1_dyn, tt1_dyn, res2_total, tt2_total,
#             res2_thermo, tt2_thermo, res2_dyn, tt2_dyn, res3_total, tt3_total, res3_thermo, tt3_thermo, res3_dyn, tt3_dyn, 1)
