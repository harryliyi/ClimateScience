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
# reg_names = ['mainland SEA', 'Central India']
# reg_str = ['mainSEA', 'ctInd']
# reg_lats = [[10, 20], [16.5, 26.5]]
# reg_lons = [[100, 110], [74.5, 86.5]]

# reg_names = ['mainland SEA', 'South Asia']
# reg_str = ['mainSEA', 'SA']
# reg_lats = [[10, 20], [10, 35]]
# reg_lons = [[100, 110], [70, 90]]

reg_names = ['South India', 'North India']
reg_str = ['stInd', 'nrInd']
reg_lats = [[8, 20], [20, 28]]
reg_lons = [[70, 90], [65, 90]]

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

# calculate the difference of mean and significant is defined by if three quarters of samples have the same sign of mean


def get_stats_diff(var):

    res_mean = np.mean(var)
    count = 0
    n = len(var)
    for ivar in var:
        if (ivar*res_mean > 0):
            count = count + 1
    if (count > n*0.75):
        res_tt = 3
    else:
        res_tt = 0

    return res_mean, res_tt


# calculate hypothesis test of mean


def getstats_mean(var):
    n = var.shape[0]
    varmean = np.mean(var, axis=0)
    varstd = np.std(var, axis=0)

    varttest = varmean/(varstd/np.sqrt(n))

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
def plotdiff(lons, lats, res1, tt1, res2, tt2, res3, tt3, label, clevs, var_unit, forcingstr, fname, opt):
    fig = plt.figure()
    # Res1
    ax1 = fig.add_subplot(311)
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
    # clevs = np.arange(-18, 18.1, 3)
    cs = map.contourf(x, y, res1, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt1.max()]
        csm = plt.contourf(x, y, tt1, levels=levels, colors='none', hatches=["", "....."], alpha=0)
#    print(tt_total)

    # Res2
    ax2 = fig.add_subplot(312)
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
    ax3 = fig.add_subplot(313)
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

    fig.subplots_adjust(hspace=0.2)
    cbar_ax = fig.add_axes([0.69, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax)
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
# plot for differences
def plotalldiff(lons, lats, res, tt, clevs, var_unit, title, fname, opt):
    fig = plt.figure()
    # total response,fast+slow
    ax1 = fig.add_subplot(331)
    # ax1.set_title(r'$\Delta_{total} MFC$, regional mean='+str(round(np.mean(res1_total), 4))+var_unit,fontsize=5,pad=3)
    ax1.set_title('Total Aerosol', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    # clevs = np.arange(-1.4, 1.5, 0.2)
    cs = map.contourf(x, y, res[0], clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt[0].max()]
        csm = ax1.contourf(x, y, tt[0], levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax1.xaxis.set_tick_params(labelsize=4)
    ax1.yaxis.set_tick_params(labelsize=4)
    ax1.set_ylabel(r'$\delta MC$', fontsize=7, labelpad=12)

    # total response,fast
    ax2 = fig.add_subplot(332)
#    ax2.set_title(r'$\Delta_{fast} MFC$, regional mean='+str(round(np.mean(res2_total), 4))+var_unit,fontsize=5,pad=3)
    ax2.set_title('Atmosphere-forced', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res[1], clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt[1].max()]
        csm = ax2.contourf(x, y, tt[1], levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax2.xaxis.set_tick_params(labelsize=4)
    ax2.yaxis.set_tick_params(labelsize=4)

    # total response,slow
    ax3 = fig.add_subplot(333)
#    ax3.set_title(r'$\Delta_{slow} MFC$, regional mean='+str(round(np.mean(res3_total), 4))+var_unit,fontsize=5,pad=3)
    ax3.set_title('Ocean-mediated', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res[2], clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt[2].max()]
        csm = ax3.contourf(x, y, tt[2], levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax3.xaxis.set_tick_params(labelsize=4)
    ax3.yaxis.set_tick_params(labelsize=4)

    # thermodynamic response,fast+slow
    ax4 = fig.add_subplot(334)
    # ax4.set_title(r'$\Delta_{total,thermo} MFC$, regional mean='+str(round(np.mean(res1_thermo), 4))+var_unit,fontsize=5,pad=3)
    # ax4.set_title(r'$\Delta_{total,thermo} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res[3], clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt[3].max()]
        csm = ax4.contourf(x, y, tt[3], levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax4.xaxis.set_tick_params(labelsize=4)
    ax4.yaxis.set_tick_params(labelsize=4)
    ax4.set_ylabel(r'$\delta TH$', fontsize=7, labelpad=12)

    # thermodynamic response,fast
    ax5 = fig.add_subplot(335)
    # ax5.set_title(r'$\Delta_{fast,thermo} MFC$, regional mean='+str(round(np.mean(res2_thermo), 4))+var_unit,fontsize=5,pad=3)
    # ax5.set_title(r'$\Delta_{fast,thermo} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res[4], clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt[4].max()]
        csm = ax5.contourf(x, y, tt[4], levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax5.xaxis.set_tick_params(labelsize=4)
    ax5.yaxis.set_tick_params(labelsize=4)

    # thermodynamic response,slow
    ax6 = fig.add_subplot(336)
    # ax6.set_title(r'$\Delta_{slow,thermo} MFC$, regional mean='+str(round(np.mean(res3_thermo), 4))+var_unit,fontsize=5,pad=3)
    # ax6.set_title(r'$\Delta_{slow,thermo} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res[5], clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt[5].max()]
        csm = ax6.contourf(x, y, tt[5], levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax6.xaxis.set_tick_params(labelsize=4)
    ax6.yaxis.set_tick_params(labelsize=4)

    # dynamic response
    ax7 = fig.add_subplot(337)
    # ax7.set_title(r'$\Delta_{total,dynamics} MFC$, regional mean='+str(round(np.mean(res1_dyn), 5))+var_unit,fontsize=5,pad=3)
    # ax7.set_title(r'$\Delta_{total,dynamics} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res[6], clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt[6].max()]
        csm = ax7.contourf(x, y, tt[6], levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax7.xaxis.set_tick_params(labelsize=4)
    ax7.yaxis.set_tick_params(labelsize=4)
    ax7.set_ylabel(r'$\delta DY$', fontsize=7, labelpad=12)

    # dynamic response
    ax8 = fig.add_subplot(338)
    # ax8.set_title(r'$\Delta_{fast,dynamics} MFC$, regional mean='+str(round(np.mean(res2_dyn), 5))+var_unit,fontsize=5,pad=3)
    # ax8.set_title(r'$\Delta_{fast,dynamics} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res[7], clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt[7].max()]
        csm = ax8.contourf(x, y, tt[7], levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax8.xaxis.set_tick_params(labelsize=4)
    ax8.yaxis.set_tick_params(labelsize=4)

    # dynamic response
    ax9 = fig.add_subplot(339)
    # ax9.set_title(r'$\Delta_{slow,dynamics} MFC$, regional mean='+str(round(np.mean(res3_dyn), 5))+var_unit,fontsize=5,pad=3)
    # ax9.set_title(r'$\Delta_{slow,dynamics} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res[8], clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt[8].max()]
        csm = ax9.contourf(x, y, tt[8], levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax9.xaxis.set_tick_params(labelsize=4)
    ax9.yaxis.set_tick_params(labelsize=4)

    fig.subplots_adjust(bottom=0.2, wspace=0.15, hspace=0.03)
    cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('['+var_unit+']', fontsize=6, labelpad=-0.3)

    # add title
    if (opt == 1):
        plt.savefig(fname+"_with_siglev.png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fname+".png", dpi=600, bbox_inches='tight')
    plt.suptitle(title, fontsize=7, y=0.95)
#    plt.tight_layout()
    if (opt == 1):
        plt.savefig(fname+"_with_siglev.pdf", bbox_inches='tight')
    else:
        plt.savefig(fname+".pdf", bbox_inches='tight')

    plt.close(fig)


################################################################################################
# plot for the Covariance fraction

def plotcovbars(res_frac, reg_names, percentile, fname):

    index = np.arange(9)+1
    xticks = [r'$\delta MC$', r'$\delta DY$', r'$\delta TH$',
              r'$\delta MC$', r'$\delta DY$', r'$\delta TH$',
              r'$\delta MC$', r'$\delta DY$', r'$\delta TH$', ]
    # xlabels = ['Total', 'ATM', 'OCN']
    xlabels = ['Total', 'Atmospheric-forced', 'Ocean-mediated']
    fig = plt.figure()

    ax1 = fig.add_subplot(211)

    ax1.bar(index[0:3], res_frac[0:3], width=0.8, color='orange', linewidth=0)
    ax1.bar(index[3:6], res_frac[3:6], width=0.8, color='yellowgreen', linewidth=0)
    ax1.bar(index[6:9], res_frac[6:9], width=0.8, color='darkcyan', linewidth=0)
    ax1.set_ylabel(reg_names[0], fontsize=8, labelpad=3)
    ax1.set_xticklabels([])
    ax1.yaxis.set_tick_params(labelsize=6)

    ax2 = fig.add_subplot(212)

    ax2.bar(index[0:3], res_frac[9:12], width=0.8, color='orange', linewidth=0)
    ax2.bar(index[3:6], res_frac[12:15], width=0.8, color='yellowgreen', linewidth=0)
    ax2.bar(index[6:9], res_frac[15:18], width=0.8, color='darkcyan', linewidth=0)
    ax2.set_ylabel(reg_names[1], fontsize=8, labelpad=3)
    ax2.set_xticks(index)
    ax2.xaxis.set_tick_params(labelsize=6)
    ax2.yaxis.set_tick_params(labelsize=6)
    ax2.set_xticklabels(xticks, fontsize=7)
    # ax2.set_xlabel(xlabels, fontsize=6, labelpad=-2)

    fig.subplots_adjust(bottom=0.12)

    ax3 = ax2.twiny()
    ax3.xaxis.set_ticks_position("bottom")
    ax3.xaxis.set_label_position("bottom")
    ax3.spines["bottom"].set_position(("axes", -0.15))
    ax2.spines["bottom"].set_visible(True)
    ax3.set_frame_on(False)
    ax3.patch.set_visible(False)
    ax3.set_xticks([.2, .5, .8])
    ax3.set_xticklabels(xlabels, fontsize=8)

    # add title
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(fname+".png", dpi=600, bbox_inches='tight')
    plt.suptitle("Fraction of variance of each Moisture Convergence component to P field at "+str(percentile)+"th", fontsize=8, y=0.95)
#    plt.tight_layout()
    plt.savefig(fname+".pdf", bbox_inches='tight')

    plt.close(fig)

################################################################################################
# plot for the means bars


def plotmeanbars(res_mean, reg_names, percentile, fname):

    index = np.arange(12)+1
    xticks = [r'$\delta P$', r'$\delta MC$', r'$\delta DY$', r'$\delta TH$',
              r'$\delta P$', r'$\delta MC$', r'$\delta DY$', r'$\delta TH$',
              r'$\delta P$', r'$\delta MC$', r'$\delta DY$', r'$\delta TH$', ]
    # xlabels = ['Total', 'ATM', 'OCN']
    xlabels = ['Total', 'Atmospheric-forced', 'Ocean-mediated']
    fig = plt.figure()

    ax1 = fig.add_subplot(211)

    ax1.bar(index[0:4], res_mean[0:4], width=0.8, color='orange', linewidth=0)
    ax1.bar(index[4:8], res_mean[4:8], width=0.8, color='yellowgreen', linewidth=0)
    ax1.bar(index[8:12], res_mean[8:12], width=0.8, color='darkcyan', linewidth=0)
    ax1.set_ylabel(reg_names[0]+' (mm/day)', fontsize=8, labelpad=3)
    ax1.set_xticklabels([])
    ax1.yaxis.set_tick_params(labelsize=6)

    ax2 = fig.add_subplot(212)

    ax2.bar(index[0:4], res_mean[12:16], width=0.8, color='orange', linewidth=0)
    ax2.bar(index[4:8], res_mean[16:20], width=0.8, color='yellowgreen', linewidth=0)
    ax2.bar(index[8:12], res_mean[20:24], width=0.8, color='darkcyan', linewidth=0)
    ax2.set_ylabel(reg_names[1]+' (mm/day)', fontsize=8, labelpad=3)
    ax2.set_xticks(index)
    ax2.xaxis.set_tick_params(labelsize=6)
    ax2.yaxis.set_tick_params(labelsize=6)
    ax2.set_xticklabels(xticks, fontsize=7)
    # ax2.set_xlabel(xlabels, fontsize=6, labelpad=-2)

    fig.subplots_adjust(bottom=0.12, wspace=0.2)

    ax3 = ax2.twiny()
    ax3.xaxis.set_ticks_position("bottom")
    ax3.xaxis.set_label_position("bottom")
    ax3.spines["bottom"].set_position(("axes", -0.15))
    ax2.spines["bottom"].set_visible(True)
    ax3.set_frame_on(False)
    ax3.patch.set_visible(False)
    ax3.set_xticks([.2, .5, .8])
    ax3.set_xticklabels(xlabels, fontsize=8)

    # add title
    fig.subplots_adjust(wspace=0.1)
    plt.savefig(fname+".png", dpi=600, bbox_inches='tight')
    plt.suptitle("Contribution of each Moisture Convergence component to P field at "+str(percentile)+"th", fontsize=8, y=0.95)
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
print('Calculating convergence of mass...')
starttime = datetime.datetime.now()

convu1 = getconv(lats, lons, u1, v1)
convu2 = getconv(lats, lons, u2, v2)
convu3 = getconv(lats, lons, u3, v3)
convu4 = getconv(lats, lons, u4, v4)
convu5 = getconv(lats, lons, u5, v5)
print(convu1.shape)

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


print('Calculating pressure dlevels...')
dlevs1 = dpres_plevel(levs, ps1, ptop)
dlevs2 = dpres_plevel(levs, ps2, ptop)
dlevs3 = dpres_plevel(levs, ps3, ptop)
dlevs4 = dpres_plevel(levs, ps4, ptop)
dlevs5 = dpres_plevel(levs, ps5, ptop)
print(dlevs1.shape)

#################################################################################################
# plot climatology
#################################################################################################

# select precip extreme precentile
for idx_percent, percentile in enumerate(percentile_ranges):

    outdir_percent = outdir+str(percentile)+'th/'
    percentile_top = percentile_tops[idx_percent]
    percentile_bot = percentile_bots[idx_percent]
    idx_top = int(ntime*percentile_top/100)
    idx_bot = int(ntime*percentile_bot/100)

    print('Current percentile is '+str(percentile)+'th...')

    pre_idx1 = np.argsort(np.argsort(pre1, axis=0), axis=0)
    pre_idx2 = np.argsort(np.argsort(pre2, axis=0), axis=0)
    pre_idx3 = np.argsort(np.argsort(pre3, axis=0), axis=0)
    pre_idx4 = np.argsort(np.argsort(pre4, axis=0), axis=0)
    pre_idx5 = np.argsort(np.argsort(pre5, axis=0), axis=0)

    pre_ext1 = np.zeros((nlats, nlons))
    pre_ext2 = np.zeros((nlats, nlons))
    pre_ext3 = np.zeros((nlats, nlons))
    pre_ext4 = np.zeros((nlats, nlons))
    pre_ext5 = np.zeros((nlats, nlons))

    msc_ext1 = np.zeros((nlats, nlons))
    msc_ext2 = np.zeros((nlats, nlons))
    msc_ext3 = np.zeros((nlats, nlons))
    msc_ext4 = np.zeros((nlats, nlons))
    msc_ext5 = np.zeros((nlats, nlons))

    res_allaero_all = np.zeros((nlats, nlons))
    res_allaero_tt1 = np.zeros((nlats, nlons))
    res_allaero_dyn = np.zeros((nlats, nlons))
    res_allaero_tt2 = np.zeros((nlats, nlons))
    res_allaero_thm = np.zeros((nlats, nlons))
    res_allaero_tt3 = np.zeros((nlats, nlons))

    res_fastaero_all = np.zeros((nlats, nlons))
    res_fastaero_tt1 = np.zeros((nlats, nlons))
    res_fastaero_dyn = np.zeros((nlats, nlons))
    res_fastaero_tt2 = np.zeros((nlats, nlons))
    res_fastaero_thm = np.zeros((nlats, nlons))
    res_fastaero_tt3 = np.zeros((nlats, nlons))

    res_slowaero_all = np.zeros((nlats, nlons))
    res_slowaero_tt1 = np.zeros((nlats, nlons))
    res_slowaero_dyn = np.zeros((nlats, nlons))
    res_slowaero_tt2 = np.zeros((nlats, nlons))
    res_slowaero_thm = np.zeros((nlats, nlons))
    res_slowaero_tt3 = np.zeros((nlats, nlons))

    res_GHG_all = np.zeros((nlats, nlons))
    res_GHG_tt1 = np.zeros((nlats, nlons))
    res_GHG_dyn = np.zeros((nlats, nlons))
    res_GHG_tt2 = np.zeros((nlats, nlons))
    res_GHG_thm = np.zeros((nlats, nlons))
    res_GHG_tt3 = np.zeros((nlats, nlons))

    res_all_all = np.zeros((nlats, nlons))
    res_all_tt1 = np.zeros((nlats, nlons))
    res_all_dyn = np.zeros((nlats, nlons))
    res_all_tt2 = np.zeros((nlats, nlons))
    res_all_thm = np.zeros((nlats, nlons))
    res_all_tt3 = np.zeros((nlats, nlons))

    ################################################################################
    print('Plotting dynamical and thermodynamical changes for extreme events...')

    for ilat in range(nlats):
        for ilon in range(nlons):
            pre_bool1 = (pre_idx1[:, ilat, ilon] >= idx_bot) & (pre_idx1[:, ilat, ilon] < idx_top)
            pre_bool2 = (pre_idx2[:, ilat, ilon] >= idx_bot) & (pre_idx2[:, ilat, ilon] < idx_top)
            pre_bool3 = (pre_idx3[:, ilat, ilon] >= idx_bot) & (pre_idx3[:, ilat, ilon] < idx_top)
            pre_bool4 = (pre_idx4[:, ilat, ilon] >= idx_bot) & (pre_idx4[:, ilat, ilon] < idx_top)
            pre_bool5 = (pre_idx5[:, ilat, ilon] >= idx_bot) & (pre_idx5[:, ilat, ilon] < idx_top)

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

            msc_ext1[ilat, ilon] = np.mean(msc1[pre_bool1, ilat, ilon])
            msc_ext2[ilat, ilon] = np.mean(msc2[pre_bool2, ilat, ilon])
            msc_ext3[ilat, ilon] = np.mean(msc3[pre_bool3, ilat, ilon])
            msc_ext4[ilat, ilon] = np.mean(msc4[pre_bool4, ilat, ilon])
            msc_ext5[ilat, ilon] = np.mean(msc5[pre_bool5, ilat, ilon])

            q1_ext = q1[pre_bool1, :, ilat, ilon]
            q2_ext = q2[pre_bool2, :, ilat, ilon]
            q3_ext = q3[pre_bool3, :, ilat, ilon]
            q4_ext = q4[pre_bool4, :, ilat, ilon]
            q5_ext = q5[pre_bool5, :, ilat, ilon]

            convu1_ext = convu1[pre_bool1, :, ilat, ilon]
            convu2_ext = convu2[pre_bool2, :, ilat, ilon]
            convu3_ext = convu3[pre_bool3, :, ilat, ilon]
            convu4_ext = convu4[pre_bool4, :, ilat, ilon]
            convu5_ext = convu5[pre_bool5, :, ilat, ilon]

            dlevs1_ext = dlevs1[pre_bool1, :, ilat, ilon]
            dlevs2_ext = dlevs2[pre_bool2, :, ilat, ilon]
            dlevs3_ext = dlevs3[pre_bool3, :, ilat, ilon]
            dlevs4_ext = dlevs4[pre_bool4, :, ilat, ilon]
            dlevs5_ext = dlevs5[pre_bool5, :, ilat, ilon]

            ########################################
            # Calculate all aerosol response
            # temp = q2[pre_bool2, :, ilat, ilon]
            # print(temp.shape)

            res_allaero_all[ilat, ilon], res_allaero_tt1[ilat, ilon] = getstats_mean(
                msc2[pre_bool2, ilat, ilon] - msc5[pre_bool5, ilat, ilon])
            res_allaero_dyn[ilat, ilon], res_allaero_tt2[ilat, ilon] = getstats_mean(
                np.sum(q5_ext * (convu2_ext - convu5_ext) * dlevs5_ext, axis=1)/oro_water/g)
            res_allaero_thm[ilat, ilon], res_allaero_tt3[ilat, ilon] = getstats_mean(
                np.sum((q2_ext - q5_ext) * convu5_ext * dlevs5_ext, axis=1)/oro_water/g)

            res_allaero_dyn[ilat, ilon] = res_allaero_dyn[ilat, ilon] * 86400 * 1000
            res_allaero_thm[ilat, ilon] = res_allaero_thm[ilat, ilon] * 86400 * 1000

            ########################################
            # Calculate aerosol fast response
            temp1 = msc2[pre_bool2, ilat, ilon] - msc4[pre_bool4, ilat, ilon]
            temp2 = msc3[pre_bool3, ilat, ilon] - msc5[pre_bool5, ilat, ilon]
            res_fastaero_all[ilat, ilon], res_fastaero_tt1[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            temp1 = np.sum(q4_ext * (convu2_ext - convu4_ext) * dlevs4_ext, axis=1)/oro_water/g
            temp2 = np.sum(q5_ext * (convu3_ext - convu5_ext) * dlevs5_ext, axis=1)/oro_water/g
            res_fastaero_dyn[ilat, ilon], res_fastaero_tt2[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            temp1 = np.sum((q2_ext - q4_ext) * convu4_ext * dlevs4_ext, axis=1)/oro_water/g
            temp2 = np.sum((q3_ext - q5_ext) * convu5_ext * dlevs5_ext, axis=1)/oro_water/g
            res_fastaero_thm[ilat, ilon], res_fastaero_tt3[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            res_fastaero_dyn[ilat, ilon] = res_fastaero_dyn[ilat, ilon] * 86400 * 1000
            res_fastaero_thm[ilat, ilon] = res_fastaero_thm[ilat, ilon] * 86400 * 1000

            ########################################
            # Calculate aerosol slow response
            temp1 = msc2[pre_bool2, ilat, ilon] - msc3[pre_bool3, ilat, ilon]
            temp2 = msc4[pre_bool4, ilat, ilon] - msc5[pre_bool5, ilat, ilon]
            res_slowaero_all[ilat, ilon], res_slowaero_tt1[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            temp1 = np.sum(q3_ext * (convu2_ext - convu3_ext) * dlevs3_ext, axis=1)/oro_water/g
            temp2 = np.sum(q5_ext * (convu4_ext - convu5_ext) * dlevs5_ext, axis=1)/oro_water/g
            res_slowaero_dyn[ilat, ilon], res_slowaero_tt2[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            temp1 = np.sum((q2_ext - q3_ext) * convu3_ext * dlevs3_ext, axis=1)/oro_water/g
            temp2 = np.sum((q4_ext - q5_ext) * convu5_ext * dlevs5_ext, axis=1)/oro_water/g
            res_slowaero_thm[ilat, ilon], res_slowaero_tt3[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            res_slowaero_dyn[ilat, ilon] = res_slowaero_dyn[ilat, ilon] * 86400 * 1000
            res_slowaero_thm[ilat, ilon] = res_slowaero_thm[ilat, ilon] * 86400 * 1000

            ########################################
            # GHG and natural forcing
            res_GHG_all[ilat, ilon], res_GHG_tt1[ilat, ilon] = getstats_mean(
                msc5[pre_bool5, ilat, ilon] - msc1[pre_bool1, ilat, ilon])
            res_GHG_dyn[ilat, ilon], res_GHG_tt2[ilat, ilon] = getstats_mean(
                np.sum(q1_ext * (convu5_ext - convu1_ext) * dlevs1_ext, axis=1)/oro_water/g)
            res_GHG_thm[ilat, ilon], res_GHG_tt3[ilat, ilon] = getstats_mean(
                np.sum((q5_ext - q1_ext) * convu1_ext * dlevs1_ext, axis=1)/oro_water/g)

            res_GHG_dyn[ilat, ilon] = res_GHG_dyn[ilat, ilon] * 86400 * 1000
            res_GHG_thm[ilat, ilon] = res_GHG_thm[ilat, ilon] * 86400 * 1000

            ########################################
            # All forcings
            res_all_all[ilat, ilon], res_all_tt1[ilat, ilon] = getstats_mean(
                msc2[pre_bool2, ilat, ilon] - msc1[pre_bool1, ilat, ilon])
            res_all_dyn[ilat, ilon], res_all_tt2[ilat, ilon] = getstats_mean(
                np.sum(q1_ext * (convu2_ext - convu1_ext) * dlevs1_ext, axis=1)/oro_water/g)
            res_all_thm[ilat, ilon], res_all_tt3[ilat, ilon] = getstats_mean(
                np.sum((q2_ext - q1_ext) * convu1_ext * dlevs1_ext, axis=1)/oro_water/g)

            res_all_dyn[ilat, ilon] = res_all_dyn[ilat, ilon] * 86400 * 1000
            res_all_thm[ilat, ilon] = res_all_thm[ilat, ilon] * 86400 * 1000

    labels = ['Total Response', 'Dynamic Response', 'Thermodynamic Response']
    clevs = np.arange(-18, 18.1, 3)
    var_unit = 'mm/day'

    ########################################
    # all aerosol response
    forcingstr = "All aerosol forcings"
    forcingfname = "allaerosols"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_decompose_"+forcingfname
    plotdiff(lons, lats, res_allaero_all, res_allaero_tt1, res_allaero_dyn, res_allaero_tt2,
             res_allaero_thm, res_allaero_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_allaero_all, res_allaero_tt1, res_allaero_dyn, res_allaero_tt2,
             res_allaero_thm, res_allaero_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # aerosol fast response
    forcingstr = "Aerosol fast response"
    forcingfname = "fastaerosol"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_decompose_"+forcingfname
    plotdiff(lons, lats, res_fastaero_all, res_fastaero_tt1, res_fastaero_dyn, res_fastaero_tt2,
             res_fastaero_thm, res_fastaero_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_fastaero_all, res_fastaero_tt1, res_fastaero_dyn, res_fastaero_tt2,
             res_fastaero_thm, res_fastaero_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # aerosol slow response
    forcingstr = "Aerosol slow response"
    forcingfname = "slowaerosol"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_decompose_"+forcingfname
    plotdiff(lons, lats, res_slowaero_all, res_slowaero_tt1, res_slowaero_dyn, res_slowaero_tt2,
             res_slowaero_thm, res_slowaero_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_slowaero_all, res_slowaero_tt1, res_slowaero_dyn, res_slowaero_tt2,
             res_slowaero_thm, res_slowaero_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # GHG and natural forcing
    forcingstr = "GHG and natural forcings"
    forcingfname = "GHGforcings"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_decompose_"+forcingfname
    plotdiff(lons, lats, res_GHG_all, res_GHG_tt1, res_GHG_dyn, res_GHG_tt2,
             res_GHG_thm, res_GHG_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_GHG_all, res_GHG_tt1, res_GHG_dyn, res_GHG_tt2,
             res_GHG_thm, res_GHG_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # All forcings
    forcingstr = "All forcings"
    forcingfname = "allforcings"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_decompose_"+forcingfname
    plotdiff(lons, lats, res_all_all, res_all_tt1, res_all_dyn, res_all_tt2,
             res_all_thm, res_all_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_all_all, res_all_tt1, res_all_dyn, res_all_tt2,
             res_all_thm, res_all_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # All Aerosol in one
    res = [res_allaero_all, res_fastaero_all, res_slowaero_all,
           res_allaero_thm, res_fastaero_thm, res_slowaero_thm,
           res_allaero_dyn, res_fastaero_dyn, res_slowaero_dyn]

    print(res[0].shape)
    print(len(res))

    tt = [res_allaero_tt1, res_fastaero_tt1, res_slowaero_tt1,
          res_allaero_tt2, res_fastaero_tt2, res_slowaero_tt2,
          res_allaero_tt3, res_fastaero_tt3, res_slowaero_tt3]

    title = 'Changes in the product of humidity and mass convergence to aerosol forcing'
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_decompose_aerosolsinone"
    plotalldiff(lons, lats, res, tt, clevs, var_unit, title, fname, 0)
    plotalldiff(lons, lats, res, tt, clevs, var_unit, title, fname, 1)

    #################################################################################################
    # plot for the spatial covariance fraction
    #################################################################################################

    PE_total = pre_ext2 - pre_ext5
    PE_fast = (pre_ext2 + pre_ext3 - pre_ext4 - pre_ext5)/2
    PE_slow = (pre_ext2 + pre_ext4 - pre_ext3 - pre_ext5)/2

    res_frac = []
    res_mean = []

    print(PE_total)
    print(PE_total.shape)

    for idx in range(len(reg_names)):
        reg_name = reg_names[idx]

        reg_latli = np.abs(lats - reg_lats[idx][0]).argmin()
        reg_latui = np.abs(lats - reg_lats[idx][1]).argmin()

        reg_lonli = np.abs(lons - reg_lons[idx][0]).argmin()
        reg_lonui = np.abs(lons - reg_lons[idx][1]).argmin()
        print(lats[reg_latli])
        print(lats[reg_latui])

        PE_var_total = np.var(PE_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        PE_var_fast = np.var(PE_fast[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        PE_var_slow = np.var(PE_slow[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])

        reg_cov_res1_tot = np.cov(PE_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                                  res_allaero_all[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
        reg_cov_res1_dyn = np.cov(PE_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                                  res_allaero_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
        reg_cov_res1_thm = np.cov(PE_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                                  res_allaero_thm[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]

        reg_cov_res2_tot = np.cov(PE_fast[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                                  res_fastaero_all[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
        reg_cov_res2_dyn = np.cov(PE_fast[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                                  res_fastaero_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
        reg_cov_res2_thm = np.cov(PE_fast[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                                  res_fastaero_thm[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]

        reg_cov_res3_tot = np.cov(PE_slow[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                                  res_slowaero_all[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
        reg_cov_res3_dyn = np.cov(PE_slow[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                                  res_slowaero_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
        reg_cov_res3_thm = np.cov(PE_slow[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                                  res_slowaero_thm[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]

        res_frac.append(reg_cov_res1_tot/PE_var_total)
        res_frac.append(reg_cov_res1_dyn/PE_var_total)
        res_frac.append(reg_cov_res1_thm/PE_var_total)
        res_frac.append(reg_cov_res2_tot/PE_var_fast)
        res_frac.append(reg_cov_res2_dyn/PE_var_fast)
        res_frac.append(reg_cov_res2_thm/PE_var_fast)
        res_frac.append(reg_cov_res3_tot/PE_var_slow)
        res_frac.append(reg_cov_res3_dyn/PE_var_slow)
        res_frac.append(reg_cov_res3_thm/PE_var_slow)

        PE_var_total = np.mean(PE_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        PE_var_fast = np.mean(PE_fast[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        PE_var_slow = np.mean(PE_slow[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])

        reg_cov_res1_tot = np.mean(res_allaero_all[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        reg_cov_res1_dyn = np.mean(res_allaero_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        reg_cov_res1_thm = np.mean(res_allaero_thm[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])

        reg_cov_res2_tot = np.mean(res_fastaero_all[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        reg_cov_res2_dyn = np.mean(res_fastaero_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        reg_cov_res2_thm = np.mean(res_fastaero_thm[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])

        reg_cov_res3_tot = np.mean(res_slowaero_all[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        reg_cov_res3_dyn = np.mean(res_slowaero_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
        reg_cov_res3_thm = np.mean(res_slowaero_thm[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])

        res_mean.append(PE_var_total)
        res_mean.append(reg_cov_res1_tot)
        res_mean.append(reg_cov_res1_dyn)
        res_mean.append(reg_cov_res1_thm)
        res_mean.append(PE_var_fast)
        res_mean.append(reg_cov_res2_tot)
        res_mean.append(reg_cov_res2_dyn)
        res_mean.append(reg_cov_res2_thm)
        res_mean.append(PE_var_slow)
        res_mean.append(reg_cov_res3_tot)
        res_mean.append(reg_cov_res3_dyn)
        res_mean.append(reg_cov_res3_thm)

    print(res_frac)
    print(res_mean)
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_"+str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_decompose_conv"
    plotcovbars(res_frac, reg_names, percentile, fname+'_'+reg_str[0]+'and'+reg_str[1])
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_"+str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_decompose_mean"
    plotmeanbars(res_mean, reg_names, percentile, fname+'_'+reg_str[0]+'and'+reg_str[1])

    ################################################################################
    print('Plotting dynamical and thermodynamical relative changes for extreme events...')

    for ilat in range(nlats):
        for ilon in range(nlons):
            pre_bool1 = (pre_idx1[:, ilat, ilon] >= idx_bot) & (pre_idx1[:, ilat, ilon] < idx_top)
            pre_bool2 = (pre_idx2[:, ilat, ilon] >= idx_bot) & (pre_idx2[:, ilat, ilon] < idx_top)
            pre_bool3 = (pre_idx3[:, ilat, ilon] >= idx_bot) & (pre_idx3[:, ilat, ilon] < idx_top)
            pre_bool4 = (pre_idx4[:, ilat, ilon] >= idx_bot) & (pre_idx4[:, ilat, ilon] < idx_top)
            pre_bool5 = (pre_idx5[:, ilat, ilon] >= idx_bot) & (pre_idx5[:, ilat, ilon] < idx_top)

            df1 = np.sum(pre_bool1, axis=0)
            df2 = np.sum(pre_bool2, axis=0)
            df3 = np.sum(pre_bool3, axis=0)
            df4 = np.sum(pre_bool4, axis=0)
            df5 = np.sum(pre_bool5, axis=0)

            msc_ext1[ilat, ilon] = np.mean(msc1[pre_bool1, ilat, ilon])
            msc_ext2[ilat, ilon] = np.mean(msc2[pre_bool2, ilat, ilon])
            msc_ext3[ilat, ilon] = np.mean(msc3[pre_bool3, ilat, ilon])
            msc_ext4[ilat, ilon] = np.mean(msc4[pre_bool4, ilat, ilon])
            msc_ext5[ilat, ilon] = np.mean(msc5[pre_bool5, ilat, ilon])

            q1_ext = q1[pre_bool1, :, ilat, ilon]
            q2_ext = q2[pre_bool2, :, ilat, ilon]
            q3_ext = q3[pre_bool3, :, ilat, ilon]
            q4_ext = q4[pre_bool4, :, ilat, ilon]
            q5_ext = q5[pre_bool5, :, ilat, ilon]

            convu1_ext = convu1[pre_bool1, :, ilat, ilon]
            convu2_ext = convu2[pre_bool2, :, ilat, ilon]
            convu3_ext = convu3[pre_bool3, :, ilat, ilon]
            convu4_ext = convu4[pre_bool4, :, ilat, ilon]
            convu5_ext = convu5[pre_bool5, :, ilat, ilon]

            dlevs1_ext = dlevs1[pre_bool1, :, ilat, ilon]
            dlevs2_ext = dlevs2[pre_bool2, :, ilat, ilon]
            dlevs3_ext = dlevs3[pre_bool3, :, ilat, ilon]
            dlevs4_ext = dlevs4[pre_bool4, :, ilat, ilon]
            dlevs5_ext = dlevs5[pre_bool5, :, ilat, ilon]

            ########################################
            # Calculate all aerosol response
            # temp = q2[pre_bool2, :, ilat, ilon]
            # print(temp.shape)

            res_allaero_all[ilat, ilon], res_allaero_tt1[ilat, ilon] = getstats_mean(
                (msc2[pre_bool2, ilat, ilon] - msc5[pre_bool5, ilat, ilon])/msc5[pre_bool5, ilat, ilon])
            res_allaero_dyn[ilat, ilon], res_allaero_tt2[ilat, ilon] = getstats_mean(
                np.sum(q5_ext * (convu2_ext - convu5_ext) * dlevs5_ext, axis=1)/oro_water/g/msc5[pre_bool5, ilat, ilon])
            res_allaero_thm[ilat, ilon], res_allaero_tt3[ilat, ilon] = getstats_mean(
                np.sum((q2_ext - q5_ext) * convu5_ext * dlevs5_ext, axis=1)/oro_water/g/msc5[pre_bool5, ilat, ilon])

            res_allaero_all[ilat, ilon] = res_allaero_all[ilat, ilon] * 100
            res_allaero_dyn[ilat, ilon] = res_allaero_dyn[ilat, ilon] * 100 * 86400 * 1000
            res_allaero_thm[ilat, ilon] = res_allaero_thm[ilat, ilon] * 100 * 86400 * 1000

            ########################################
            # Calculate aerosol fast response
            temp1 = (msc2[pre_bool2, ilat, ilon] - msc4[pre_bool4, ilat, ilon])/msc4[pre_bool4, ilat, ilon]
            temp2 = (msc3[pre_bool3, ilat, ilon] - msc5[pre_bool5, ilat, ilon])/msc5[pre_bool5, ilat, ilon]
            res_fastaero_all[ilat, ilon], res_fastaero_tt1[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            temp1 = np.sum(q4_ext * (convu2_ext - convu4_ext) * dlevs4_ext,
                           axis=1)/oro_water/g/msc4[pre_bool4, ilat, ilon]
            temp2 = np.sum(q5_ext * (convu3_ext - convu5_ext) * dlevs5_ext,
                           axis=1)/oro_water/g/msc5[pre_bool4, ilat, ilon]
            res_fastaero_dyn[ilat, ilon], res_fastaero_tt2[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            temp1 = np.sum((q2_ext - q4_ext) * convu4_ext * dlevs4_ext, axis=1)/oro_water/g/msc4[pre_bool4, ilat, ilon]
            temp2 = np.sum((q3_ext - q5_ext) * convu5_ext * dlevs5_ext, axis=1)/oro_water/g/msc5[pre_bool4, ilat, ilon]
            res_fastaero_thm[ilat, ilon], res_fastaero_tt3[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            res_fastaero_all[ilat, ilon] = res_fastaero_all[ilat, ilon] * 100
            res_fastaero_dyn[ilat, ilon] = res_fastaero_dyn[ilat, ilon] * 100 * 86400 * 1000
            res_fastaero_thm[ilat, ilon] = res_fastaero_thm[ilat, ilon] * 100 * 86400 * 1000

            ########################################
            # Calculate aerosol slow response
            temp1 = (msc2[pre_bool2, ilat, ilon] - msc3[pre_bool3, ilat, ilon])/msc3[pre_bool3, ilat, ilon]
            temp2 = (msc4[pre_bool4, ilat, ilon] - msc5[pre_bool5, ilat, ilon])/msc5[pre_bool5, ilat, ilon]
            res_slowaero_all[ilat, ilon], res_slowaero_tt1[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            temp1 = np.sum(q3_ext * (convu2_ext - convu3_ext) * dlevs3_ext,
                           axis=1)/oro_water/g/msc3[pre_bool3, ilat, ilon]
            temp2 = np.sum(q5_ext * (convu4_ext - convu5_ext) * dlevs5_ext,
                           axis=1)/oro_water/g/msc5[pre_bool3, ilat, ilon]
            res_slowaero_dyn[ilat, ilon], res_slowaero_tt2[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            temp1 = np.sum((q2_ext - q3_ext) * convu3_ext * dlevs3_ext, axis=1)/oro_water/g/msc3[pre_bool3, ilat, ilon]
            temp2 = np.sum((q4_ext - q5_ext) * convu5_ext * dlevs5_ext, axis=1)/oro_water/g/msc5[pre_bool3, ilat, ilon]
            res_slowaero_thm[ilat, ilon], res_slowaero_tt3[ilat, ilon] = getstats_mean(np.append(temp1, temp2))

            res_slowaero_all[ilat, ilon] = res_slowaero_all[ilat, ilon] * 100
            res_slowaero_dyn[ilat, ilon] = res_slowaero_dyn[ilat, ilon] * 100 * 86400 * 1000
            res_slowaero_thm[ilat, ilon] = res_slowaero_thm[ilat, ilon] * 100 * 86400 * 1000

            ########################################
            # GHG and natural forcing
            res_GHG_all[ilat, ilon], res_GHG_tt1[ilat, ilon] = getstats_mean(
                (msc5[pre_bool5, ilat, ilon] - msc1[pre_bool1, ilat, ilon])/msc1[pre_bool1, ilat, ilon])
            res_GHG_dyn[ilat, ilon], res_GHG_tt2[ilat, ilon] = getstats_mean(
                np.sum(q1_ext * (convu5_ext - convu1_ext) * dlevs1_ext, axis=1)/oro_water/g/msc1[pre_bool1, ilat, ilon])
            res_GHG_thm[ilat, ilon], res_GHG_tt3[ilat, ilon] = getstats_mean(
                np.sum((q5_ext - q1_ext) * convu1_ext * dlevs1_ext, axis=1)/oro_water/g/msc1[pre_bool1, ilat, ilon])

            res_GHG_all[ilat, ilon] = res_GHG_all[ilat, ilon] * 100
            res_GHG_dyn[ilat, ilon] = res_GHG_dyn[ilat, ilon] * 100 * 86400 * 1000
            res_GHG_thm[ilat, ilon] = res_GHG_thm[ilat, ilon] * 100 * 86400 * 1000

            ########################################
            # All forcings
            res_all_all[ilat, ilon], res_all_tt1[ilat, ilon] = getstats_mean(
                (msc2[pre_bool2, ilat, ilon] - msc1[pre_bool1, ilat, ilon])/msc1[pre_bool1, ilat, ilon])
            res_all_dyn[ilat, ilon], res_all_tt2[ilat, ilon] = getstats_mean(
                np.sum(q1_ext * (convu2_ext - convu1_ext) * dlevs1_ext, axis=1)/oro_water/g/msc1[pre_bool1, ilat, ilon])
            res_all_thm[ilat, ilon], res_all_tt3[ilat, ilon] = getstats_mean(
                np.sum((q2_ext - q1_ext) * convu1_ext * dlevs1_ext, axis=1)/oro_water/g/msc1[pre_bool1, ilat, ilon])

            res_all_all[ilat, ilon] = res_all_all[ilat, ilon] * 100
            res_all_dyn[ilat, ilon] = res_all_dyn[ilat, ilon] * 100 * 86400 * 1000
            res_all_thm[ilat, ilon] = res_all_thm[ilat, ilon] * 100 * 86400 * 1000

    labels = ['Total Response', 'Dynamic Response', 'Thermodynamic Response']
    clevs = np.arange(-75, 75.1, 15)
    var_unit = '%'

    ########################################
    # all aerosol response
    forcingstr = "All aerosol forcings"
    forcingfname = "allaerosols"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_percent_decompose_"+forcingfname
    plotdiff(lons, lats, res_allaero_all, res_allaero_tt1, res_allaero_dyn, res_allaero_tt2,
             res_allaero_thm, res_allaero_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_allaero_all, res_allaero_tt1, res_allaero_dyn, res_allaero_tt2,
             res_allaero_thm, res_allaero_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # aerosol fast response
    forcingstr = "Aerosol fast response"
    forcingfname = "fastaerosol"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_percent_decompose_"+forcingfname
    plotdiff(lons, lats, res_fastaero_all, res_fastaero_tt1, res_fastaero_dyn, res_fastaero_tt2,
             res_fastaero_thm, res_fastaero_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_fastaero_all, res_fastaero_tt1, res_fastaero_dyn, res_fastaero_tt2,
             res_fastaero_thm, res_fastaero_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # aerosol slow response
    forcingstr = "Aerosol slow response"
    forcingfname = "slowaerosol"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_percent_decompose_"+forcingfname
    plotdiff(lons, lats, res_slowaero_all, res_slowaero_tt1, res_slowaero_dyn, res_slowaero_tt2,
             res_slowaero_thm, res_slowaero_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_slowaero_all, res_slowaero_tt1, res_slowaero_dyn, res_slowaero_tt2,
             res_slowaero_thm, res_slowaero_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # GHG and natural forcing
    forcingstr = "GHG and natural forcings"
    forcingfname = "GHGforcings"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_percent_decompose_"+forcingfname
    plotdiff(lons, lats, res_GHG_all, res_GHG_tt1, res_GHG_dyn, res_GHG_tt2,
             res_GHG_thm, res_GHG_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_GHG_all, res_GHG_tt1, res_GHG_dyn, res_GHG_tt2,
             res_GHG_thm, res_GHG_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # All forcings
    forcingstr = "All forcings"
    forcingfname = "allforcings"
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_percent_decompose_"+forcingfname
    plotdiff(lons, lats, res_all_all, res_all_tt1, res_all_dyn, res_all_tt2,
             res_all_thm, res_all_tt3, labels, clevs, var_unit, forcingstr, fname, 0)
    plotdiff(lons, lats, res_all_all, res_all_tt1, res_all_dyn, res_all_tt2,
             res_all_thm, res_all_tt3, labels, clevs, var_unit, forcingstr, fname, 1)

    ########################################
    # All forcings
    res = [res_allaero_all, res_fastaero_all, res_slowaero_all,
           res_allaero_thm, res_fastaero_thm, res_slowaero_thm,
           res_allaero_dyn, res_fastaero_dyn, res_slowaero_dyn]

    tt = [res_allaero_tt1, res_fastaero_tt1, res_slowaero_tt1,
          res_allaero_tt2, res_fastaero_tt2, res_slowaero_tt2,
          res_allaero_tt3, res_fastaero_tt3, res_slowaero_tt3]

    title = 'Relative Changes in the product of humidity and mass convergence to aerosol forcing'
    fname = outdir_percent+"vrseasia_aerosol_amip_jja_" + \
        str(percentile)+"th_extreme_"+varfname+"_response_SEA_contour_dyn_thm_percent_decompose_aerosolsinone"
    plotalldiff(lons, lats, res, tt, clevs, var_unit, title, fname, 0)
    plotalldiff(lons, lats, res, tt, clevs, var_unit, title, fname, 1)


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
