# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import math as math
import pandas as pd
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import matplotlib.cm as cm
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
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

expdir1 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case1+"/atm/"
expdir2 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case2+"/atm/"
expdir3 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case3+"/atm/"
expdir4 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case4+"/atm/"
expdir5 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case5+"/atm/"

# set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/mfc/"

# set up variable names and file name
varname = 'MFC'
varfname = "mfc"
varstr = "Moisture Flux Convergence"
var_res = "fv09"
var_unit = 'mm/day'

# define inital year and end year
iniyear = 2
endyear = 50

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# latbounds = [ -40 , 40 ]
# lonbounds = [ 10 , 160 ]

asia_latbounds = [-20, 50]
asia_lonbounds = [60, 140]

# define top layer
ptop = 200

# contants
oro_water = 997
g = 9.8
r_earth = 6371000


# define regions
reg_names = ['mainland SEA', 'Central India']
reg_str = ['mainSEA', 'ctInd']
reg_lats = [[10, 20], [16.5, 26.5]]
reg_lons = [[100, 110], [74.5, 86.5]]

# reg_names = ['mainland SEA', 'South Asia']
# reg_str = ['mainSEA', 'SA']
# reg_lats = [[10, 20], [10, 35]]
# reg_lons = [[100, 110], [70, 90]]

# reg_names = ['South India', 'North India']
# reg_str = ['stInd', 'nrInd']
# reg_lats = [[8, 20], [20, 28]]
# reg_lons = [[70, 90], [65, 90]]

################################################################################################
# S0-Define functions
################################################################################################
# calculate difference of mean and significance level


def getstats_diff(var1, var2):
    n1 = var1.shape[0]
    n2 = var2.shape[0]

    var1mean = np.mean(var1, axis=0)
    var2mean = np.mean(var2, axis=0)
    var1std = np.std(var1, axis=0)
    var2std = np.std(var2, axis=0)

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

# calculate seasonal mean


def season_mean(var):
    varseasts = np.zeros(((endyear-iniyear+1), latui-latli+1, lonui-lonli+1))
    for iyear in range(endyear-iniyear+1):
        varseasts[iyear, :, :] = np.mean(var[iyear*12+5:iyear*12+8, :, :], axis=0)

    return np.mean(varseasts, axis=0)

# calculate seasonal mean


def season_ts(var):
    if (len(var.shape) == 3):
        varseasts = np.zeros(((endyear-iniyear+1), var.shape[1], var.shape[2]))
        for iyear in range(endyear-iniyear+1):
            varseasts[iyear, :, :] = np.mean(var[iyear*12+5:iyear*12+8, :, :], axis=0)
    if (len(var.shape) == 4):
        varseasts = np.zeros(((endyear-iniyear+1), var.shape[1], var.shape[2], var.shape[3]))
        for iyear in range(endyear-iniyear+1):
            varseasts[iyear, :, :, :] = np.mean(var[iyear*12+5:iyear*12+8, :, :, :], axis=0)

    return varseasts

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
#    for itime in range(ntime):
#        for ilat in range(nlat):
#            for ilon in range(nlon):
#                levli = np.abs(levs - ptop).argmin()
#                levui = np.abs(levs - ps[itime,ilat,ilon]).argmin()
#                layer_thickness[itime, levli:levui+1, ilat, ilon] = dlevs[levli:levui+1]

    return layer_thickness

# calculat divergence


def getdiv1(lats, lons, u, v):
    dlats = lats[latli:latui+1]
    dlons = lons[lonli:lonui+1]
#    print(dlats)

    dtemp = np.zeros((len(dlats), len(dlons)))
    for ilat in range(len(dlats)):
        dtemp[ilat, :] = v[ilat, :]/r_earth * np.tan(np.deg2rad(dlats[ilat]))
    diverge = np.gradient(u, dlons, axis=1)/np.pi*180/r_earth + np.gradient(v, dlats, axis=0)/np.pi*180/r_earth - dtemp

    return diverge


def getdiv2(lats, lons, u, v):

    dlats = r_earth * np.deg2rad(np.gradient(lats[latli:latui+1]))
    dlons = r_earth * np.deg2rad(np.gradient(lons[lonli:lonui+1]))

#    dlats = lats[latli:latui+1]
#    dlons = lons[lonli:lonui+1]
#    print(dlon)

#    print(dlons[1])

#    print(u)
#    print(v)
#    print(np.gradient(u,dlon,axis=1))

#    print(u[0,:])
#    print(len(dlats))
#    dtemp = np.zeros((len(dlat),len(dlon)))
#    for ilat in range(len(dlat)):
#        dtemp[ilat,:] = v[ilat,:]/r_earth * np.tan(np.deg2rad(dlat[ilat]))
#    diverge = np.gradient(u,dlon,axis=1)/np.pi*180/r_earth + np.gradient(v,dlat,axis=0)/np.pi*180/r_earth - dtemp
#    print(str(np.gradient(u,dlons,axis=1)[0,1]))

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


# vertical integration
def getvint(lats, lons, levs, var, ps, ptop):
    layer_thickness = dpres_plevel(levs, ps, ptop)
    var_int = np.sum(var*layer_thickness, axis=1)/oro_water/g

    return var_int


# plot for climatology
def plotclim(lons, lats, var, titlestr, fname):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6)
    mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
    x, y = map(mlons, mlats)
    clevs = np.arange(-15., 15.1, 1.)
    cs = map.contourf(x, y, var, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")

    # add colorbar.
    cbar = map.colorbar(cs, location='bottom', pad="5%")
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+' ['+var_unit+']')

    # add title
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_"+fname+".png", dpi=600)
    plt.title(titlestr+" JJA "+varstr, fontsize=11, y=1.08)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_"+fname+".pdf")

    plt.close(fig)


# plot for differences
def plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo, tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, opt):
    fig = plt.figure()

    # P-E changes
    ax1 = fig.add_subplot(221)
    ax1.set_title('P-E response, mean='+str(round(np.mean(res_PE), 3))+var_unit, fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
    x, y = map(mlons, mlats)
    clevs = np.arange(-1.4, 1.5, 0.2)
    cs = map.contourf(x, y, res_PE, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt_PE.max()]
        csm = plt.contourf(x, y, tt_PE, levels=levels, colors='none', hatches=["", "....."], alpha=0)
#    print(tt_total)

    # total response
    ax2 = fig.add_subplot(222)
    ax2.set_title('Total response, mean='+str(round(np.mean(res_total), 3))+var_unit, fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
    x, y = map(mlons, mlats)
    clevs = np.arange(-1.4, 1.5, 0.2)
    cs = map.contourf(x, y, res_total, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt_total.max()]
        csm = plt.contourf(x, y, tt_total, levels=levels, colors='none', hatches=["", "....."], alpha=0)
#    print(tt_total)

    # thermodynamic response
    ax3 = fig.add_subplot(223)
    ax3.set_title('Thermodynamic response, mean='+str(round(np.mean(res_thermo), 3))+var_unit, fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    clevs = np.arange(-1.4, 1.5, 0.2)
    cs = map.contourf(x, y, res_thermo, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt_thermo.max()]
        csm = plt.contourf(x, y, tt_thermo, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    # dynamic response
    ax4 = fig.add_subplot(224)
    ax4.set_title('Dynamic response, mean='+str(round(np.mean(res_total), 3))+var_unit, fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    clevs = np.arange(-1.4, 1.5, 0.2)
    cs = map.contourf(x, y, res_dyn, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt_dyn.max()]
        csm = plt.contourf(x, y, tt_dyn, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    fig.subplots_adjust(bottom=0.2, wspace=0.15, hspace=0.1)
    cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
#    cbar = fig.colorbar(cs,orientation='horizontal',fraction=0.15, aspect= 25,shrink = 0.8)
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('['+var_unit+']', fontsize=6, labelpad=-0.3)

    # add title
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_response_with_siglev_" +
                    forcingfname+".png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_response_" +
                    forcingfname+".png", dpi=600, bbox_inches='tight')
    plt.suptitle(forcingstr+" "+varstr+" changes", fontsize=7, y=0.95)
#    plt.tight_layout()
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_with_siglev_"+forcingfname+".pdf", bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_"+forcingfname+".pdf", bbox_inches='tight')

    plt.close(fig)


# plot for eddy decomposition
def ploteddy(lons, lats, prect, evap, mfc, titlestr, fname):
    fig = plt.figure()
    # total precipitation
    ax1 = fig.add_subplot(221)
    ax1.set_title('a) Total precipitation', fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
    x, y = map(mlons, mlats)
    clevs = np.arange(-15., 15.1, 1.)
    cs = map.contourf(x, y, prect, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")

    # evaporation
    ax2 = fig.add_subplot(222)
    ax2.set_title('b) Evaporation', fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    clevs = np.arange(-15., 15.1, 1.)
    cs = map.contourf(x, y, evap, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")

    # mfc
    ax3 = fig.add_subplot(223)
    ax3.set_title('c) Moisture flux convergence', fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    clevs = np.arange(-15., 15.1, 1.)
    cs = map.contourf(x, y, mfc, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")

    # eddy terms
    ax4 = fig.add_subplot(224)
    ax4.set_title('d) Residuals', fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    clevs = np.arange(-15., 15.1, 1.)
    cs = map.contourf(x, y, prect-evap-mfc, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")

    fig.subplots_adjust(bottom=0.2, wspace=0.15, hspace=0.1)
    cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
#    cbar = fig.colorbar(cs,orientation='horizontal',fraction=0.15, aspect= 25,shrink = 0.8)
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label(varstr+' ['+var_unit+']', fontsize=6, labelpad=-0.3)

    # add title
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_" +
                fname+"_eddy_decompose.png", dpi=600, bbox_inches='tight')
    plt.suptitle(titlestr+" JJA "+varstr+" and eddies", fontsize=11, y=0.95)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_contour_" +
                fname+"_eddy_decompose.pdf", bbox_inches='tight')
    plt.close(fig)


################################################################################################
# plot for differences
def plotalldiff(lons, lats, res1_total, tt1_total, res1_thermo, tt1_thermo, res1_dyn, tt1_dyn, res2_total, tt2_total, res2_thermo, tt2_thermo, res2_dyn, tt2_dyn, res3_total, tt3_total, res3_thermo, tt3_thermo, res3_dyn, tt3_dyn, opt):
    fig = plt.figure()
    # total response,fast+slow
    ax1 = fig.add_subplot(331)
#    ax1.set_title(r'$\Delta_{total} MFC$, regional mean='+str(round(np.mean(res1_total), 4))+var_unit,fontsize=5,pad=3)
    ax1.set_title('Total response', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(lons[lonli:lonui+1], lats[latli:latui+1])
    x, y = map(mlons, mlats)
    clevs = np.arange(-1.4, 1.5, 0.2)
    cs = map.contourf(x, y, res1_total, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt1_total.max()]
        csm = ax1.contourf(x, y, tt1_total, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax1.xaxis.set_tick_params(labelsize=4)
    ax1.yaxis.set_tick_params(labelsize=4)
    ax1.set_ylabel(r'$\delta MC$', fontsize=7, labelpad=12)

    # total response,fast
    ax2 = fig.add_subplot(332)
#    ax2.set_title(r'$\Delta_{fast} MFC$, regional mean='+str(round(np.mean(res2_total), 4))+var_unit,fontsize=5,pad=3)
    ax2.set_title('Atmospheric-forced', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res2_total, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt2_total.max()]
        csm = ax2.contourf(x, y, tt2_total, levels=levels, colors='none', hatches=["", "....."], alpha=0)

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
    cs = map.contourf(x, y, res3_total, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt3_total.max()]
        csm = ax3.contourf(x, y, tt3_total, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax3.xaxis.set_tick_params(labelsize=4)
    ax3.yaxis.set_tick_params(labelsize=4)

    # thermodynamic response,fast+slow
    ax4 = fig.add_subplot(334)
#    ax4.set_title(r'$\Delta_{total,thermo} MFC$, regional mean='+str(round(np.mean(res1_thermo), 4))+var_unit,fontsize=5,pad=3)
    # ax4.set_title(r'$\Delta_{total,thermo} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res1_thermo, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt1_thermo.max()]
        csm = ax4.contourf(x, y, tt1_thermo, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax4.xaxis.set_tick_params(labelsize=4)
    ax4.yaxis.set_tick_params(labelsize=4)
    ax4.set_ylabel(r'$\delta TH$', fontsize=7, labelpad=12)

    # thermodynamic response,fast
    ax5 = fig.add_subplot(335)
#    ax5.set_title(r'$\Delta_{fast,thermo} MFC$, regional mean='+str(round(np.mean(res2_thermo), 4))+var_unit,fontsize=5,pad=3)
    # ax5.set_title(r'$\Delta_{fast,thermo} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res2_thermo, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt2_thermo.max()]
        csm = ax5.contourf(x, y, tt2_thermo, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax5.xaxis.set_tick_params(labelsize=4)
    ax5.yaxis.set_tick_params(labelsize=4)

    # thermodynamic response,slow
    ax6 = fig.add_subplot(336)
#    ax6.set_title(r'$\Delta_{slow,thermo} MFC$, regional mean='+str(round(np.mean(res3_thermo), 4))+var_unit,fontsize=5,pad=3)
    # ax6.set_title(r'$\Delta_{slow,thermo} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res3_thermo, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt3_thermo.max()]
        csm = ax6.contourf(x, y, tt3_thermo, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax6.xaxis.set_tick_params(labelsize=4)
    ax6.yaxis.set_tick_params(labelsize=4)

    # dynamic response
    ax7 = fig.add_subplot(337)
#    ax7.set_title(r'$\Delta_{total,dynamics} MFC$, regional mean='+str(round(np.mean(res1_dyn), 5))+var_unit,fontsize=5,pad=3)
    # ax7.set_title(r'$\Delta_{total,dynamics} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res1_dyn, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt1_dyn.max()]
        csm = ax7.contourf(x, y, tt1_dyn, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax7.xaxis.set_tick_params(labelsize=4)
    ax7.yaxis.set_tick_params(labelsize=4)
    ax7.set_ylabel(r'$\delta DY$', fontsize=7, labelpad=12)

    # dynamic response
    ax8 = fig.add_subplot(338)
#    ax8.set_title(r'$\Delta_{fast,dynamics} MFC$, regional mean='+str(round(np.mean(res2_dyn), 5))+var_unit,fontsize=5,pad=3)
    # ax8.set_title(r'$\Delta_{fast,dynamics} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res2_dyn, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt2_dyn.max()]
        csm = ax8.contourf(x, y, tt2_dyn, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax8.xaxis.set_tick_params(labelsize=4)
    ax8.yaxis.set_tick_params(labelsize=4)

    # dynamic response
    ax9 = fig.add_subplot(339)
#    ax9.set_title(r'$\Delta_{slow,dynamics} MFC$, regional mean='+str(round(np.mean(res3_dyn), 5))+var_unit,fontsize=5,pad=3)
    # ax9.set_title(r'$\Delta_{slow,dynamics} MFC$', fontsize=7, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.contourf(x, y, res3_dyn, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt3_dyn.max()]
        csm = ax9.contourf(x, y, tt3_dyn, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    ax9.xaxis.set_tick_params(labelsize=4)
    ax9.yaxis.set_tick_params(labelsize=4)

    fig.subplots_adjust(bottom=0.2, wspace=0.15, hspace=0.03)
    cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('Changes in MFC ['+var_unit+']', fontsize=6, labelpad=0.1)

    # add title
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_with_siglev_aerosolsinone.png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_aerosolsinone.png", dpi=600, bbox_inches='tight')
    plt.suptitle("Aerosol Responses "+varstr+" changes", fontsize=7, y=0.95)
#    plt.tight_layout()
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_with_siglev_aerosolsinone.pdf", bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                    "_SEA_contour_response_aerosolsinone.pdf", bbox_inches='tight')

    plt.close(fig)


################################################################################################
# plot for the Covariance fraction

def plotcovbars(res_frac, reg_names):

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
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                "_SEA_contour_response_dyn_thm_decompose_conv_"+reg_str[0]+"and"+reg_str[1]+".png", dpi=600, bbox_inches='tight')
    plt.suptitle("Fraction of variance of each Moisture Convergence component to P-E field", fontsize=8, y=0.95)
#    plt.tight_layout()
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                "_SEA_contour_response_dyn_thm_decompose_conv_"+reg_str[0]+"and"+reg_str[1]+".pdf", bbox_inches='tight')

    plt.close(fig)

################################################################################################
# plot for the means bars


def plotmeanbars(res_mean, reg_names):

    index = np.arange(12)+1
    xticks = [r'$\delta (P-E)$', r'$\delta MC$', r'$\delta DY$', r'$\delta TH$',
              r'$\delta (P-E)$', r'$\delta MC$', r'$\delta DY$', r'$\delta TH$',
              r'$\delta (P-E)$', r'$\delta MC$', r'$\delta DY$', r'$\delta TH$', ]
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
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                "_SEA_contour_response_dyn_thm_decompose_mean_"+reg_str[0]+"and"+reg_str[1]+".png", dpi=600, bbox_inches='tight')
    plt.suptitle("Contribution of each Moisture Convergence component to P-E field", fontsize=8, y=0.95)
#    plt.tight_layout()
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname +
                "_SEA_contour_response_dyn_thm_decompose_mean_"+reg_str[0]+"and"+reg_str[1]+".pdf", bbox_inches='tight')

    plt.close(fig)


################################################################################################
# S1-read climatological data
################################################################################################
# read lats,lons,levs,
fname1 = var_res+"_Q_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fdata1 = Dataset(expdir1+fname1)
lats = fdata1.variables['lat'][:]
lons = fdata1.variables['lon'][:]
levs = fdata1.variables['lev'][:]

# latitude/longitude  lower and upper contour index
latli = np.abs(lats - latbounds[0]).argmin()
latui = np.abs(lats - latbounds[1]).argmin()

lonli = np.abs(lons - lonbounds[0]).argmin()
lonui = np.abs(lons - lonbounds[1]).argmin()

lattest = np.abs(lats[latli:latui+1] - 25).argmin()
lontest = np.abs(lons[lonli:lonui+1] - 98).argmin()
print('reading data...')

# read Q
fname1 = var_res+"_Q_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fname2 = var_res+"_Q_"+case2+".cam.h0.0001-0050_vertical_interp.nc"
fname3 = var_res+"_Q_"+case3+".cam.h0.0001-0050_vertical_interp.nc"
fname4 = var_res+"_Q_"+case4+".cam.h0.0001-0050_vertical_interp.nc"
fname5 = var_res+"_Q_"+case5+".cam.h0.0001-0050_vertical_interp.nc"

fdata1 = Dataset(expdir1+fname1)
fdata2 = Dataset(expdir2+fname2)
fdata3 = Dataset(expdir3+fname3)
fdata4 = Dataset(expdir4+fname4)
fdata5 = Dataset(expdir5+fname5)

# read the monthly data for a larger region
q1 = fdata1.variables['Q'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
q2 = fdata2.variables['Q'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
q3 = fdata3.variables['Q'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
q4 = fdata4.variables['Q'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
q5 = fdata5.variables['Q'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]

# read U wind
fname1 = var_res+"_U_WIND_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fname2 = var_res+"_U_WIND_"+case2+".cam.h0.0001-0050_vertical_interp.nc"
fname3 = var_res+"_U_WIND_"+case3+".cam.h0.0001-0050_vertical_interp.nc"
fname4 = var_res+"_U_WIND_"+case4+".cam.h0.0001-0050_vertical_interp.nc"
fname5 = var_res+"_U_WIND_"+case5+".cam.h0.0001-0050_vertical_interp.nc"

fdata1 = Dataset(expdir1+fname1)
fdata2 = Dataset(expdir2+fname2)
fdata3 = Dataset(expdir3+fname3)
fdata4 = Dataset(expdir4+fname4)
fdata5 = Dataset(expdir5+fname5)

# read the monthly data for a larger region
uwnd1 = fdata1.variables['U'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
uwnd2 = fdata2.variables['U'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
uwnd3 = fdata3.variables['U'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
uwnd4 = fdata4.variables['U'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
uwnd5 = fdata5.variables['U'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]

# read V wind
fname1 = var_res+"_V_WIND_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fname2 = var_res+"_V_WIND_"+case2+".cam.h0.0001-0050_vertical_interp.nc"
fname3 = var_res+"_V_WIND_"+case3+".cam.h0.0001-0050_vertical_interp.nc"
fname4 = var_res+"_V_WIND_"+case4+".cam.h0.0001-0050_vertical_interp.nc"
fname5 = var_res+"_V_WIND_"+case5+".cam.h0.0001-0050_vertical_interp.nc"

fdata1 = Dataset(expdir1+fname1)
fdata2 = Dataset(expdir2+fname2)
fdata3 = Dataset(expdir3+fname3)
fdata4 = Dataset(expdir4+fname4)
fdata5 = Dataset(expdir5+fname5)

# read the monthly data for a larger region
vwnd1 = fdata1.variables['V'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
vwnd2 = fdata2.variables['V'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
vwnd3 = fdata3.variables['V'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
vwnd4 = fdata4.variables['V'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
vwnd5 = fdata5.variables['V'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]

# read PS
fname1 = var_res+"_PS_"+case1+".cam.h0.0001-0050.nc"
fname2 = var_res+"_PS_"+case2+".cam.h0.0001-0050.nc"
fname3 = var_res+"_PS_"+case3+".cam.h0.0001-0050.nc"
fname4 = var_res+"_PS_"+case4+".cam.h0.0001-0050.nc"
fname5 = var_res+"_PS_"+case5+".cam.h0.0001-0050.nc"

fdata1 = Dataset(expdir1+fname1)
fdata2 = Dataset(expdir2+fname2)
fdata3 = Dataset(expdir3+fname3)
fdata4 = Dataset(expdir4+fname4)
fdata5 = Dataset(expdir5+fname5)

# read the monthly data for a larger region
ps1 = fdata1.variables['PS'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
ps2 = fdata2.variables['PS'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
ps3 = fdata3.variables['PS'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
ps4 = fdata4.variables['PS'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
ps5 = fdata5.variables['PS'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]

ps1 = ps1/100
ps2 = ps2/100
ps3 = ps3/100
ps4 = ps4/100
ps5 = ps5/100

print(ps1[5, lattest, lontest])

# read PRECT
fname1 = var_res+"_PREC_"+case1+".cam.h0.0001-0050.nc"
fname2 = var_res+"_PREC_"+case2+".cam.h0.0001-0050.nc"
fname3 = var_res+"_PREC_"+case3+".cam.h0.0001-0050.nc"
fname4 = var_res+"_PREC_"+case4+".cam.h0.0001-0050.nc"
fname5 = var_res+"_PREC_"+case5+".cam.h0.0001-0050.nc"

fdata1 = Dataset(expdir1+fname1)
fdata2 = Dataset(expdir2+fname2)
fdata3 = Dataset(expdir3+fname3)
fdata4 = Dataset(expdir4+fname4)
fdata5 = Dataset(expdir5+fname5)

# read the monthly data for a larger region
pre1 = fdata1.variables['PRECT'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
pre2 = fdata2.variables['PRECT'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
pre3 = fdata3.variables['PRECT'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
pre4 = fdata4.variables['PRECT'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
pre5 = fdata5.variables['PRECT'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]


# read Evaporation
fname1 = var_res+"_EVAP_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fname2 = var_res+"_EVAP_"+case2+".cam.h0.0001-0050_vertical_interp.nc"
fname3 = var_res+"_EVAP_"+case3+".cam.h0.0001-0050_vertical_interp.nc"
fname4 = var_res+"_EVAP_"+case4+".cam.h0.0001-0050_vertical_interp.nc"
fname5 = var_res+"_EVAP_"+case5+".cam.h0.0001-0050_vertical_interp.nc"

fdata1 = Dataset(expdir1+fname1)
fdata2 = Dataset(expdir2+fname2)
fdata3 = Dataset(expdir3+fname3)
fdata4 = Dataset(expdir4+fname4)
fdata5 = Dataset(expdir5+fname5)

# read the monthly data for a larger region
evap1 = fdata1.variables['EVAPPREC'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap2 = fdata2.variables['EVAPPREC'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap3 = fdata3.variables['EVAPPREC'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap4 = fdata4.variables['EVAPPREC'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap5 = fdata5.variables['EVAPPREC'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]

evap1 = evap1 + fdata1.variables['EVAPQCM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap2 = evap2 + fdata2.variables['EVAPQCM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap3 = evap3 + fdata3.variables['EVAPQCM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap4 = evap4 + fdata4.variables['EVAPQCM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap5 = evap5 + fdata5.variables['EVAPQCM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]

evap1 = evap1 + fdata1.variables['EVAPQZM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap2 = evap2 + fdata2.variables['EVAPQZM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap3 = evap3 + fdata3.variables['EVAPQZM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap4 = evap4 + fdata4.variables['EVAPQZM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]
evap5 = evap5 + fdata5.variables['EVAPQZM'][(iniyear-1)*12: (endyear)*12, :, latli:latui+1, lonli:lonui+1]


# read QFLX
fname1 = var_res+"_QFLX_"+case1+".cam.h0.0001-0050.nc"
fname2 = var_res+"_QFLX_"+case2+".cam.h0.0001-0050.nc"
fname3 = var_res+"_QFLX_"+case3+".cam.h0.0001-0050.nc"
fname4 = var_res+"_QFLX_"+case4+".cam.h0.0001-0050.nc"
fname5 = var_res+"_QFLX_"+case5+".cam.h0.0001-0050.nc"

fdata1 = Dataset(expdir1+fname1)
fdata2 = Dataset(expdir2+fname2)
fdata3 = Dataset(expdir3+fname3)
fdata4 = Dataset(expdir4+fname4)
fdata5 = Dataset(expdir5+fname5)

# read the monthly data for a larger region
qflx1 = fdata1.variables['QFLX'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
qflx2 = fdata2.variables['QFLX'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
qflx3 = fdata3.variables['QFLX'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
qflx4 = fdata4.variables['QFLX'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]
qflx5 = fdata5.variables['QFLX'][(iniyear-1)*12: (endyear)*12, latli:latui+1, lonli:lonui+1]


print('finished reading...')

uwnd1 = season_ts(uwnd1)
uwnd2 = season_ts(uwnd2)
uwnd3 = season_ts(uwnd3)
uwnd4 = season_ts(uwnd4)
uwnd5 = season_ts(uwnd5)

vwnd1 = season_ts(vwnd1)
vwnd2 = season_ts(vwnd2)
vwnd3 = season_ts(vwnd3)
vwnd4 = season_ts(vwnd4)
vwnd5 = season_ts(vwnd5)

q1 = season_ts(q1)
q2 = season_ts(q2)
q3 = season_ts(q3)
q4 = season_ts(q4)
q5 = season_ts(q5)

ps1 = season_ts(ps1)
ps2 = season_ts(ps2)
ps3 = season_ts(ps3)
ps4 = season_ts(ps4)
ps5 = season_ts(ps5)

pre1 = season_ts(pre1) * 86400 * 1000
pre2 = season_ts(pre2) * 86400 * 1000
pre3 = season_ts(pre3) * 86400 * 1000
pre4 = season_ts(pre4) * 86400 * 1000
pre5 = season_ts(pre5) * 86400 * 1000

evap1 = season_ts(evap1)
evap2 = season_ts(evap2)
evap3 = season_ts(evap3)
evap4 = season_ts(evap4)
evap5 = season_ts(evap5)

qflx1 = season_ts(qflx1)/oro_water * 86400 * 1000
qflx2 = season_ts(qflx2)/oro_water * 86400 * 1000
qflx3 = season_ts(qflx3)/oro_water * 86400 * 1000
qflx4 = season_ts(qflx4)/oro_water * 86400 * 1000
qflx5 = season_ts(qflx5)/oro_water * 86400 * 1000


print('calculating mfc...')
# calculate mfc
mfc1 = getmfc(lats, lons, levs, uwnd1, vwnd1, q1, ps1, ptop)
mfc2 = getmfc(lats, lons, levs, uwnd2, vwnd2, q2, ps2, ptop)
mfc3 = getmfc(lats, lons, levs, uwnd3, vwnd3, q3, ps3, ptop)
mfc4 = getmfc(lats, lons, levs, uwnd4, vwnd4, q4, ps4, ptop)
mfc5 = getmfc(lats, lons, levs, uwnd5, vwnd5, q5, ps5, ptop)

print('calculating vertically integrated evaporation...')
# calculate vertically integrated evaporation
evap_int1 = getvint(lats, lons, levs, evap1, ps1, ptop) * 86400 * 1000  # convet from m/s to mm/day
evap_int2 = getvint(lats, lons, levs, evap2, ps2, ptop) * 86400 * 1000
evap_int3 = getvint(lats, lons, levs, evap3, ps3, ptop) * 86400 * 1000
evap_int4 = getvint(lats, lons, levs, evap4, ps4, ptop) * 86400 * 1000
evap_int5 = getvint(lats, lons, levs, evap5, ps5, ptop) * 86400 * 1000


#################################################################################################
# plot climatology
#################################################################################################
print('plotting climatology...')
fname = "case1"
titlestr = case1
plotclim(lons, lats, np.mean(mfc1 * 86400 * 1000, axis=0), titlestr, fname)

fname = "case2"
titlestr = case2
plotclim(lons, lats, np.mean(mfc2 * 86400 * 1000, axis=0), titlestr, fname)

fname = "case3"
titlestr = case3
plotclim(lons, lats, np.mean(mfc3 * 86400 * 1000, axis=0), titlestr, fname)

fname = "case4"
titlestr = case4
plotclim(lons, lats, np.mean(mfc4 * 86400 * 1000, axis=0), titlestr, fname)

fname = "case5"
titlestr = case5
plotclim(lons, lats, np.mean(mfc5 * 86400 * 1000, axis=0), titlestr, fname)

print('plotting eddy decomposition...')

fname = "case1"
titlestr = case1
ploteddy(lons, lats, np.mean(pre1, axis=0), np.mean(qflx1, axis=0),
         np.mean(mfc1 * 86400 * 1000, axis=0), titlestr, fname)

pre_res = np.mean(pre1, axis=0)
evp_res = np.mean(qflx1, axis=0)
mfc_res = np.mean(mfc1 * 86400 * 1000, axis=0)
diff_res = pre_res - evp_res

asia_latli = np.abs(lats[latli: latui+1] - asia_latbounds[0]).argmin()
asia_latui = np.abs(lats[latli: latui+1] - asia_latbounds[1]).argmin()

asia_lonli = np.abs(lons[lonli: lonui+1] - asia_lonbounds[0]).argmin()
asia_lonui = np.abs(lons[lonli: lonui+1] - asia_lonbounds[1]).argmin()


diff_res = diff_res[asia_latli: asia_latui+1, asia_lonli: asia_lonui+1]
mfc_res = mfc_res[asia_latli: asia_latui+1, asia_lonli: asia_lonui+1]
asia_cor = np.corrcoef(diff_res.flatten(), mfc_res.flatten())[0][1]
asia_mean_diff = np.mean(diff_res)
asia_mean_mfc = np.mean(mfc_res)
print('Spatial correlation between P-E and MFC over Asia in control runs: '+str(asia_cor))
print('Mean of P-E over Asia in control runs: '+str(asia_mean_diff))
print('Mean of MFC over Asia in control runs: '+str(asia_mean_mfc))


fname = "case2"
titlestr = case2
ploteddy(lons, lats, np.mean(pre2, axis=0), np.mean(qflx2, axis=0),
         np.mean(mfc2 * 86400 * 1000, axis=0), titlestr, fname)

fname = "case3"
titlestr = case3
ploteddy(lons, lats, np.mean(pre3, axis=0), np.mean(qflx3, axis=0),
         np.mean(mfc3 * 86400 * 1000, axis=0), titlestr, fname)

fname = "case4"
titlestr = case4
ploteddy(lons, lats, np.mean(pre4, axis=0), np.mean(qflx4, axis=0),
         np.mean(mfc4 * 86400 * 1000, axis=0), titlestr, fname)

fname = "case5"
titlestr = case5
ploteddy(lons, lats, np.mean(pre5, axis=0), np.mean(qflx5, axis=0),
         np.mean(mfc5 * 86400 * 1000, axis=0), titlestr, fname)


#################################################################################################
# plot for different forcings
#################################################################################################
print('plotting responses...')

# all aerosol foring
forcingstr = "All aerosol forcings"
forcingfname = "allaerosols"

res_PE, tt_PE = getstats_mean((pre2-qflx2) - (pre5-qflx5))
res_total, tt_total = getstats_mean((mfc2-mfc5) * 86400 * 1000)
res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd5, vwnd5, q2-q5, ps5, ptop) * 86400 * 1000)
res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd5, vwnd2-vwnd5, q5, ps5, ptop) * 86400 * 1000)

plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)


# aerosol fast response1
forcingstr = "Aerosol fast response"
forcingfname = "fastaerosol1"

res_PE, tt_PE = getstats_mean((pre2-qflx2) - (pre4-qflx4))
res_total, tt_total = getstats_mean((mfc2-mfc4) * 86400 * 1000)
res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd4, vwnd4, q2-q4, ps4, ptop) * 86400 * 1000)
res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd4, vwnd2-vwnd4, q4, ps4, ptop) * 86400 * 1000)

plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)

# aerosol slow response1
forcingstr = "Aerosol slow response"
forcingfname = "slowaerosol1"

res_PE, tt_PE = getstats_mean((pre4-qflx4) - (pre5-qflx5))
res_total, tt_total = getstats_mean((mfc4-mfc5) * 86400 * 1000)
res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd5, vwnd5, q4-q5, ps5, ptop) * 86400 * 1000)
res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd4-uwnd5, vwnd4-vwnd5, q5, ps5, ptop) * 86400 * 1000)

plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)

# aerosol fast response2
forcingstr = "Aerosol fast response"
forcingfname = "fastaerosol2"

res_PE, tt_PE = getstats_mean((pre3-qflx3) - (pre5-qflx5))
res_total, tt_total = getstats_mean((mfc3-mfc5) * 86400 * 1000)
res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd5, vwnd5, q3-q5, ps5, ptop) * 86400 * 1000)
res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd3-uwnd5, vwnd3-vwnd5, q5, ps5, ptop) * 86400 * 1000)

plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)

# aerosol slow response2
forcingstr = "Aerosol slow response"
forcingfname = "slowaerosol2"

res_PE, tt_PE = getstats_mean((pre2-qflx2) - (pre3-qflx3))
res_total, tt_total = getstats_mean((mfc2-mfc3) * 86400 * 1000)
res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd3, vwnd3, q2-q3, ps3, ptop) * 86400 * 1000)
res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd3, vwnd2-vwnd3, q3, ps3, ptop) * 86400 * 1000)

plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)

# GHG and natural forcing
forcingstr = "GHG and natural forcings"
forcingfname = "GHGforcings"

res_PE, tt_PE = getstats_mean((pre5-qflx5) - (pre1-qflx1))
res_total, tt_total = getstats_mean((mfc5-mfc1) * 86400 * 1000)
res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd1, vwnd1, q5-q1, ps1, ptop) * 86400 * 1000)
res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd5-uwnd1, vwnd5-vwnd1, q1, ps1, ptop) * 86400 * 1000)

plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)

# All forcings
forcingstr = "All forcings"
forcingfname = "allforcings"

res_PE, tt_PE = getstats_mean((pre2-qflx2) - (pre1-qflx1))
res_total, tt_total = getstats_mean((mfc2-mfc1) * 86400 * 1000)
res_thermo, tt_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd1, vwnd1, q2-q1, ps1, ptop) * 86400 * 1000)
res_dyn, tt_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd1, vwnd2-vwnd1, q1, ps1, ptop) * 86400 * 1000)

plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 0)
plotdiff(lons, lats, res_PE, tt_PE, res_total, tt_total, res_thermo,
         tt_thermo, res_dyn, tt_dyn, forcingstr, forcingfname, 1)
print(forcingstr+" mean response is: "+str(np.mean(res_total))+" "+var_unit)


# plot all in one
# total
res1_total, tt1_total = getstats_mean((mfc2-mfc5) * 86400 * 1000)
res1_thermo, tt1_thermo = getstats_mean(getmfc(lats, lons, levs, uwnd5, vwnd5, q2-q5, ps5, ptop) * 86400 * 1000)
res1_dyn, tt1_dyn = getstats_mean(getmfc(lats, lons, levs, uwnd2-uwnd5, vwnd2-vwnd5, q5, ps5, ptop) * 86400 * 1000)

# fast
res2_total, tt2_total = getstats_mean((mfc2 + mfc3 - mfc4 - mfc5)/2 * 86400 * 1000)
res2_thermo, tt2_thermo = getstats_mean(getmfc(lats, lons, levs, (uwnd4+uwnd5)/2,
                                               (vwnd4+vwnd5)/2, (q2+q3-q4-q5)/2, (ps4+ps5)/2, ptop) * 86400 * 1000)
res2_dyn, tt2_dyn = getstats_mean(getmfc(lats, lons, levs, (uwnd2+uwnd3-uwnd4-uwnd5)/2,
                                         (vwnd2+vwnd3-vwnd4-vwnd5)/2, (q4+q5)/2, (ps4+ps5)/2, ptop) * 86400 * 1000)

# slow
res3_total, tt3_total = getstats_mean((mfc4 + mfc2 - mfc5 - mfc3)/2 * 86400 * 1000)
res3_thermo, tt3_thermo = getstats_mean(getmfc(lats, lons, levs, (uwnd5+uwnd3)/2,
                                               (vwnd5+vwnd3)/2, (q4+q2-q5-q3)/2, (ps5+ps3)/2, ptop) * 86400 * 1000)
res3_dyn, tt3_dyn = getstats_mean(getmfc(lats, lons, levs, (uwnd4+uwnd2-uwnd5-uwnd3)/2,
                                         (vwnd4+vwnd2-vwnd5-vwnd3)/2, (q5+q3)/2, (ps5+ps3)/2, ptop) * 86400 * 1000)


plotalldiff(lons, lats, res1_total, tt1_total, res1_thermo, tt1_thermo, res1_dyn, tt1_dyn, res2_total, tt2_total,
            res2_thermo, tt2_thermo, res2_dyn, tt2_dyn, res3_total, tt3_total, res3_thermo, tt3_thermo, res3_dyn, tt3_dyn, 0)

plotalldiff(lons, lats, res1_total, tt1_total, res1_thermo, tt1_thermo, res1_dyn, tt1_dyn, res2_total, tt2_total,
            res2_thermo, tt2_thermo, res2_dyn, tt2_dyn, res3_total, tt3_total, res3_thermo, tt3_thermo, res3_dyn, tt3_dyn, 1)


#################################################################################################
# plot for the spatial covariance fraction
#################################################################################################

# pre_total = np.mean(pre2, axis=0) - np.mean(pre5, axis=0)
# pre_fast = np.mean(pre2, axis=0) + np.mean(pre3, axis=0) - np.mean(pre4, axis=0) - np.mean(pre5, axis=0)
# pre_slow = np.mean(pre2, axis=0) + np.mean(pre4, axis=0) - np.mean(pre3, axis=0) - np.mean(pre5, axis=0)
#
# evap_total = np.mean(qflx2, axis=0) - np.mean(qflx5, axis=0)
# evap_fast = np.mean(qflx2, axis=0) + np.mean(qflx3, axis=0) - np.mean(qflx4, axis=0) - np.mean(qflx5, axis=0)
# evap_slow = np.mean(qflx2, axis=0) + np.mean(qflx4, axis=0) - np.mean(qflx3, axis=0) - np.mean(qflx5, axis=0)

PE_total, PE_total_tt = getstats_mean((pre2-qflx2) - (pre5-qflx5))
PE_fast, PE_fast_tt = getstats_mean(((pre2-qflx2) + (pre3-qflx3) - (pre4-qflx4) - (pre5-qflx5))/2)
PE_slow, PE_slow_tt = getstats_mean(((pre2-qflx2) + (pre4-qflx4) - (pre3-qflx3) - (pre5-qflx5))/2)

blats = lats[latli:latui+1]
blons = lons[lonli:lonui+1]
res_frac = []
res_mean = []

for idx in range(len(reg_names)):
    reg_name = reg_names[idx]

    reg_latli = np.abs(blats - reg_lats[idx][0]).argmin()
    reg_latui = np.abs(blats - reg_lats[idx][1]).argmin()

    reg_lonli = np.abs(blons - reg_lons[idx][0]).argmin()
    reg_lonui = np.abs(blons - reg_lons[idx][1]).argmin()
    print(blats[reg_latli])
    print(blats[reg_latui])

    PE_var_total = np.var(PE_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
    PE_var_fast = np.var(PE_fast[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
    PE_var_slow = np.var(PE_slow[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])

    reg_cov_res1_tot = np.cov(PE_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                              res1_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
    reg_cov_res1_dyn = np.cov(PE_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                              res1_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
    reg_cov_res1_thm = np.cov(PE_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                              res1_thermo[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]

    reg_cov_res2_tot = np.cov(PE_fast[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                              res2_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
    reg_cov_res2_dyn = np.cov(PE_fast[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                              res2_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
    reg_cov_res2_thm = np.cov(PE_fast[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                              res2_thermo[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]

    reg_cov_res3_tot = np.cov(PE_slow[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                              res3_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
    reg_cov_res3_dyn = np.cov(PE_slow[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                              res3_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]
    reg_cov_res3_thm = np.cov(PE_slow[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten(),
                              res3_thermo[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1].flatten())[0][1]

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

    reg_cov_res1_tot = np.mean(res1_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
    reg_cov_res1_dyn = np.mean(res1_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
    reg_cov_res1_thm = np.mean(res1_thermo[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])

    reg_cov_res2_tot = np.mean(res2_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
    reg_cov_res2_dyn = np.mean(res2_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
    reg_cov_res2_thm = np.mean(res2_thermo[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])

    reg_cov_res3_tot = np.mean(res3_total[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
    reg_cov_res3_dyn = np.mean(res3_dyn[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])
    reg_cov_res3_thm = np.mean(res3_thermo[reg_latli: reg_latui + 1, reg_lonli: reg_lonui + 1])

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
plotcovbars(res_frac, reg_names)
plotmeanbars(res_mean, reg_names)
