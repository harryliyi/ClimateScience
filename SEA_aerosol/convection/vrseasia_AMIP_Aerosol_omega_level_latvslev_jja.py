# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

#import libraries
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
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/verticalcirculation/latvslev/"

# set up variable names and file name
varname = 'OMEGA'
varfname = "omega"
varstr = "Vertical velocity"
var_res = "fv09"
var_unit = r'$10^{-3}$ Pa/s'

# define inital year and end year
iniyear = 2
endyear = 50

# define the contour plot region
latbounds = [-40, 60]
lonbounds = [70, 120]

# define top layer
ptop = 100
levbounds = [150, 1000]

# contants
oro_water = 997
g = 9.8
r_earth = 6371000

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

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

########################################################################################
# plot for climatology


def plotclim_lat(lats, levs, var, tt, titlestr, fname, opt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xx, zz = np.meshgrid(lats, levs)
    clevs = np.arange(-80., 80.1, 10.)
    cs = ax1.contourf(xx, zz, var, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    if (opt == 1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx, zz, tt, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    ax1.set_ylabel("Pressure [hPa]")
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)
#    ax1.set_yscale('log')

    ax1.set_xlabel('Latitude [degrees]')

    # add colorbar.
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.15, aspect=25, shrink=0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']', fontsize=8)

    plt.gca().invert_yaxis()

    # add title
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_"+fname+".png")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_"+fname+".png")
    plt.title(titlestr+" JJA "+varstr, fontsize=11, y=1.08)
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_"+fname+".pdf")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_"+fname+".pdf")

    plt.close(fig)


# plot for climatology with vector
def plotclim_lat_vector(lats, levs, var, tt, uwnd, titlestr, fname, opt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xx, zz = np.meshgrid(lats, levs)
    clevs = np.arange(-80., 80.1, 10.)
    cs = ax1.contourf(xx, zz, var, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    cq = ax1.quiver(xx[::1, ::ratio], zz[::1, ::ratio], uwnd[::1, ::ratio], -var[::1, ::ratio], color='grey')
    if (opt == 1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx, zz, tt, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    ax1.set_ylabel("Pressure [hPa]")
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)
#    ax1.set_yscale('log')

    ax1.set_xlabel('Latitude [degrees]')

    # add colorbar.
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.15, aspect=25, shrink=0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']', fontsize=8)

    plt.gca().invert_yaxis()

    # add title
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_"+fname+".png")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_"+fname+".png")

    plt.title(titlestr+" JJA "+varstr, fontsize=11, y=1.08)
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_"+fname+".pdf")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_"+fname+".pdf")

    plt.close(fig)


########################################################################################
# plot for response
def plotdiff_lat(lats, levs, var, tt, clim, forcingstr, forcingfname, opt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xx, zz = np.meshgrid(lats, levs)
    clevs = np.arange(-3.5, 3.6, 0.5)
    cs = ax1.contourf(xx, zz, var, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    csc = ax1.contour(xx, zz, clim, levels=np.arange(-60., 60.1, 15.), linewidths=0.5, colors='k')
    ax1.clabel(csc, fontsize=5, inline=1)
    if (opt == 1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx, zz, tt, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    ax1.set_ylabel("Pressure [hPa]")
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)
#    ax1.set_yscale('log')

    ax1.set_xlabel('Latitude [degrees]')

    # add colorbar.
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.15, aspect=25, shrink=0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']', fontsize=8)

    plt.gca().invert_yaxis()

    # add title
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_"+forcingfname+".png")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_"+forcingfname+".png")

    plt.title(forcingstr+" "+varstr+" changes", fontsize=11, y=1.08)
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_"+forcingfname+".pdf")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_"+forcingfname+".pdf")

    plt.close(fig)


# plot for response with vector
def plotdiff_lat_vector(lats, levs, var, tt, uwnd, clim, forcingstr, forcingfname, opt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xx, zz = np.meshgrid(lats, levs)
    clevs = np.arange(-3.5, 3.6, 0.5)
    cs = ax1.contourf(xx, zz, var, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    csc = ax1.contour(xx, zz, clim, levels=np.arange(-60., 60.1, 15.), linewidths=0.5, colors='k')
    ax1.clabel(csc, fontsize=5, inline=1)
    cq = ax1.quiver(xx[::1, ::ratio], zz[::1, ::ratio], uwnd[::1, ::ratio], -var[::1, ::ratio], color='grey')
    if (opt == 1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx, zz, tt, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    ax1.set_ylabel("Pressure [hPa]")
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)
#    ax1.set_yscale('log')

    ax1.set_xlabel('Latitude [degrees]')

    # add colorbar.
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.15, aspect=25, shrink=0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']', fontsize=8)

    plt.gca().invert_yaxis()

    # add title
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_"+forcingfname+".png")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_"+forcingfname+".png")

    plt.title(forcingstr+" "+varstr+" changes", fontsize=11, y=1.08)
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_"+forcingfname+".pdf")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_"+forcingfname+".pdf")

    plt.close(fig)


########################################################################################
# plot for all responses together
def plotalldiff_lat(lats, levs, res1, tt1, clim1, res2, tt2, clim2, res4, tt4, clim4, opt):
    fig = plt.figure()
    xx, zz = np.meshgrid(lats, levs)
    clevs = np.arange(-3.5, 3.6, 0.5)

    # total response
    ax1 = fig.add_subplot(311)
    cs = ax1.contourf(xx, zz, res1, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    csc = ax1.contour(xx, zz, clim1, levels=np.arange(-60., 60.1, 15.), linewidths=0.5, colors='k')
    ax1.clabel(csc, fontsize=5, inline=1)
    if (opt == 1):
        levels = [0., 2.01, tt1.max()]
        csm = ax1.contourf(xx, zz, tt1, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    # ax1.set_title(r'$\Delta_{total} OMEGA$', fontsize=6, pad=2)
    ax1.set_title('a) Total response', fontsize=7, pad=2)
    ax1.set_ylabel("Pressure [hPa]", fontsize=7)
#    ax1.set_xlabel('Latitude [degrees]',fontsize=7)
#    ax1.xaxis.set_tick_params(labelsize=6)
    ax1.set_xticklabels([])
    ax1.yaxis.set_tick_params(labelsize=6)
    ax1.invert_yaxis()

    # fast response
    ax2 = fig.add_subplot(312)
    cs = ax2.contourf(xx, zz, res2, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    csc = ax2.contour(xx, zz, clim2, levels=np.arange(-60., 60.1, 15.), linewidths=0.5, colors='k')
    ax2.clabel(csc, fontsize=5, inline=1)
    if (opt == 1):
        levels = [0., 2.01, tt2.max()]
        csm = ax2.contourf(xx, zz, tt2, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    # ax2.set_title(r'$\Delta_{fast} OMEGA$', fontsize=6, pad=2)
    ax2.set_title('b) Atmospheric-forced', fontsize=7, pad=2)
    ax2.set_ylabel("Pressure [hPa]", fontsize=7)
#    ax2.set_xlabel('Latitude [degrees]',fontsize=7)
#    ax2.xaxis.set_tick_params(labelsize=6)
    ax2.set_xticklabels([])
    ax2.yaxis.set_tick_params(labelsize=6)
    ax2.invert_yaxis()

    # slow response
    ax4 = fig.add_subplot(313)
    cs = ax4.contourf(xx, zz, res4, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    csc = ax4.contour(xx, zz, clim4, levels=np.arange(-60., 60.1, 15.), linewidths=0.5, colors='k')
    ax4.clabel(csc, fontsize=5, inline=1)
    if (opt == 1):
        levels = [0., 2.01, tt4.max()]
        csm = ax4.contourf(xx, zz, tt4, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    # ax4.set_title(r'$\Delta_{slow} OMEGA$', fontsize=6, pad=2)
    ax4.set_title('c) Ocean-mediated', fontsize=7, pad=2)
    ax4.set_ylabel("Pressure [hPa]", fontsize=7)
    ax4.set_xlabel('Latitude [degrees]', fontsize=7)
    ax4.set_xticks(np.arange(-40, 61, 20))
    ax4.set_xticks(np.arange(-40, 61, 20))
    ax4.xaxis.set_tick_params(labelsize=6)
    ax4.yaxis.set_tick_params(labelsize=6)
    ax4.invert_yaxis()

    # add colorbar.
    fig.subplots_adjust(right=0.7, hspace=0.15)
    cbar_ax = fig.add_axes([0.72, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']', fontsize=6, labelpad=0.7)

    # add title
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_aerosolsinone.png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_aerosolsinone.png", dpi=600, bbox_inches='tight')

    plt.suptitle("Aerosol Responses "+varstr+" changes", fontsize=10, y=0.95)
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_aerosolsinone.pdf", bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_aerosolsinone.pdf", bbox_inches='tight')

    plt.close(fig)


# plot for all responses together with vector
def plotalldiff_lat_vector(lats, levs, res1, tt1, uwnd1, clim1, res2, tt2, uwnd2, clim2, res4, tt4, uwnd4, clim4, opt):
    fig = plt.figure()
    xx, zz = np.meshgrid(lats, levs)
    clevs = np.arange(-3.5, 3.6, 0.5)

    # total response
    ax1 = fig.add_subplot(311)
    cs = ax1.contourf(xx, zz, res1, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    csc = ax1.contour(xx, zz, clim1, levels=np.arange(-60., 60.1, 15.), linewidths=0.5, colors='k')
    ax1.clabel(csc, fontsize=5, inline=1)
    cq = ax1.quiver(xx[::1, ::ratio], zz[::1, ::ratio], uwnd1[::1, ::ratio], -res1[::1, ::ratio], color='grey')
    if (opt == 1):
        levels = [0., 2.01, tt1.max()]
        csm = ax1.contourf(xx, zz, tt1, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    ax1.set_title(r'$\Delta_{total} OMEGA$', fontsize=6, pad=2)
    ax1.set_ylabel("Pressure [hPa]", fontsize=7)
#    ax1.set_xlabel('Latitude [degrees]',fontsize=7)
#    ax1.xaxis.set_tick_params(labelsize=6)
    ax1.set_xticklabels([])
    ax1.yaxis.set_tick_params(labelsize=6)
    ax1.invert_yaxis()

    # fast response
    ax2 = fig.add_subplot(312)
    cs = ax2.contourf(xx, zz, res2, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    csc = ax2.contour(xx, zz, clim2, levels=np.arange(-60., 60.1, 15.), linewidths=0.5, colors='k')
    ax2.clabel(csc, fontsize=5, inline=1)
    cq = ax2.quiver(xx[::1, ::ratio], zz[::1, ::ratio], uwnd2[::1, ::ratio], -res2[::1, ::ratio], color='grey')
    if (opt == 1):
        levels = [0., 2.01, tt2.max()]
        csm = ax2.contourf(xx, zz, tt2, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    ax2.set_title(r'$\Delta_{fast} OMEGA$', fontsize=6, pad=2)
    ax2.set_ylabel("Pressure [hPa]", fontsize=7)
#    ax2.set_xlabel('Latitude [degrees]',fontsize=7)
#    ax2.xaxis.set_tick_params(labelsize=6)
    ax2.set_xticklabels([])
    ax2.yaxis.set_tick_params(labelsize=6)
    ax2.invert_yaxis()

    # slow response
    ax4 = fig.add_subplot(313)
    cs = ax4.contourf(xx, zz, res4, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")
    csc = ax4.contour(xx, zz, clim4, levels=np.arange(-60., 60.1, 15.), linewidths=0.5, colors='k')
    ax4.clabel(csc, fontsize=5, inline=1)
    cq = ax4.quiver(xx[::1, ::ratio], zz[::1, ::ratio], uwnd4[::1, ::ratio], -res4[::1, ::ratio], color='grey')
    if (opt == 1):
        levels = [0., 2.01, tt4.max()]
        csm = ax4.contourf(xx, zz, tt4, levels=levels, colors='none', hatches=["", "..."], alpha=0)

    ax4.set_title(r'$\Delta_{slow} OMEGA$', fontsize=6, pad=2)
    ax4.set_ylabel("Pressure [hPa]", fontsize=7)
    ax4.set_xlabel('Latitude [degrees]', fontsize=7)
    ax4.xaxis.set_tick_params(labelsize=6)
    ax4.yaxis.set_tick_params(labelsize=6)
    ax4.invert_yaxis()

    # add colorbar.
    fig.subplots_adjust(right=0.7, hspace=0.15)
    cbar_ax = fig.add_axes([0.72, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']', fontsize=5, labelpad=0.7)

    # add title
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_aerosolsinone.png", bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_aerosolsinone.png", bbox_inches='tight')

    plt.suptitle("Aerosol Responses "+varstr+" changes", fontsize=10, y=0.95)
    if (opt == 1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_with_siglev_aerosolsinone.pdf", bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_response_vector_" +
                    str(lonbounds[0])+"E_to_"+str(lonbounds[1])+"E_aerosolsinone.pdf", bbox_inches='tight')

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

levli = np.abs(levs - levbounds[0]).argmin()
levui = np.abs(levs - levbounds[1]).argmin()

lats = lats[latli:latui+1]
lons = lons[lonli:lonui+1]
levs = levs[levli:levui+1]

lattest = np.abs(lats - 25).argmin()
lontest = np.abs(lons - 98).argmin()

print(len(levs))
print(len(lats))
print(len(lons))

ratio = int(1.*len(lons)/len(levs))
print(ratio)

print('reading data...')


# read OMEGA
fname1 = var_res+"_OMEGA_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fname2 = var_res+"_OMEGA_"+case2+".cam.h0.0001-0050_vertical_interp.nc"
fname3 = var_res+"_OMEGA_"+case3+".cam.h0.0001-0050_vertical_interp.nc"
fname4 = var_res+"_OMEGA_"+case4+".cam.h0.0001-0050_vertical_interp.nc"
fname5 = var_res+"_OMEGA_"+case5+".cam.h0.0001-0050_vertical_interp.nc"

fdata1 = Dataset(expdir1+fname1)
fdata2 = Dataset(expdir2+fname2)
fdata3 = Dataset(expdir3+fname3)
fdata4 = Dataset(expdir4+fname4)
fdata5 = Dataset(expdir5+fname5)

# read the monthly data for a larger region
w1 = fdata1.variables['OMEGA'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]
w2 = fdata2.variables['OMEGA'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]
w3 = fdata3.variables['OMEGA'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]
w4 = fdata4.variables['OMEGA'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]
w5 = fdata5.variables['OMEGA'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]


# read U wind
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
v1 = fdata1.variables['V'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]
v2 = fdata2.variables['V'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]
v3 = fdata3.variables['V'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]
v4 = fdata4.variables['V'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]
v5 = fdata5.variables['V'][(iniyear-1)*12: (endyear)*12, levli:levui+1, latli:latui+1, lonli:lonui+1]


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
print(levs)

for ilev in range(len(levs)):
    w1[:, ilev, :, :] = np.ma.masked_where(ps1 < levs[ilev], w1[:, ilev, :, :])
    w2[:, ilev, :, :] = np.ma.masked_where(ps2 < levs[ilev], w2[:, ilev, :, :])
    w3[:, ilev, :, :] = np.ma.masked_where(ps3 < levs[ilev], w3[:, ilev, :, :])
    w4[:, ilev, :, :] = np.ma.masked_where(ps4 < levs[ilev], w4[:, ilev, :, :])
    w5[:, ilev, :, :] = np.ma.masked_where(ps5 < levs[ilev], w5[:, ilev, :, :])

    v1[:, ilev, :, :] = np.ma.masked_where(ps1 < levs[ilev], v1[:, ilev, :, :])
    v2[:, ilev, :, :] = np.ma.masked_where(ps2 < levs[ilev], v2[:, ilev, :, :])
    v3[:, ilev, :, :] = np.ma.masked_where(ps3 < levs[ilev], v3[:, ilev, :, :])
    v4[:, ilev, :, :] = np.ma.masked_where(ps4 < levs[ilev], v4[:, ilev, :, :])
    v5[:, ilev, :, :] = np.ma.masked_where(ps5 < levs[ilev], v5[:, ilev, :, :])

print(w1[5, :, lattest, lontest])

w1 = season_ts(w1)
w2 = season_ts(w2)
w3 = season_ts(w3)
w4 = season_ts(w4)
w5 = season_ts(w5)

v1 = season_ts(v1)
v2 = season_ts(v2)
v3 = season_ts(v3)
v4 = season_ts(v4)
v5 = season_ts(v5)

ps1 = season_ts(ps1)
ps2 = season_ts(ps2)
ps3 = season_ts(ps3)
ps4 = season_ts(ps4)
ps5 = season_ts(ps5)

w1_lat = np.mean(w1, axis=3)
w2_lat = np.mean(w2, axis=3)
w3_lat = np.mean(w3, axis=3)
w4_lat = np.mean(w4, axis=3)
w5_lat = np.mean(w5, axis=3)

v1_lat = np.mean(v1, axis=3)
v2_lat = np.mean(v2, axis=3)
v3_lat = np.mean(v3, axis=3)
v4_lat = np.mean(v4, axis=3)
v5_lat = np.mean(v5, axis=3)

ps1_lat = np.mean(ps1, axis=2)
ps2_lat = np.mean(ps2, axis=2)
ps3_lat = np.mean(ps3, axis=2)
ps4_lat = np.mean(ps4, axis=2)
ps5_lat = np.mean(ps5, axis=2)

# print(ps1_lat[:,lattest])


#################################################################################################
# plot climatology
#################################################################################################

# print(ps1_lat[:,lattest])
print('plotting climatology...')
fname = "case1"
titlestr = case1
var, tt = getstats_mean(w1_lat*1000)
ps = np.mean(ps1_lat, axis=0)
vwnd = np.mean(v1_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotclim_lat(lats, levs, var, tt, titlestr, fname, 0)
plotclim_lat(lats, levs, var, tt, titlestr, fname, 1)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 0)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 1)


fname = "case2"
titlestr = case2
var, tt = getstats_mean(w2_lat*1000)
ps = np.mean(ps2_lat, axis=0)
vwnd = np.mean(v2_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotclim_lat(lats, levs, var, tt, titlestr, fname, 0)
plotclim_lat(lats, levs, var, tt, titlestr, fname, 1)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 0)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 1)


fname = "case3"
titlestr = case3
var, tt = getstats_mean(w3_lat*1000)
ps = np.mean(ps3_lat, axis=0)
vwnd = np.mean(v3_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotclim_lat(lats, levs, var, tt, titlestr, fname, 0)
plotclim_lat(lats, levs, var, tt, titlestr, fname, 1)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 0)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 1)

fname = "case4"
titlestr = case4
var, tt = getstats_mean(w4_lat*1000)
ps = np.mean(ps4_lat, axis=0)
vwnd = np.mean(v4_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotclim_lat(lats, levs, var, tt, titlestr, fname, 0)
plotclim_lat(lats, levs, var, tt, titlestr, fname, 1)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 0)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 1)


fname = "case5"
titlestr = case5
var, tt = getstats_mean(w5_lat*1000)
ps = np.mean(ps5_lat, axis=0)
vwnd = np.mean(v5_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotclim_lat(lats, levs, var, tt, titlestr, fname, 0)
plotclim_lat(lats, levs, var, tt, titlestr, fname, 1)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 0)
plotclim_lat_vector(lats, levs, var, tt, vwnd, titlestr, fname, 1)


#################################################################################################
# plot for different forcings
#################################################################################################

print('plotting responses...')

# all aerosol foring
forcingstr = "All aerosol forcings"
forcingfname = "allaerosols"

res, tt = getstats_mean((w2_lat - w5_lat) * 1000)
clim = np.mean(w5_lat * 1000, axis=0)
vwnd = np.mean((v2_lat - v5_lat), axis=0)
ps = np.mean(ps5_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    clim[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 0)
plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 1)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 0)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 1)


# aerosol fast response1
forcingstr = "Aerosol fast response"
forcingfname = "fastaerosol1"

res, tt = getstats_mean((w2_lat - w4_lat) * 1000)
clim = np.mean(w4_lat * 1000, axis=0)
vwnd = np.mean((v2_lat - v4_lat), axis=0)
ps = np.mean(ps4_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    clim[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 0)
plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 1)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 0)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 1)


# aerosol slow response1
forcingstr = "Aerosol slow response"
forcingfname = "slowaerosol1"

res, tt = getstats_mean((w4_lat - w5_lat) * 1000)
clim = np.mean(w5_lat * 1000, axis=0)
vwnd = np.mean((v4_lat - v5_lat), axis=0)
ps = np.mean(ps5_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    clim[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 0)
plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 1)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 0)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 1)


# aerosol fast response2
forcingstr = "Aerosol fast response"
forcingfname = "fastaerosol2"

res, tt = getstats_mean((w3_lat - w5_lat) * 1000)
clim = np.mean(w5_lat * 1000, axis=0)
vwnd = np.mean((v3_lat - v5_lat), axis=0)
ps = np.mean(ps5_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    clim[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 0)
plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 1)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 0)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 1)


# aerosol slow response2
forcingstr = "Aerosol slow response"
forcingfname = "slowaerosol2"

res, tt = getstats_mean((w2_lat - w3_lat) * 1000)
clim = np.mean(w3_lat * 1000, axis=0)
vwnd = np.mean((v2_lat - v3_lat), axis=0)
ps = np.mean(ps3_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    clim[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 0)
plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 1)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 0)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 1)


# GHG and natural forcing
forcingstr = "GHG and natural forcings"
forcingfname = "GHGforcings"

res, tt = getstats_mean((w5_lat - w1_lat) * 1000)
clim = np.mean(w1_lat * 1000, axis=0)
vwnd = np.mean((v5_lat - v1_lat), axis=0)
ps = np.mean(ps1_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    clim[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 0)
plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 1)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 0)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 1)


# All forcings
forcingstr = "All forcings"
forcingfname = "allforcings"

res, tt = getstats_mean((w2_lat - w1_lat) * 1000)
clim = np.mean(w1_lat * 1000, axis=0)
vwnd = np.mean((v2_lat - v1_lat), axis=0)
ps = np.mean(ps1_lat, axis=0)
vwnd = vwnd / np.mean(vwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev, ps < levs[ilev]] = 0
    var[ilev, :] = np.ma.masked_where(ps < levs[ilev], var[ilev, :])
    clim[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim[ilev, :])
    vwnd[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd[ilev, :])

plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 0)
plotdiff_lat(lats, levs, res, tt, clim, forcingstr, forcingfname, 1)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 0)
plotdiff_lat_vector(lats, levs, res, tt, vwnd, clim, forcingstr, forcingfname, 1)


#################################################################################################
# plot all aerosol respoenses in one figure

res1, tt1 = getstats_mean((w2_lat - w5_lat) * 1000)
clim1 = np.mean(w5_lat * 1000, axis=0)
vwnd1 = np.mean((v2_lat - v5_lat), axis=0)
vwnd1 = vwnd1/np.mean(vwnd1) * np.mean(res1)

res2, tt2 = getstats_mean((w2_lat + w3_lat - w4_lat - w5_lat)/2 * 1000)
clim2 = np.mean((w4_lat+w5_lat)/2 * 1000, axis=0)
vwnd2 = np.mean((v2_lat + v3_lat - v4_lat - v5_lat)/2, axis=0)
vwnd2 = vwnd2/np.mean(vwnd2) * np.mean(res2)

res3, tt3 = getstats_mean((w4_lat + w2_lat - w5_lat - w3_lat)/2 * 1000)
clim3 = np.mean((w5_lat+w3_lat)/2 * 1000, axis=0)
vwnd3 = np.mean((v4_lat + v2_lat - v5_lat - v3_lat)/2, axis=0)
vwnd3 = vwnd3/np.mean(vwnd3) * np.mean(res3)
ps = np.mean(ps5_lat, axis=0)

for ilev in range(len(levs)):
    tt1[ilev, ps < levs[ilev]] = 0
    res1[ilev, :] = np.ma.masked_where(ps < levs[ilev], res1[ilev, :])
    clim1[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim1[ilev, :])
    vwnd1[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd1[ilev, :])

    tt2[ilev, ps < levs[ilev]] = 0
    res2[ilev, :] = np.ma.masked_where(ps < levs[ilev], res2[ilev, :])
    clim2[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim2[ilev, :])
    vwnd2[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd2[ilev, :])

    tt3[ilev, ps < levs[ilev]] = 0
    res3[ilev, :] = np.ma.masked_where(ps < levs[ilev], res3[ilev, :])
    clim3[ilev, :] = np.ma.masked_where(ps < levs[ilev], clim3[ilev, :])
    vwnd3[ilev, :] = np.ma.masked_where(ps < levs[ilev], vwnd3[ilev, :])


plotalldiff_lat(lats, levs, res1, tt1, clim1, res2, tt2, clim2, res3, tt3, clim3, 0)
plotalldiff_lat(lats, levs, res1, tt1, clim1, res2, tt2, clim2, res3, tt3, clim3, 1)
plotalldiff_lat_vector(lats, levs, res1, tt1, vwnd1, clim1, res2, tt2, vwnd2, clim2, res3, tt3, vwnd3, clim3, 0)
plotalldiff_lat_vector(lats, levs, res1, tt1, vwnd1, clim1, res2, tt2, vwnd2, clim2, res3, tt3, vwnd3, clim3, 1)
