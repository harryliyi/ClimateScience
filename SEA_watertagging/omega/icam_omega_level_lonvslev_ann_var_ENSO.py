#This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

#S1-plot climatological data
#S2-plot contours  
#
#by Harry Li

#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import pandas as pd
import math as math


#set up data directories and filenames
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

#set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/verticalcirculation/lonvslev/"

#set up variable names and file name
varname = 'OMEGA'
varfname = "omega"
varstr  = "Vertical velocity"
var_res = "fv09"
var_unit = r'$10^{-3}$ Pa/s'

#define inital year and end year
iniyear = 2
endyear = 50

#define the contour plot region
latbounds = [ 20 , 30 ]
lonbounds = [ 20 , 160 ]

#define top layer
ptop = 150
levbounds = [150, 1000]

#contants
oro_water = 997
g         = 9.8
r_earth   = 6371000

#define Southeast region
reg_lons = [10, 25]
reg_lons = [100, 110]

################################################################################################
#S0-Define functions
################################################################################################
#calculate difference of mean and significance level
def getstats_diff(var1,var2):
    n1=var1.shape[0]
    n2=var2.shape[0]

    var1mean = np.mean(var1,axis = 0)
    var2mean = np.mean(var2,axis = 0)
    var1std  = np.std(var1,axis = 0)
    var2std  = np.std(var2,axis = 0)

    vardiff  = var1mean - var2mean
    varttest = vardiff/np.sqrt(var1std**2/n1+var2std**2/n2)

    return vardiff,abs(varttest)

#calculate hypothesis test of mean
def getstats_mean(var):
    n=var.shape[0]
    varmean = np.mean(var,axis = 0)
    varstd  = np.std(var,axis = 0)

    varttest = varmean/(varstd/n)

    return varmean,abs(varttest)

#calculate seasonal mean
def season_ts(var):
    if (len(var.shape)==3):
        varseasts = np.zeros(((endyear-iniyear+1),var.shape[1],var.shape[2]))
        for iyear in range(endyear-iniyear+1):
            varseasts[iyear,:,:] = np.mean(var[iyear*12+5:iyear*12+8,:,:],axis=0)
    if (len(var.shape)==4):
        varseasts = np.zeros(((endyear-iniyear+1),var.shape[1],var.shape[2],var.shape[3]))
        for iyear in range(endyear-iniyear+1):
            varseasts[iyear,:,:,:] = np.mean(var[iyear*12+5:iyear*12+8,:,:,:],axis=0)

    return varseasts
##########################################################################
#plot for climatology
def plotclim_lon_vector(lons,levs,var,tt,uwnd,titlestr,fname,opt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xx, zz = np.meshgrid(lons, levs)
    clevs = np.arange(-8.,8.1,0.5)
    cs = ax1.contourf(xx,zz,var,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    cq = ax1.quiver(xx[::1,::ratio],zz[::1,::ratio],uwnd[::1,::ratio],-var[::1,::ratio],color='grey')
#    qk = ax1.quiverkey(cq, 0.9, 0.9, 35, '35 '+var_unit, labelpos='E', coordinates='figure',fontproperties={'size': 8})
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx,zz,tt,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax1.set_ylabel("Pressure [hPa]")
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)
#    ax1.set_yscale('log')

    ax1.set_xlabel('Longitude [degrees]')

    # add colorbar.
    cbar = fig.colorbar(cs,orientation='horizontal',fraction=0.15, aspect= 25, shrink = 0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+r' [$ \times 10^{-2}$ Pa/s]',fontsize=8)

    plt.gca().invert_yaxis()

    # add title
    plt.title(titlestr+" JJA "+varstr,fontsize=11,y=1.08)
    if (opt==1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_vector_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_with_siglev_"+fname+".pdf")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_vector_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_"+fname+".pdf")

    plt.close(fig)


#plot for climatology without vector
def plotclim_lon(lons,levs,var,tt,titlestr,fname,opt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xx, zz = np.meshgrid(lons, levs)
    clevs = np.arange(-8.,8.1,0.5)
    cs = ax1.contourf(xx,zz,var,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx,zz,tt,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax1.set_ylabel("Pressure [hPa]")
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)
    ax1.set_xlabel('Longitude [degrees]')

    # add colorbar.
    cbar = fig.colorbar(cs,orientation='horizontal',fraction=0.15, aspect= 25, shrink = 0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+r' [$ \times 10^{-2}$ Pa/s]',fontsize=8)

    plt.gca().invert_yaxis()

    # add title
    plt.title(titlestr+" JJA "+varstr,fontsize=11,y=1.08)
    if (opt==1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_with_siglev_"+fname+".pdf")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_"+fname+".pdf")

    plt.close(fig)

##########################################################################
#plot for response
def plotdiff_lon_vector(lons,levs,var,tt,uwnd,clim,forcingstr,forcingfname,opt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xx, zz = np.meshgrid(lons, levs)
    clevs = np.arange(-5.0,5.1,0.5)
    cs = ax1.contourf(xx,zz,var,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    csc= ax1.contour(xx, zz, clim, levels=np.arange(-60.,60.1,15.), linewidths=0.5, colors='k')
    cq = ax1.quiver(xx[::1,::ratio],zz[::1,::ratio],uwnd[::1,::ratio],-var[::1,::ratio],color='grey')
    ax1.clabel(csc, fontsize=5, inline=1)
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx,zz,tt,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax1.set_ylabel("Pressure [hPa]")
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)
#    ax1.set_yscale('log')

    ax1.set_xlabel('Longitude [degrees]')

    # add colorbar.
    cbar = fig.colorbar(cs,orientation='horizontal',fraction=0.15, aspect= 25, shrink = 0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']',fontsize=8)

    plt.gca().invert_yaxis()

    # add title
    plt.title(forcingstr+" "+varstr+" changes",fontsize=11,y=1.08)
    if (opt==1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_response_vector_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_with_siglev_"+forcingfname+".pdf")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_response_vector_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_"+forcingfname+".pdf")

    plt.close(fig)


#plot for response without vector
def plotdiff_lon(lons,levs,var,tt,clim,forcingstr,forcingfname,opt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xx, zz = np.meshgrid(lons, levs)
    clevs = np.arange(-5.0,5.1,0.5)
    cs = ax1.contourf(xx,zz,var,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    csc= ax1.contour(xx, zz, clim, levels=np.arange(-60.,60.1,15.), linewidths=0.5, colors='k')
    ax1.clabel(csc, fontsize=5, inline=1)
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx,zz,tt,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax1.set_ylabel("Pressure [hPa]")
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)
    ax1.set_xlabel('Longitude [degrees]')

    # add colorbar.
    cbar = fig.colorbar(cs,orientation='horizontal',fraction=0.15, aspect= 25, shrink = 0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']',fontsize=8)

    plt.gca().invert_yaxis()

    # add title
    plt.title(forcingstr+" "+varstr+" changes",fontsize=11,y=1.08)
    if (opt==1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_response_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_with_siglev_"+forcingfname+".pdf")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_response_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_"+forcingfname+".pdf")

    plt.close(fig)


##########################################################################
#plot for all responses together
def plotalldiff_lon_vector(lons,levs,res1,tt1,uwnd1,clim1,res2,tt2,uwnd2,clim2,res4,tt4,uwnd4,clim4,opt):
    fig = plt.figure()
    xx, zz = np.meshgrid(lons, levs)
    clevs = np.arange(-5.,5.1,0.5)

    #total response
    ax1 = fig.add_subplot(311)
    cs = ax1.contourf(xx,zz,res1,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    csc= ax1.contour(xx, zz, clim1, levels=np.arange(-60.,60.1,15.), linewidths=0.5, colors='k')
    cq = ax1.quiver(xx[::1,::ratio],zz[::1,::ratio],uwnd1[::1,::ratio],-res1[::1,::ratio],color='grey')
    ax1.clabel(csc, fontsize=5, inline=1)
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx,zz,tt1,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax1.set_title(r'$\Delta_{total} OMEGA$',fontsize=6,pad=2)
    ax1.set_ylabel("Pressure [hPa]",fontsize=7)
#    ax1.set_xlabel('Longitude [degrees]',fontsize=7)
#    ax1.xaxis.set_tick_params(labelsize=6)
    ax1.set_xticklabels([])
    ax1.yaxis.set_tick_params(labelsize=6)
    ax1.invert_yaxis()

    #fast response
    ax2 = fig.add_subplot(312)
    cs = ax2.contourf(xx,zz,res2,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    csc= ax2.contour(xx, zz, clim2, levels=np.arange(-60.,60.1,15.), linewidths=0.5, colors='k')
    cq = ax2.quiver(xx[::1,::ratio],zz[::1,::ratio],uwnd2[::1,::ratio],-res2[::1,::ratio],color='grey')
    ax2.clabel(csc, fontsize=5, inline=1)
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax2.contourf(xx,zz,tt2,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax2.set_title(r'$\Delta_{fast} OMEGA$',fontsize=6,pad=2)
    ax2.set_ylabel("Pressure [hPa]",fontsize=7)
#    ax2.set_xlabel('Longitude [degrees]',fontsize=7)
#    ax2.xaxis.set_tick_params(labelsize=6)
    ax2.set_xticklabels([])
    ax2.yaxis.set_tick_params(labelsize=6)
    ax2.invert_yaxis()
 
    #slow response
    ax4 = fig.add_subplot(313)
    cs = ax4.contourf(xx,zz,res4,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    csc= ax4.contour(xx, zz, clim4, levels=np.arange(-60.,60.1,15.), linewidths=0.5, colors='k')
    cq = ax4.quiver(xx[::1,::ratio],zz[::1,::ratio],uwnd4[::1,::ratio],-res4[::1,::ratio],color='grey')
    ax4.clabel(csc, fontsize=5, inline=1)
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax4.contourf(xx,zz,tt4,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax4.set_title(r'$\Delta_{slow} OMEGA$',fontsize=6,pad=2)
    ax4.set_ylabel("Pressure [hPa]",fontsize=7)
    ax4.set_xlabel('Longitude [degrees]',fontsize=7)
    ax4.xaxis.set_tick_params(labelsize=6)
    ax4.yaxis.set_tick_params(labelsize=6)
    ax4.invert_yaxis()

    # add colorbar.
    fig.subplots_adjust(right=0.7,hspace = 0.15)
    cbar_ax = fig.add_axes([0.72, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs,cax = cbar_ax,orientation='vertical')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']',fontsize=5,labelpad=0.7)

    # add title
    plt.suptitle("Aerosol Responses "+varstr+" changes",fontsize=10,y=0.95)
    if (opt==1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_response_vector_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_with_siglev_aerosolsinone.pdf",bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_response_vector_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_aerosolsinone.pdf",bbox_inches='tight')

    plt.close(fig)



#plot for all responses together without vector
def plotalldiff_lon(lons,levs,res1,tt1,clim1,res2,tt2,clim2,res4,tt4,clim4,opt):
    fig = plt.figure()
    xx, zz = np.meshgrid(lons, levs)
    clevs = np.arange(-5.,5.1,0.5)

    #total response
    ax1 = fig.add_subplot(311)
    cs = ax1.contourf(xx,zz,res1,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    csc= ax1.contour(xx, zz, clim1, levels=np.arange(-60.,60.1,15.), linewidths=0.5, colors='k')
    ax1.clabel(csc, fontsize=5, inline=1)
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax1.contourf(xx,zz,tt1,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax1.set_title(r'$\Delta_{total} OMEGA$',fontsize=6,pad=2)
    ax1.set_ylabel("Pressure [hPa]",fontsize=7)
    ax1.set_xticklabels([])
    ax1.yaxis.set_tick_params(labelsize=6)
    ax1.invert_yaxis()

    #fast response
    ax2 = fig.add_subplot(312)
    cs = ax2.contourf(xx,zz,res2,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    csc= ax2.contour(xx, zz, clim2, levels=np.arange(-60.,60.1,15.), linewidths=0.5, colors='k')
    ax2.clabel(csc, fontsize=5, inline=1)
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax2.contourf(xx,zz,tt2,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax2.set_title(r'$\Delta_{fast} OMEGA$',fontsize=6,pad=2)
    ax2.set_ylabel("Pressure [hPa]",fontsize=7)
    ax2.set_xticklabels([])
    ax2.yaxis.set_tick_params(labelsize=6)
    ax2.invert_yaxis()

    #slow response
    ax4 = fig.add_subplot(313)
    cs = ax4.contourf(xx,zz,res4,clevs,cmap=cm.RdBu_r,alpha = 0.9,extend="both")
    csc= ax4.contour(xx, zz, clim4, levels=np.arange(-60.,60.1,15.), linewidths=0.5, colors='k')
    ax4.clabel(csc, fontsize=5, inline=1)
    if (opt==1):
        levels = [0., 2.01, tt.max()]
        csm = ax4.contourf(xx,zz,tt4,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax4.set_title(r'$\Delta_{slow} OMEGA$',fontsize=6,pad=2)
    ax4.set_ylabel("Pressure [hPa]",fontsize=7)
    ax4.set_xlabel('Longitude [degrees]',fontsize=7)
    ax4.xaxis.set_tick_params(labelsize=6)
    ax4.yaxis.set_tick_params(labelsize=6)
    ax4.invert_yaxis()

    # add colorbar.
    fig.subplots_adjust(right=0.7,hspace = 0.15)
    cbar_ax = fig.add_axes([0.72, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs,cax = cbar_ax,orientation='vertical')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label(varstr+r' [$\times$ '+var_unit+']',fontsize=5,labelpad=0.7)

    # add title
    plt.suptitle("Aerosol Responses "+varstr+" changes",fontsize=10,y=0.95)
    if (opt==1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_response_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_with_siglev_aerosolsinone.pdf",bbox_inches='tight')
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_lonvslev_response_"+str(latbounds[0])+"N_to_"+str(latbounds[1])+"N_aerosolsinone.pdf",bbox_inches='tight')

    plt.close(fig)


################################################################################################
#S1-read climatological data
################################################################################################
#read lats,lons,levs,
fname1 = var_res+"_Q_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fdata1  = Dataset(expdir1+fname1)
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

lats  = lats[latli:latui+1]
lons  = lons[lonli:lonui+1]
levs  = levs[levli:levui+1]

print(len(levs))
print(len(lats))
print(len(lons))

ratio = int(1.*len(lons)/len(levs))-1
print(ratio)

lattest = np.abs(lats - 25).argmin()
lontest = np.abs(lons - 98).argmin()

print('reading data...')


#read OMEGA
fname1 = var_res+"_OMEGA_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fname2 = var_res+"_OMEGA_"+case2+".cam.h0.0001-0050_vertical_interp.nc"
fname3 = var_res+"_OMEGA_"+case3+".cam.h0.0001-0050_vertical_interp.nc"
fname4 = var_res+"_OMEGA_"+case4+".cam.h0.0001-0050_vertical_interp.nc"
fname5 = var_res+"_OMEGA_"+case5+".cam.h0.0001-0050_vertical_interp.nc"

fdata1  = Dataset(expdir1+fname1)
fdata2  = Dataset(expdir2+fname2)
fdata3  = Dataset(expdir3+fname3)
fdata4  = Dataset(expdir4+fname4)
fdata5  = Dataset(expdir5+fname5)

#read the monthly data for a larger region
w1 = fdata1.variables['OMEGA'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]
w2 = fdata2.variables['OMEGA'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]
w3 = fdata3.variables['OMEGA'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]
w4 = fdata4.variables['OMEGA'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]
w5 = fdata5.variables['OMEGA'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]


#read U wind
fname1 = var_res+"_U_WIND_"+case1+".cam.h0.0001-0050_vertical_interp.nc"
fname2 = var_res+"_U_WIND_"+case2+".cam.h0.0001-0050_vertical_interp.nc"
fname3 = var_res+"_U_WIND_"+case3+".cam.h0.0001-0050_vertical_interp.nc"
fname4 = var_res+"_U_WIND_"+case4+".cam.h0.0001-0050_vertical_interp.nc"
fname5 = var_res+"_U_WIND_"+case5+".cam.h0.0001-0050_vertical_interp.nc"

fdata1  = Dataset(expdir1+fname1)
fdata2  = Dataset(expdir2+fname2)
fdata3  = Dataset(expdir3+fname3)
fdata4  = Dataset(expdir4+fname4)
fdata5  = Dataset(expdir5+fname5)

#read the monthly data for a larger region
u1 = fdata1.variables['U'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]
u2 = fdata2.variables['U'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]
u3 = fdata3.variables['U'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]
u4 = fdata4.variables['U'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]
u5 = fdata5.variables['U'][ (iniyear-1)*12 : (endyear)*12 , levli:levui+1 , latli:latui+1 , lonli:lonui+1 ]

#read PS
fname1 = var_res+"_PS_"+case1+".cam.h0.0001-0050.nc"
fname2 = var_res+"_PS_"+case2+".cam.h0.0001-0050.nc"
fname3 = var_res+"_PS_"+case3+".cam.h0.0001-0050.nc"
fname4 = var_res+"_PS_"+case4+".cam.h0.0001-0050.nc"
fname5 = var_res+"_PS_"+case5+".cam.h0.0001-0050.nc"

fdata1  = Dataset(expdir1+fname1)
fdata2  = Dataset(expdir2+fname2)
fdata3  = Dataset(expdir3+fname3)
fdata4  = Dataset(expdir4+fname4)
fdata5  = Dataset(expdir5+fname5)

#read the monthly data for a larger region
ps1 = fdata1.variables['PS'][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]
ps2 = fdata2.variables['PS'][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]
ps3 = fdata3.variables['PS'][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]
ps4 = fdata4.variables['PS'][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]
ps5 = fdata5.variables['PS'][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]

ps1 = ps1/100
ps2 = ps2/100
ps3 = ps3/100
ps4 = ps4/100
ps5 = ps5/100

print(ps1[5,lattest,lontest])
print(levs)

for ilev in range(len(levs)):
    w1[:,ilev,:,:] = np.ma.masked_where(ps1<levs[ilev],w1[:,ilev,:,:])
    w2[:,ilev,:,:] = np.ma.masked_where(ps2<levs[ilev],w2[:,ilev,:,:])
    w3[:,ilev,:,:] = np.ma.masked_where(ps3<levs[ilev],w3[:,ilev,:,:])
    w4[:,ilev,:,:] = np.ma.masked_where(ps4<levs[ilev],w4[:,ilev,:,:])
    w5[:,ilev,:,:] = np.ma.masked_where(ps5<levs[ilev],w5[:,ilev,:,:])    

    u1[:,ilev,:,:] = np.ma.masked_where(ps1<levs[ilev],u1[:,ilev,:,:])
    u2[:,ilev,:,:] = np.ma.masked_where(ps2<levs[ilev],u2[:,ilev,:,:])
    u3[:,ilev,:,:] = np.ma.masked_where(ps3<levs[ilev],u3[:,ilev,:,:])
    u4[:,ilev,:,:] = np.ma.masked_where(ps4<levs[ilev],u4[:,ilev,:,:])
    u5[:,ilev,:,:] = np.ma.masked_where(ps5<levs[ilev],u5[:,ilev,:,:])

print(w1[5,:,lattest,lontest])

w1    = season_ts(w1)
w2    = season_ts(w2)
w3    = season_ts(w3)
w4    = season_ts(w4)
w5    = season_ts(w5)

u1    = season_ts(u1)
u2    = season_ts(u2)
u3    = season_ts(u3)
u4    = season_ts(u4)
u5    = season_ts(u5)

ps1   = season_ts(ps1)
ps2   = season_ts(ps2)
ps3   = season_ts(ps3)
ps4   = season_ts(ps4)
ps5   = season_ts(ps5)

w1_lon    = np.mean(w1, axis=2)
w2_lon    = np.mean(w2, axis=2)
w3_lon    = np.mean(w3, axis=2)
w4_lon    = np.mean(w4, axis=2)
w5_lon    = np.mean(w5, axis=2)

u1_lon    = np.mean(u1, axis=2)
u2_lon    = np.mean(u2, axis=2)
u3_lon    = np.mean(u3, axis=2)
u4_lon    = np.mean(u4, axis=2)
u5_lon    = np.mean(u5, axis=2)

ps1_lon   = np.mean(ps1, axis=1)
ps2_lon   = np.mean(ps2, axis=1)
ps3_lon   = np.mean(ps3, axis=1)
ps4_lon   = np.mean(ps4, axis=1)
ps5_lon   = np.mean(ps5, axis=1)

#print(ps1_lon[:,lattest])


#################################################################################################
#plot climatology
#################################################################################################

#print(ps1_lon[:,lattest])
print('plotting climatology...')
fname = "case1"
titlestr = case1
var,tt = getstats_mean(w1_lon*100)
ps     = np.mean(ps1_lon, axis=0)
uwnd   = np.mean(u1_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,0)
plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,1)
plotclim_lon(lons,levs,var,tt, titlestr, fname,0)
plotclim_lon(lons,levs,var,tt, titlestr, fname,1)

fname = "case2"
titlestr = case2
var,tt = getstats_mean(w2_lon*100)
ps     = np.mean(ps2_lon, axis=0)
uwnd   = np.mean(u2_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,0)
plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,1)
plotclim_lon(lons,levs,var,tt, titlestr, fname,0)
plotclim_lon(lons,levs,var,tt, titlestr, fname,1)

fname = "case3"
titlestr = case3
var,tt = getstats_mean(w3_lon*100)
ps     = np.mean(ps3_lon, axis=0)
uwnd   = np.mean(u3_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,0)
plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,1)
plotclim_lon(lons,levs,var,tt, titlestr, fname,0)
plotclim_lon(lons,levs,var,tt, titlestr, fname,1)


fname = "case4"
titlestr = case4
var,tt = getstats_mean(w4_lon*100)
ps     = np.mean(ps4_lon, axis=0)
uwnd   = np.mean(u4_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,0)
plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,1)
plotclim_lon(lons,levs,var,tt, titlestr, fname,0)
plotclim_lon(lons,levs,var,tt, titlestr, fname,1)


fname = "case5"
titlestr = case5
var,tt = getstats_mean(w5_lon*100)
ps     = np.mean(ps5_lon, axis=0)
uwnd   = np.mean(u5_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(var)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,0)
plotclim_lon_vector(lons,levs,var,tt,uwnd, titlestr, fname,1)
plotclim_lon(lons,levs,var,tt, titlestr, fname,0)
plotclim_lon(lons,levs,var,tt, titlestr, fname,1)


#################################################################################################
#plot for different forcings
#################################################################################################

print('plotting responses...')

#all aerosol foring
forcingstr   = "All aerosol forcings"
forcingfname = "allaerosols"

res,tt    = getstats_mean((w2_lon - w5_lon)* 1000)
clim      = np.mean(w5_lon * 1000,axis=0)
uwnd      = np.mean((u2_lon - u5_lon),axis=0)
ps        = np.mean(ps5_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    clim[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,0)
plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,1)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,0)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,1)

#aerosol fast response1
forcingstr   = "Aerosol fast response"
forcingfname = "fastaerosol1"

res,tt    = getstats_mean((w2_lon - w4_lon)* 1000)
clim      = np.mean(w4_lon * 1000,axis=0)
uwnd      = np.mean((u2_lon - u4_lon),axis=0)
ps        = np.mean(ps4_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    clim[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,0)
plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,1)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,0)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,1)


#aerosol slow response1
forcingstr   = "Aerosol slow response"
forcingfname = "slowaerosol1"

res,tt    = getstats_mean((w4_lon - w5_lon)* 1000)
clim      = np.mean(w5_lon * 1000,axis=0)
uwnd      = np.mean((u4_lon - u5_lon),axis=0)
ps        = np.mean(ps5_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    clim[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,0)
plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,1)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,0)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,1)


#aerosol fast response2
forcingstr   = "Aerosol fast response"
forcingfname = "fastaerosol2"

res,tt    = getstats_mean((w3_lon - w5_lon)* 1000)
clim      = np.mean(w5_lon * 1000,axis=0)
uwnd      = np.mean((u3_lon - u5_lon),axis=0)
ps        = np.mean(ps5_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    clim[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,0)
plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,1)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,0)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,1)


#aerosol slow response2
forcingstr   = "Aerosol slow response"
forcingfname = "slowaerosol2"

res,tt    = getstats_mean((w2_lon - w3_lon)* 1000)
clim      = np.mean(w3_lon * 1000,axis=0)
uwnd      = np.mean((u2_lon - u3_lon),axis=0)
ps        = np.mean(ps3_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    clim[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,0)
plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,1)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,0)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,1)


#GHG and natural forcing
forcingstr   = "GHG and natural forcings"
forcingfname = "GHGforcings"

res,tt    = getstats_mean((w5_lon - w1_lon)* 1000)
clim      = np.mean(w1_lon * 1000,axis=0)
uwnd      = np.mean((u5_lon - u1_lon),axis=0)
ps        = np.mean(ps1_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    clim[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,0)
plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,1)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,0)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,1)


#All forcings
forcingstr   = "All forcings"
forcingfname = "allforcings"

res,tt    = getstats_mean((w2_lon - w1_lon)* 1000)
clim      = np.mean(w1_lon * 1000,axis=0)
uwnd      = np.mean((u2_lon - u1_lon),axis=0)
ps        = np.mean(ps1_lon, axis=0)
uwnd      = uwnd /np.mean(uwnd) * np.mean(res)

for ilev in range(len(levs)):
    tt[ilev,ps<levs[ilev]] = 0
    var[ilev,:]= np.ma.masked_where(ps<levs[ilev],var[ilev,:])
    clim[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim[ilev,:])
    uwnd[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd[ilev,:])

plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,0)
plotdiff_lon_vector(lons,levs,res,tt,uwnd,clim, forcingstr,forcingfname,1)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,0)
plotdiff_lon(lons,levs,res,tt,clim, forcingstr,forcingfname,1)


#################################################################################################
#plot all aerosol respoenses in one figure

res1,tt1    = getstats_mean((w2_lon - w5_lon)* 1000)
clim1      = np.mean(w5_lon * 1000,axis=0)
uwnd1      = np.mean((u2_lon - u5_lon),axis=0)
uwnd1      = uwnd1/np.mean(uwnd1) * np.mean(res1)

res2,tt2    = getstats_mean((w2_lon + w3_lon - w4_lon - w5_lon)/2* 1000)
clim2      = np.mean((w4_lon+w5_lon)/2 * 1000,axis=0)
uwnd2      = np.mean((u2_lon + u3_lon - u4_lon - u5_lon)/2,axis=0)
uwnd2      = uwnd2/np.mean(uwnd2) * np.mean(res2)

res3,tt3    = getstats_mean((w4_lon + w2_lon - w5_lon - w3_lon)/2* 1000)
clim3      = np.mean((w5_lon + w3_lon)/2 * 1000,axis=0)
uwnd3      = np.mean((u4_lon + u2_lon - u5_lon - u3_lon)/2,axis=0)
ps         = np.mean(ps5_lon, axis=0)
uwnd3      = uwnd3/np.mean(uwnd3) * np.mean(res3)

for ilev in range(len(levs)):
    tt1[ilev,ps<levs[ilev]] = 0
    res1[ilev,:]= np.ma.masked_where(ps<levs[ilev],res1[ilev,:])
    clim1[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim1[ilev,:])
    uwnd1[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd1[ilev,:])

    tt2[ilev,ps<levs[ilev]] = 0
    res2[ilev,:]= np.ma.masked_where(ps<levs[ilev],res2[ilev,:])
    clim2[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim2[ilev,:])
    uwnd2[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd2[ilev,:])

    tt3[ilev,ps<levs[ilev]] = 0
    res3[ilev,:]= np.ma.masked_where(ps<levs[ilev],res3[ilev,:])
    clim3[ilev,:]= np.ma.masked_where(ps<levs[ilev],clim3[ilev,:])
    uwnd3[ilev,:]= np.ma.masked_where(ps<levs[ilev],uwnd3[ilev,:])



plotalldiff_lon_vector(lons,levs,res1,tt1,uwnd1,clim1,res2,tt2,uwnd2,clim2,res3,tt3,uwnd3,clim3,0)
plotalldiff_lon_vector(lons,levs,res1,tt1,uwnd1,clim1,res2,tt2,uwnd2,clim2,res3,tt3,uwnd3,clim3,1)
plotalldiff_lon(lons,levs,res1,tt1,clim1,res2,tt2,clim2,res3,tt3,clim3,0)
plotalldiff_lon(lons,levs,res1,tt1,clim1,res2,tt2,clim2,res3,tt3,clim3,1)

