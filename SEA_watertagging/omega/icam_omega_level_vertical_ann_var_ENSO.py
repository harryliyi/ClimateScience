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
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/verticalcirculation/"

#set up variable names and file name
varname = 'OMEGA'
varfname = "omega"
varstr  = "Vertical velocity"
var_res = "fv09"
var_unit = 'x10^-3 Pa/s'

#define inital year and end year
iniyear = 2
endyear = 50

#define the contour plot region
latbounds = [ -20 , 50 ]
lonbounds = [ 40 , 160 ]

#define top layer
ptop = 200
levbounds = [150, 1000]

#contants
oro_water = 997
g         = 9.8
r_earth   = 6371000

#define Southeast region
reg_lats = [10, 25]
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

#plot for climatology
def plotclim(lats,levs,var,tt,titlestr,fname,opt):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xx, zz = np.meshgrid(lats, levs)
    clevs = np.arange(-15.,15.1,1.)
    cs = ax1.contourf(xx,zz,var,clevs,cmap=cm.bwr,alpha = 0.9,extend="both")
    if (opt==1):
        levels = [0., 2.01, tt_dyn.max()]
        csm = plt.contourf(x,y,tt_dyn,levels=levels,colors='none',hatches=["", "..."], alpha=0)

    ax1.set_ylabel("Pressure [hPa]")
#    ax1.set_yscale('log')

    ax1.set_xlabel('Latitude [degrees]')

    # add colorbar.
    cbar = fig.colorbar(cs,orientation='horizontal',fraction=0.15)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+' ['+var_unit+']')

    plt.gca().invert_yaxis()

    # add title
    plt.title(titlestr+" JJA "+varstr,fontsize=11,y=1.08)
    if (opt==1):
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_contour_with_siglev_"+fname+".pdf")
    else:
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_latvslev_contour_"+fname+".pdf")



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

print('reading data...')


#read Q
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


w1    = season_ts(w1)
w2    = season_ts(w2)
w3    = season_ts(w3)
w4    = season_ts(w4)
w5    = season_ts(w5)

w1_lat    = np.mean(w1, axis=3)
w2_lat    = np.mean(w2, axis=3)
w3_lat    = np.mean(w3, axis=3)
w4_lat    = np.mean(w4, axis=3)
w5_lat    = np.mean(w5, axis=3)

#################################################################################################
#plot climatology
#################################################################################################
print('plotting climatology...')
fname = "case1"
titlestr = case1
var,tt = getstats_mean(w1_lat*1000)
plotclim(lats[latli:latui+1],levs[levli:levui+1],var,tt, titlestr, fname,0)
plotclim(lats[latli:latui+1],levs[levli:levui+1],var,tt, titlestr, fname,1)








