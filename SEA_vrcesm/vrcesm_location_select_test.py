#This script is used to select vrcesm data directly from the output without regriding
#and compare with each other
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

#set up vrcesm and fv1x1 data directories and filenames
vrcesmdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/vrseasia_AMIP_1979_to_2005/atm/hist/"
vrcesmfname1 = "fv02_prec_vrseasia_AMIP_1979_to_2005.cam.h1.1980-01-01-00000.nc"
vrcesmfname2 = "vrseasia_AMIP_1979_to_2005.cam.h1.1980-01-01-00000.nc"

#define inital year and end year
iniyear = 1980
endyear = 2005

#set up percentile
percentile = 97

#select location
lats = [19.88]
lons = [99.83]

#define a function to select index
def findindex(lons,lats,lon,lat):
    x = np.abs(lons-lon)
    y = np.abs(lats-lat)
    costj = np.multiply(x,y)
    #print(costj)
    index = costj.argmin()

    return index

def nearest(lons,lats,lon,lat):
    x = np.abs(lons-lon)
    y = np.abs(lats-lat)
    index = []
     
    for ind,ix in enumerate(x):
        if (ix<0.31)&(y[ind]<0.25):
            index.append(ind)

    index=np.array(index,dtype=int)

    return index

def findindex4(lons,lats,lon,lat):
    x = np.abs(lons-lon)
    y = np.abs(lats-lat)
    costj = np.multiply(x,y)
    #print(costj)
    index4 = np.zeros((4,),dtype=int)
    for ind,icost in enumerate(costj):
        order = -1
        for i,iind in enumerate(index4):
            if (costj[iind]>=icost):
                order = i

        if (order>0):
            for i in range(order):
                index4[i] = index4[i+1]
            index4[order] = ind

        if (order==0):
            index4[order] = ind

    return index4


#open vrcesm file
vrcesmdata1 = Dataset(vrcesmdir+vrcesmfname1)
vrcesmdata2 = Dataset(vrcesmdir+vrcesmfname2)

#read lat/lon grids
vrlats1 = vrcesmdata1.variables['lat'][:]
vrlons1 = vrcesmdata1.variables['lon'][:]
vrlats2 = vrcesmdata2.variables['lat'][:]
vrlons2 = vrcesmdata2.variables['lon'][:]

print(np.amax(vrlats2))
print(np.amax(vrlons2))

stnlat = lats[0]
stnlon = lons[0]

vrlat_idx1 = np.abs(vrlats1 - stnlat).argmin()
vrlon_idx1 = np.abs(vrlons1 - stnlon).argmin()

#vr_idx2 = findindex(vrlons2,vrlats2,stnlon,stnlat)
vr_idx2 = nearest(vrlons2,vrlats2,stnlon,stnlat)

print(vr_idx2)
print(vrlats2[vr_idx2])
print(vrlons2[vr_idx2])

vrpre1 = vrcesmdata1.variables['PRECT'][:,vrlat_idx1,vrlon_idx1] * 86400 * 1000
#vrpre2 = vrcesmdata2.variables['PRECT'][:,vr_idx2] * 86400 * 1000
vrpre2 = vrcesmdata2.variables['PRECT'][:,vr_idx2] * 86400 * 1000
vrpre2 = np.mean(vrpre2,axis=1)


print(vrpre1 - vrpre2)
print(np.sqrt(((vrpre1 - vrpre2)**2).mean()))
