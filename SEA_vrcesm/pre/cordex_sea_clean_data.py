#import modules
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc
import matplotlib.pyplot as plt
import datetime as datetime

def readcordex(varname,iniyear,endyear,project,modelname,frequency,latbounds,lonbounds):
    models = {
             'ICHEC-EC-EARTH'   :'ICHEC-EC-EARTH_historical_r1i1p1_ICTP-RegCM4-3_v4',
             'IPSL-IPSL-CM5A-LR':'IPSL-IPSL-CM5A-LR_historical_r1i1p1_ICTP-RegCM4-3_v4',
             'MOHC-HadGEM2-ES'  :'MOHC-HadGEM2-ES_historical_r1i1p1_SMHI-RCA4_v1',
             'MPI-M-MPI-ESM-MR' :'MPI-M-MPI-ESM-MR_historical_r1i1p1_ICTP-RegCM4-3_v4',
             }
    yearts = range(iniyear,endyear+1,1)
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01','%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01','%Y-%m-%d')

    fdir  = '/scratch/d/dylan/harryli/obsdataset/CORDEX_SEA/'+modelname+'/historical/'+varname+'/'+frequency+'/'
    fname = varname+'_'+project+'_'+models[modelname]+'_'+frequency+'_'
    fyears  = [fdir+fname+str(iyear)+'.nc' for iyear in yearts]
    #print(fyears)

    dataset = nc.MFDataset(fyears)
    time_var = dataset.variables['time']
    dtime  = nc.num2date(time_var[:],time_var.units)
    dtime  = pd.to_datetime(dtime)
    print(dtime)
    select_dtime = (dtime>=inidate)&(dtime<enddate)&(~((dtime.month==2)&(dtime.day==29)))
    dtime  = dtime[select_dtime]
    
    lats = dataset.variables['lat'][:,0]
    lons = dataset.variables['lon'][0,:]

    #print(lats)
    #print(lons)    

    lat_1 = np.argmin( np.abs( lats - latbounds[0] ) )
    lat_2 = np.argmin( np.abs( lats - latbounds[1] ) )
    lon_1 = np.argmin( np.abs( lons - lonbounds[0] ) )
    lon_2 = np.argmin( np.abs( lons - lonbounds[1] ) )

    var  = dataset.variables[varname][select_dtime, lat_1 : lat_2 + 1, lon_1 : lon_2 + 1]
    lats = lats[lat_1 : lat_2 + 1]
    lons = lons[lon_1 : lon_2 + 1]

    return var,dtime,lats,lons 

varname   = 'pr'
iniyear   = 1980
endyear   = 2005
project   = 'SEA-22'
modelname = 'ICHEC-EC-EARTH'
frequency = 'day'

#define the contour plot region
latbounds = [ -1 , 25 ]
lonbounds = [ 90 , 130 ]

var,time,lats, lons = readcordex(varname,iniyear,endyear,project,modelname,frequency,latbounds,lonbounds)
print(lats)
print(lons)
print(time)
print(var.shape)
    
