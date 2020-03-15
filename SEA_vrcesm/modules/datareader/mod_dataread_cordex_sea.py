'''
#This is a function to read data from CORDEX-SEA data.

#Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as datetime
from mpl_toolkits import basemap

'''
This is a function used to read data from vrcesm

Description on parameters:
1) varname: the variable name in netCDF

2) iniyear/endyear: the time bounds

3) project: here is SEA-22

4) fequency: monthly, daily or hourly data to read

5) latbounds/lonbounds: the horizontal boundaries of data

6) oceanmask: whether need to mask the ocean, default will not maks the ocean,
    0: do not mask any
    1: maks the ocean
   -1: mask the land

7) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readcordex(varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, **kwargs):

    models = {
        'ICHEC-EC-EARTH': 'ICHEC-EC-EARTH_historical_r1i1p1_ICTP-RegCM4-3_v4',
        'IPSL-IPSL-CM5A-LR': 'IPSL-IPSL-CM5A-LR_historical_r1i1p1_ICTP-RegCM4-3_v4',
        'MOHC-HadGEM2-ES': 'MOHC-HadGEM2-ES_historical_r1i1p1_SMHI-RCA4_v1',
        'MPI-M-MPI-ESM-MR': 'MPI-M-MPI-ESM-MR_historical_r1i1p1_ICTP-RegCM4-3_v4',
    }

    yearts = range(iniyear, endyear+1, 1)
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    # setup for land fraction
    dir_lndfrc = '/scratch/d/dylan/harryli/obsdataset/CORDEX_SEA/MOHC-HadGEM2-ES/historical/sftlf/'
    flndfrc = 'sftlf_SEA-22_MOHC-HadGEM2-ES_historical_r0i0p0_SMHI-RCA4_v1_fx.nc'

    fdir = '/scratch/d/dylan/harryli/obsdataset/CORDEX_SEA/' + \
        modelname+'/historical/'+varname+'/'+frequency+'/'

    if modelname == 'MOHC-HadGEM2-ES':
        fname = varname+'_'+project+'_'+models[modelname]+'_'+frequency+'_1951-2005.nc'
        dataset = nc.Dataset(fdir+fname)
    else:
        fname = varname+'_'+project+'_'+models[modelname]+'_'+frequency+'_'
        fyears = [fdir+fname+str(iyear)+'.nc' for iyear in yearts]
        # print(fyears)
        dataset = nc.MFDataset(fyears)

    time_var = dataset.variables['time']
    cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    dtime = []
    for istr in cftime:
        temp = istr.strftime('%Y-%m-%d %H:%M:%S')
        dtime.append(temp)
    # dtime  = nc.num2date(time_var[:],time_var.units)
    # print(cftime)
    # print(dtime)
    dtime = pd.to_datetime(dtime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # print(dtime)
    if modelname == 'MOHC-HadGEM2-ES':
        select_dtime = (dtime >= inidate) & (dtime < enddate) & (
            ~ np.in1d(dtime.year, ignore_years))
    else:
        select_dtime = (dtime >= inidate) & (dtime < enddate) & (
            ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
    dtime = dtime[select_dtime]
    # print(select_dtime)
    # print(dtime[0:365])

    # print(np.amin(dataset.variables['lat'][:,:]))
    # print(np.amax(dataset.variables['lat'][:,:]))
    # print(np.amin(dataset.variables['lon'][:,:]))
    # print(np.amax(dataset.variables['lon'][:,:]))

    lats = dataset.variables['lat'][:, 0]
    lons = dataset.variables['lon'][0, :]
    # lats = dataset.variables['lat'][:, :]
    # lons = dataset.variables['lon'][:, :]

    # print(len(lats))
    # print(lats)
    # print(lons)

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))
    # print(lat_1,lat_2,lon_1,lon_2)

    var = dataset.variables[varname][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    # var = dataset.variables[varname][select_dtime, :, :]

    var = np.ma.array(var)

    if oceanmask == 1:
        dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc)
        lnd_lats = dataset_lndfrc.variables['lat'][:, 0]
        lnd_lons = dataset_lndfrc.variables['lon'][0, :]
        lnd_lat_1 = np.argmin(np.abs(lnd_lats - latbounds[0]))
        lnd_lat_2 = np.argmin(np.abs(lnd_lats - latbounds[1]))
        lnd_lon_1 = np.argmin(np.abs(lnd_lons - lonbounds[0]))
        lnd_lon_2 = np.argmin(np.abs(lnd_lons - lonbounds[1]))
        lndfrc = dataset_lndfrc.variables['sftlf'][lnd_lat_1: lnd_lat_2 +
                                                   1, lnd_lon_1: lnd_lon_2 + 1]
        lnd_lats = lnd_lats[lnd_lat_1: lnd_lat_2 + 1]
        lnd_lons = lnd_lons[lnd_lon_1: lnd_lon_2 + 1]
        # print(lndfrc)
        # print(var[0,:,:])
        temp = []
        lonsout, latsout = np.meshgrid(lnd_lons, lnd_lats)
        for idx in range(len(dtime)):
            # print(dtime[idx])
            tempinterp = basemap.interp(var[idx, :, :], lons, lats, lonsout, latsout, order=1)
            tempinterp = np.ma.masked_where(lndfrc < 50., tempinterp)
            temp.append(tempinterp)
        temp = np.ma.array(temp)
        var = temp
        # print(var[0,:,:].mask)
        lats = lnd_lats
        lons = lnd_lons

        return var, dtime, lats, lons

    if oceanmask == -1:
        dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc)
        lnd_lats = dataset_lndfrc.variables['lat'][:, 0]
        lnd_lons = dataset_lndfrc.variables['lon'][0, :]
        lnd_lat_1 = np.argmin(np.abs(lnd_lats - latbounds[0]))
        lnd_lat_2 = np.argmin(np.abs(lnd_lats - latbounds[1]))
        lnd_lon_1 = np.argmin(np.abs(lnd_lons - lonbounds[0]))
        lnd_lon_2 = np.argmin(np.abs(lnd_lons - lonbounds[1]))
        lndfrc = dataset_lndfrc.variables['sftlf'][lnd_lat_1: lnd_lat_2 +
                                                   1, lnd_lon_1: lnd_lon_2 + 1]
        lnd_lats = lnd_lats[lnd_lat_1: lnd_lat_2 + 1]
        lnd_lons = lnd_lons[lnd_lon_1: lnd_lon_2 + 1]
        # print(lndfrc)
        # print(var[0,:,:])
        temp = []
        lonsout, latsout = np.meshgrid(lnd_lons, lnd_lats)
        for idx in range(len(dtime)):
            # print(dtime[idx])
            tempinterp = basemap.interp(var[idx, :, :], lons, lats, lonsout, latsout, order=1)
            tempinterp = np.ma.masked_where(lndfrc < 50., tempinterp)
            temp.append(tempinterp)
        temp = np.ma.array(temp)
        var = temp
        # print(var[0,:,:].mask)
        lats = lnd_lats
        lons = lnd_lons

        return var, dtime, lats, lons

    return var, dtime, lats, lons


'''
# test on the function
varname = 'pr'
iniyear = 2000
endyear = 2005
project = 'SEA-22'
modelname = 'MPI-M-MPI-ESM-MR'
frequency = 'mon'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readcordex(varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0)
var[var.mask] = np.nan
print(lats)
print(lons)
# print(lons)
print(time)
print(var.shape)
# print(var[0, :, :])
'''
