'''
# This is a function to read precip from TRMM

# Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as datetime


################################################################################################
# read monthly data
################################################################################################

'''
This is a function used to read obs data

Description on parameters:
1) varname: the name of variables

2) iniyear/endyear: the time bounds

3) latbounds/lonbounds: the horizontal boundaries of data

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_pre_TRMM_mon(varname, iniyear, endyear, latbounds, lonbounds, oceanmask=0):

    # set up directories dictionary
    dir = '/scratch/d/dylan/harryli/obsdataset/TRMM/monthly/'

    # setup for land fraction
    dir_lndfrc = '/scratch/d/dylan/harryli/obsdataset/TRMM/'
    flndfrc = 'TMPA_mask.nc'

    # file name
    fname = 'TRMM_3B43_monthly_V7_199801_200512.nc'

    factors = {'IRprecipitation_cnt': 1.,
               'precipitation_cnt': 1.,
               'HQprecipitation_cnt': 1.,
               'IRprecipitation': 24.,
               'precipitation': 24.,
               'HQprecipitation': 24.,
               'randomError': 1.,
               'randomError_cnt': 1.}

    # set up variable names
    varnames = ['IRprecipitation_cnt', 'IRprecipitation', 'precipitation', 'precipitation_cnt',
                'HQprecipitation', 'HQprecipitation_cnt', 'randomError', 'randomError_cnt']
    # set up variable name of lat/lon
    latname = 'nlat'
    lonname = 'nlon'

    # set up initial/end date
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    # create time series for TRMM data
    dtime = pd.date_range(start='1998-01-01', end='2005-12-01', freq='MS')

    # open data file
    dataset = nc.Dataset(dir+fname)

    select_dtime = (dtime >= inidate) & (dtime < enddate)
    dtime = dtime[select_dtime]
    # print(dtime)
    # print(select_dtime)

    lats = dataset.variables[latname][:]
    lons = dataset.variables[lonname][:]

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

    if varname not in varnames:
        print('Variable does not exit')
        return 0
    else:
        var = dataset.variables[varname][select_dtime, lon_1: lon_2 + 1, lat_1: lat_2 + 1]
        var = var * factors[varname]

        # swap lon/lat
        var = np.swapaxes(var, 1, 2)

    # mask the ocean/land
    if oceanmask == 1:
        dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc)   # read water converage percentage
        lats_lndfrc = dataset_lndfrc.variables['lat'][:] + 0.125
        lons_lndfrc = dataset_lndfrc.variables['lon'][:]
        lat_lnd_1 = np.argmin(np.abs(lats_lndfrc - latbounds[0]))
        lat_lnd_2 = np.argmin(np.abs(lats_lndfrc - latbounds[1]))
        lon_lnd_1 = np.argmin(np.abs(lons_lndfrc - lonbounds[0]))
        lon_lnd_2 = np.argmin(np.abs(lons_lndfrc - lonbounds[1]))
        lndfrc = dataset_lndfrc.variables['landseamask'][lat_lnd_1: lat_lnd_2 + 1, lon_lnd_1: lon_lnd_2 + 1]
        for idx in range(len(dtime)):
            var[idx, :, :] = np.ma.masked_where(lndfrc > 50., var[idx, :, :])

    if oceanmask == -1:
        dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc)
        lats_lndfrc = dataset_lndfrc.variables['lat'][:] + 0.125
        lons_lndfrc = dataset_lndfrc.variables['lon'][:]
        lat_lnd_1 = np.argmin(np.abs(lats_lndfrc - latbounds[0]))
        lat_lnd_2 = np.argmin(np.abs(lats_lndfrc - latbounds[1]))
        lon_lnd_1 = np.argmin(np.abs(lons_lndfrc - lonbounds[0]))
        lon_lnd_2 = np.argmin(np.abs(lons_lndfrc - lonbounds[1]))
        lndfrc = dataset_lndfrc.variables['landseamask'][lat_lnd_1: lat_lnd_2 + 1, lon_lnd_1: lon_lnd_2 + 1]
        for idx in range(len(dtime)):
            var[idx, :, :] = np.ma.masked_where(lndfrc < 50., var[idx, :, :])

    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    return var, dtime, lats, lons


################################################################################################
# read daily data
################################################################################################

'''
This is a function used to read obs data

Description on parameters:
1) varname: the name of variables

2) iniyear/endyear: the time bounds

3) latbounds/lonbounds: the horizontal boundaries of data

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_pre_TRMM_day(varname, iniyear, endyear, latbounds, lonbounds, oceanmask=0, **kwargs):

    # set up directories dictionary
    dir = '/scratch/d/dylan/harryli/obsdataset/TRMM/daily/'

    # setup for land fraction
    dir_lndfrc = '/scratch/d/dylan/harryli/obsdataset/TRMM/'
    flndfrc = 'TMPA_mask.nc'

    # file name
    fname = 'TRMM_3B42_daily_19980101_20051231.nc'

    factors = {'IRprecipitation_cnt': 1.,
               'precipitation_cnt': 1.,
               'HQprecipitation_cnt': 1.,
               'IRprecipitation': 1.,
               'precipitation': 1.,
               'HQprecipitation': 1.,
               'randomError': 1.,
               'randomError_cnt': 1.}

    varnames = ['IRprecipitation_cnt', 'IRprecipitation', 'precipitation', 'precipitation_cnt',
                'HQprecipitation', 'HQprecipitation_cnt', 'randomError', 'randomError_cnt']

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    # set up variable name of lat/lon
    latname = 'lat'
    lonname = 'lon'

    # set up initial/end date
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    # create time series for TRMM data
    dtime = pd.date_range(start='1998-01-01', end='2005-12-31', freq='D')

    dataset = nc.Dataset(dir+fname)

    select_dtime = (dtime >= inidate) & (dtime < enddate) & (
        ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
    dtime = dtime[select_dtime]
    # print(dtime)
    # print(select_dtime)

    lats = dataset.variables[latname][:]
    lons = dataset.variables[lonname][:]

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

    if varname not in varnames:
        print('Variable does not exit')
        return 0
    else:
        var = dataset.variables[varname][select_dtime, lon_1: lon_2 + 1, lat_1: lat_2 + 1]
        var = var * factors[varname]

        # swap lon/lat
        var = np.swapaxes(var, 1, 2)

    # mask the ocean/land
    if oceanmask == 1:
        dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc)   # read water converage percentage
        lats_lndfrc = dataset_lndfrc.variables['lat'][:] + 0.125
        lons_lndfrc = dataset_lndfrc.variables['lon'][:]
        lat_lnd_1 = np.argmin(np.abs(lats_lndfrc - latbounds[0]))
        lat_lnd_2 = np.argmin(np.abs(lats_lndfrc - latbounds[1]))
        lon_lnd_1 = np.argmin(np.abs(lons_lndfrc - lonbounds[0]))
        lon_lnd_2 = np.argmin(np.abs(lons_lndfrc - lonbounds[1]))
        lndfrc = dataset_lndfrc.variables['landseamask'][lat_lnd_1: lat_lnd_2 + 1, lon_lnd_1: lon_lnd_2 + 1]
        # print(lats_lndfrc[lat_lnd_1: lat_lnd_2 + 1])

        # for idx in range(len(dtime)):
        #     var[idx, :, :] = np.ma.masked_where(lndfrc < 50., var[idx, :, :])

        block_ts = []
        for idx in range(len(dtime)):
            # print(idx)
            if np.sum(var[idx, :, :].mask) > 0:
                print('time step '+str(idx)+': '+dtime[idx].strftime('%Y-%m-%d')+' has a original mask, will be blocked')
                block_ts.append(idx)

        if len(block_ts) > 0:
            # print(len(block_ts))
            var = np.delete(var, block_ts, 0)
            dtime = np.delete(dtime, block_ts, 0)

        # print(lndfrc > 50.)
        lndfrc_3d = np.broadcast_to(lndfrc > 50., var.shape)
        # print(np.sum(var[6, :, :].mask))
        var = np.ma.masked_where(lndfrc_3d, var)

    if oceanmask == -1:
        dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc)
        lats_lndfrc = dataset_lndfrc.variables['lat'][:] + 0.125
        lons_lndfrc = dataset_lndfrc.variables['lon'][:]
        lat_lnd_1 = np.argmin(np.abs(lats_lndfrc - latbounds[0]))
        lat_lnd_2 = np.argmin(np.abs(lats_lndfrc - latbounds[1]))
        lon_lnd_1 = np.argmin(np.abs(lons_lndfrc - lonbounds[0]))
        lon_lnd_2 = np.argmin(np.abs(lons_lndfrc - lonbounds[1]))
        lndfrc = dataset_lndfrc.variables['landseamask'][lat_lnd_1: lat_lnd_2 + 1, lon_lnd_1: lon_lnd_2 + 1]
        # print('do it')
        # for idx in range(len(dtime)):
        #     var[idx, :, :] = np.ma.masked_where(lndfrc < 50., var[idx, :, :])

        block_ts = []
        for idx in range(len(dtime)):
            # print(idx)
            if np.sum(var[idx, :, :].mask) > 0:
                print('time step '+str(idx)+': '+dtime[idx].strftime('%Y-%m-%d')+' has a original mask, will be blocked')
                block_ts.append(idx)
            # var[idx, :, :] = np.ma.masked_where(lndfrc > 50., var[idx, :, :])

        if len(block_ts) > 0:
            # print(len(block_ts))
            var = np.delete(var, block_ts, 0)
            dtime = np.delete(dtime, block_ts, 0)

        lndfrc_3d = np.broadcast_to(lndfrc < 50., var.shape)
        # print(np.sum(var[6, :, :].mask))
        var = np.ma.masked_where(lndfrc_3d, var)

    lats = lats[lat_1: lat_2 + 1]
    # print(lats)
    lons = lons[lon_1: lon_2 + 1]

    # print(np.argwhere(np.logical_xor(var[5, :, :].mask, lndfrc > 50.)))

    return var, dtime, lats, lons


################################################################################################
# test function
################################################################################################

'''
# test on the function
iniyear = 2000
endyear = 2005
varname = 'precipitation'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_pre_TRMM_mon(varname, iniyear, endyear, latbounds, lonbounds)
print(lats)
print(lons)
print(time)
print(var.shape)
print(np.mean(np.mean(var, axis=1), axis=1))
'''

'''
# test on the function
iniyear = 1998
endyear = 2005
varname = 'precipitation'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_pre_TRMM_day(varname, iniyear, endyear, latbounds, lonbounds, ignore_years=[1999])
print(lats)
print(lons)
print(time)
print(var.shape)
print(np.mean(np.mean(var, axis=1), axis=1))
'''
