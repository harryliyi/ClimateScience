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
# read temperature data
################################################################################################

'''
This is a function used to read obs data

Description on parameters:
1) varname: the name of variables

2) iniyear/endyear: the time bounds

3) frequency: either 'day' or 'mon'

4) latbounds/lonbounds: the horizontal boundaries of data

5) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_tmp_CPC(varname, iniyear, endyear, frequency, latbounds, lonbounds, **kwargs):

    # create time list to verify the data covers the whole period
    yearts = range(iniyear, endyear+1, 1)
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')
    monthts = pd.date_range(start=str(iniyear)+'-01-01', end=str(endyear)+'-12-01', freq='MS')

    # accepted variables
    varnames = ['tasavg', 'tasmax', 'tasmin']

    # set up directories dictionary
    fdir = '/scratch/d/dylan/harryli/obsdataset/CPC/global_temp_daily/'

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    # file name
    fname = 'tmax.1980.nc'
    dataset = nc.Dataset(fdir+fname)

    # set up variable name of lat/lon
    latname = 'lat'
    lonname = 'lon'

    lats = dataset.variables[latname][:]
    lons = dataset.variables[lonname][:]

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

    lat_swap = False
    lon_swap = False

    if lat_1 > lat_2:
        lat_swap = True
        temp = lat_1
        lat_1 = lat_2
        lat_2 = temp
    if lon_1 > lon_2:
        lon_swap = True
        temp = lon_1
        lon_1 = lon_2
        lon_2 = temp

    if varname not in varnames:
        print('Variable does not exit')
        return -1, -1, -1, -1
    else:
        if varname == 'tasmax':
            fnames = [fdir+'tmax.'+str(iyear)+'.nc' for iyear in yearts]
            dataset = nc.MFDataset(fnames)

            # read time and select the time
            time_var = dataset.variables['time']
            cftime = nc.num2date(time_var[:], units=time_var.units)
            # print(cftime)
            dtime = []
            for istr in cftime:
                temp = istr.strftime('%Y-%m-%d %H:%M:%S')
                dtime.append(temp)
            dtime = pd.to_datetime(dtime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
            # print(dtime)

            select_dtime = (dtime >= inidate) & (dtime < enddate) & (
                ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
            dtime = dtime[select_dtime]

            # read the variable
            var = dataset.variables['tmax'][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]

        if varname == 'tasmin':
            fnames = [fdir+'tmin.'+str(iyear)+'.nc' for iyear in yearts]
            dataset = nc.MFDataset(fnames)

            # read time and select the time
            time_var = dataset.variables['time']
            cftime = nc.num2date(time_var[:], units=time_var.units)
            dtime = []
            for istr in cftime:
                temp = istr.strftime('%Y-%m-%d %H:%M:%S')
                dtime.append(temp)
            dtime = pd.to_datetime(dtime, format='%Y-%m-%d %H:%M:%S', errors='coerce')

            select_dtime = (dtime >= inidate) & (dtime < enddate) & (
                ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
            dtime = dtime[select_dtime]

            # read the variable
            var = dataset.variables['tmin'][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]

        if varname == 'tasavg':
            fnames = [fdir+'tmin.'+str(iyear)+'.nc' for iyear in yearts]
            dataset = nc.MFDataset(fnames)

            # read time and select the time
            time_var = dataset.variables['time']
            cftime = nc.num2date(time_var[:], units=time_var.units)
            dtime = []
            for istr in cftime:
                temp = istr.strftime('%Y-%m-%d %H:%M:%S')
                dtime.append(temp)
            dtime = pd.to_datetime(dtime, format='%Y-%m-%d %H:%M:%S', errors='coerce')

            select_dtime = (dtime >= inidate) & (dtime < enddate) & (
                ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
            dtime = dtime[select_dtime]

            # read the tmin
            var1 = dataset.variables['tmin'][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]

            # read tmax
            fnames = [fdir+'tmax.'+str(iyear)+'.nc' for iyear in yearts]
            dataset = nc.MFDataset(fnames)
            var2 = dataset.variables['tmax'][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
            # get tavg from the mean of max and min
            var = (var1 + var2)/2.

    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    if lat_swap:
        var = var[:, ::-1, :]
        lats = lats[::-1]

    if lon_swap:
        var = var[:, :, ::-1]
        lons = lons[::-1]

    if frequency == 'day':
        return var, dtime, lats, lons

    if frequency == 'mon':
        mon_var = []
        for idx, imon in enumerate(monthts):
            select_dates = ((dtime.year == imon.year) & (dtime.month == imon.month))
            if varname == 'tasmax':
                mon_var.append(np.ma.mean(var[select_dates, :, :], axis=0))
            if varname == 'tasmin':
                mon_var.append(np.ma.mean(var[select_dates, :, :], axis=0))
            if varname == 'tasavg':
                mon_var.append(np.ma.mean(var[select_dates, :, :], axis=0))

            # print(dtime[select_dates])

        mon_var = np.ma.array(mon_var)

        return mon_var, monthts, lats, lons

################################################################################################
# read precipitation data
################################################################################################


'''
This is a function used to read obs data

Description on parameters:
1) varname: the name of variables

2) iniyear/endyear: the time bounds

3) frequency: either 'day' or 'mon'

4) latbounds/lonbounds: the horizontal boundaries of data

5) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_pre_CPC(varname, iniyear, endyear, frequency, latbounds, lonbounds, **kwargs):

    # create time list to verify the data covers the whole period
    yearts = range(iniyear, endyear+1, 1)
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')
    monthts = pd.date_range(start=str(iniyear)+'-01-01', end=str(endyear)+'-12-01', freq='MS')

    # accepted variables
    varnames = ['precip', 'pre', 'prect', 'precc']
    # print(varname)

    # set up directories dictionary
    fdir = '/scratch/d/dylan/harryli/obsdataset/CPC/global_precip_daily/'

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    # file name
    fname = 'precip.1980.nc'
    dataset = nc.Dataset(fdir+fname)

    # set up variable name of lat/lon
    latname = 'lat'
    lonname = 'lon'

    lats = dataset.variables[latname][:]
    lons = dataset.variables[lonname][:]

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

    lat_swap = False
    lon_swap = False

    if lat_1 > lat_2:
        lat_swap = True
        temp = lat_1
        lat_1 = lat_2
        lat_2 = temp
    if lon_1 > lon_2:
        lon_swap = True
        temp = lon_1
        lon_1 = lon_2
        lon_2 = temp

    if varname not in varnames:
        print('Variable does not exit')
        return -1, -1, -1, -1
    else:
        fnames = [fdir+'precip.'+str(iyear)+'.nc' for iyear in yearts]
        dataset = nc.MFDataset(fnames)

        # read time and select the time
        time_var = dataset.variables['time']
        cftime = nc.num2date(time_var[:], units=time_var.units)
        # print(cftime)
        dtime = []
        for istr in cftime:
            temp = istr.strftime('%Y-%m-%d %H:%M:%S')
            dtime.append(temp)
        dtime = pd.to_datetime(dtime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # print(dtime)

        select_dtime = (dtime >= inidate) & (dtime < enddate) & (
            ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
        dtime = dtime[select_dtime]

        # read the variable
        var = dataset.variables['precip'][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]

    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    if lat_swap:
        var = var[:, ::-1, :]
        lats = lats[::-1]

    if lon_swap:
        var = var[:, :, ::-1]
        lons = lons[::-1]

    if frequency == 'day':
        return var, dtime, lats, lons

    if frequency == 'mon':
        mon_var = []
        for idx, imon in enumerate(monthts):
            select_dates = ((dtime.year == imon.year) & (dtime.month == imon.month))
            mon_var.append(np.ma.mean(var[select_dates, :, :], axis=0))

        mon_var = np.ma.array(mon_var)

        return mon_var, monthts, lats, lons

################################################################################################
# test function
################################################################################################


'''
# test on the function
iniyear = 2000
endyear = 2005
varname = 'tasmax'
frequency = 'mon'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_tmp_CPC(varname, iniyear, endyear, frequency, latbounds, lonbounds)
print(lats)
print(lons)
print(time)
print(var.shape)
print(var[0, :, :])
print(np.ma.mean(np.ma.mean(var, axis=1), axis=1))
'''

'''
# test on the function
iniyear = 2000
endyear = 2005
varname = 'precip'
frequency = 'mon'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_pre_CPC(varname, iniyear, endyear, frequency, latbounds, lonbounds)
print(lats)
print(lons)
print(time)
print(var.shape)
print(var[0, :, :])
print(np.ma.mean(np.ma.mean(var, axis=1), axis=1))
'''
