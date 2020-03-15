'''
# This is a function to read precip from TRMM

# Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as datetime
import calendar


################################################################################################
# read temperature data
################################################################################################

'''
This is a function used to read obs data

Description on parameters:
1) varname: the name of variables

2) iniyear/endyear: the time bounds

3) latbounds/lonbounds: the horizontal boundaries of data

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_tmp_CRU(varname, iniyear, endyear,  latbounds, lonbounds, **kwargs):

    # accepted variables
    varnames = {'tasavg': 'tmp',
                'tasmax': 'tmx',
                'tasmin': 'tmn'}

    if varname not in varnames:
        print('Error: Variable does not exit!')
        return -1, -1, -1, -1

    # create time list to verify the data covers the whole period
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    # set up directories dictionary
    fdir = '/scratch/d/dylan/harryli/obsdataset/CRU/tmp/'

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    # file name
    fnames = {'tasavg': 'cru_ts4.03.1901.2018.tmp.dat.nc',
              'tasmax': 'cru_ts4.03.1901.2018.tmx.dat.nc',
              'tasmin': 'cru_ts4.03.1901.2018.tmn.dat.nc'}
    dataset = dataset = nc.Dataset(fdir+fnames[varname])

    # set up variable name of lat/lon
    latname = 'lat'
    lonname = 'lon'

    lats = dataset.variables[latname][:]
    lons = dataset.variables[lonname][:]

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

    # read time and select the time
    time_var = dataset.variables['time']
    if hasattr(time_var, 'calendar'):
        cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    else:
        cftime = nc.num2date(time_var[:], units=time_var.units)
    dtime = []
    for istr in cftime:
        temp = istr.strftime('%Y-%m-%d %H:%M:%S')
        dtime.append(temp)
    dtime = pd.to_datetime(dtime)
    select_dtime = (dtime >= inidate) & (dtime < enddate) & (~ np.in1d(dtime.year, ignore_years))
    dtime = dtime[select_dtime]

    # read the variable
    var = dataset.variables[varnames[varname]][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]

    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    var = np.ma.array(var)

    return var, dtime, lats, lons

################################################################################################
# read precipitation data
################################################################################################


'''
This is a function used to read obs data

Description on parameters:
1) varname: the name of variables

2) iniyear/endyear: the time bounds

3) latbounds/lonbounds: the horizontal boundaries of data

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_pre_CRU(varname, iniyear, endyear, latbounds, lonbounds, **kwargs):

    # accepted variables
    varnames = {'precip': 'pre',
                'pre': 'pre',
                'prect': 'pre',
                'precc': 'pre'}

    if varname not in varnames:
        print('Error: Variable does not exit!')
        return -1, -1, -1, -1

    # create time list to verify the data covers the whole period
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    # set up directories dictionary
    fdir = '/scratch/d/dylan/harryli/obsdataset/CRU/pre/'

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    # file name
    fnames = {'pre': 'cru_ts4.03.1901.2018.pre.dat.nc',
              'precip': 'cru_ts4.03.1901.2018.pre.dat.nc',
              'prect': 'cru_ts4.03.1901.2018.pre.dat.nc',
              'precc': 'cru_ts4.03.1901.2018.pre.dat.nc'}
    dataset = nc.Dataset(fdir+fnames[varname])

    # set up variable name of lat/lon
    latname = 'lat'
    lonname = 'lon'

    lats = dataset.variables[latname][:]
    lons = dataset.variables[lonname][:]

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

    # read time and select the time
    time_var = dataset.variables['time']
    if hasattr(time_var, 'calendar'):
        cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    else:
        cftime = nc.num2date(time_var[:], units=time_var.units)
    dtime = []
    for istr in cftime:
        temp = istr.strftime('%Y-%m-%d %H:%M:%S')
        dtime.append(temp)
    dtime = pd.to_datetime(dtime)
    select_dtime = (dtime >= inidate) & (dtime < enddate) & (~ np.in1d(dtime.year, ignore_years))
    dtime = dtime[select_dtime]

    # read the variable
    var = dataset.variables[varnames[varname]][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]

    for idx in np.arange(len(dtime)):
        var[idx, :，:] = var[idx, :，:] / calendar.monthrange(dtime[idx].year, dtime[idx].month)[1]

    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    var = np.ma.array(var)

    return var, dtime, lats, lons

################################################################################################
# test function
################################################################################################


'''
# test on the function
iniyear = 2000
endyear = 2005
varname = 'tasmax'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_tmp_CRU(varname, iniyear, endyear, latbounds, lonbounds)
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

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_pre_CRU(varname, iniyear, endyear, latbounds, lonbounds)
print(lats)
print(lons)
print(time)
print(var.shape)
print(var[0, :, :])
print(np.ma.mean(np.ma.mean(var, axis=1), axis=1))
'''
