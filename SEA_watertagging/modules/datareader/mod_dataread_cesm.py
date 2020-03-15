'''
#This is a function to read data from CORDEX-SEA data.

#Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as datetime
import cftime as ct

'''
This is a function used to read data from vrcesm

Description on parameters:
1) varname: the variable name in netCDF

2) iniyear/endyear: the time bounds

3) resolution: the regrided resolution

4) varfname: the variable name of netCDF file

5) case: the case name of experiement

6) fequency: monthly, daily or hourly data to read

7) latbounds/lonbounds: the horizontal boundaries of data

8) oceanmask: whether need to mask the ocean, default will not maks the ocean,
    0: do not mask any
    1: maks the ocean
   -1: mask the land

9) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def isleap(iyear):
    return (iyear % 4 == 0 and iyear % 100 != 0 or iyear % 400 == 0)


def readcesm(varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, **kwargs):

    # create time list to verify the data covers the whole period
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    frefname = {'mon': 'h0', 'day': 'h1', 'hr': 'h2'}

    fdir = '/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/'+case+'/atm/hist/'
    if frequency == 'day':
        fname = resolution+'_'+varfname+'_'+case+'.cam.'+frefname[frequency]+'.1979-2005.nc'
    else:
        fname = resolution+'_'+varfname+'_'+case+'.cam.'+frefname[frequency]+'.1919-2005.nc'

    # setup for land fraction
    dir_lndfrc = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
    flndfrc = {'fv02': 'USGS-gtopo30_0.23x0.31_remap_c061107.nc',
               'fv05': 'USGS-gtopo30_0.47x0.63_remap_c061106.nc',
               'fv09': 'USGS-gtopo30_0.9x1.25_remap_c051027.nc',
               'fv19': 'USGS-gtopo30_1.9x2.5_remap_c050602.nc'}

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    dataset = nc.Dataset(fdir+fname)
    time_var = dataset.variables['time']
#    dtime  = nc.num2date(time_var[:],units=time_var.units)
    cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    # print(cftime)
    dtime = []
    for istr in cftime:
        temp = istr.strftime('%Y-%m-%d %H:%M:%S')
        dtime.append(temp)
    dtime = pd.to_datetime(dtime)
    if frequency == 'mon':
        temp = []
        for idx, itime in enumerate(dtime):
            if (itime.month == 3)and(itime.day == 1)and(isleap(itime.year)):
                temp.append(itime - datetime.timedelta(days=2))
            else:
                temp.append(itime - datetime.timedelta(days=1))
        dtime = pd.to_datetime(temp)
    # print(dtime)

    select_dtime = (dtime >= inidate) & (dtime < enddate) & (
        ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
    dtime = dtime[select_dtime]
    # print(select_dtime)
    # print(dtime)

    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]

    # print(lats)
    # print(lons)

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

    var = dataset.variables[varname][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    if oceanmask == 1:
        dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc[resolution])
        lndfrc = dataset_lndfrc.variables['LANDFRAC'][lat_1: lat_2 + 1, lon_1: lon_2 + 1]
        for idx in range(len(dtime)):
            var[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var[idx, :, :])

    if oceanmask == -1:
        dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc[resolution])
        lndfrc = dataset_lndfrc.variables['LANDFRAC'][lat_1: lat_2 + 1, lon_1: lon_2 + 1]
        for idx in range(len(dtime)):
            var[idx, :, :] = np.ma.masked_where(lndfrc > 0.5, var[idx, :, :])

    return var, dtime, lats, lons


# read 3D variables
def readcesm_3D(varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, **kwargs):

    # create time list to verify the data covers the whole period
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    frefname = {'mon': 'h0', 'day': 'h1', 'hr': 'h2'}

    fdir = '/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/'+case+'/atm/hist/'
    if frequency == 'day':
        fname = resolution+'_'+varfname+'_'+case+'.cam.'+frefname[frequency]+'.1979-2005_vertical_interp.nc'
    else:
        fname = resolution+'_'+varfname+'_'+case+'.cam.'+frefname[frequency]+'.1919-2005_vertical_interp.nc'

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    dataset = nc.Dataset(fdir+fname)
    time_var = dataset.variables['time']
#    dtime  = nc.num2date(time_var[:],units=time_var.units)
    cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    # print(cftime)
    dtime = []
    for istr in cftime:
        temp = istr.strftime('%Y-%m-%d %H:%M:%S')
        dtime.append(temp)
    dtime = pd.to_datetime(dtime)
    if frequency == 'mon':
        temp = []
        for idx, itime in enumerate(dtime):
            if (itime.month == 3)and(itime.day == 1)and(isleap(itime.year)):
                temp.append(itime - datetime.timedelta(days=2))
            else:
                temp.append(itime - datetime.timedelta(days=1))
        dtime = pd.to_datetime(temp)
    # print(dtime)

    select_dtime = (dtime >= inidate) & (dtime < enddate) & (
        ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
    dtime = dtime[select_dtime]
    # print(select_dtime)
    # print(dtime)

    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    levs = dataset.variables['lev'][:]

    # print(lats)
    # print(lons)

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))

    var = dataset.variables[varname][select_dtime, :, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]
    var = var

    return var, dtime, levs, lats, lons


# read 3D variables at a certain level
def readcesm_3Dlevel(varname, iniyear, endyear, resolution, varfname, case, frequency, plev, latbounds, lonbounds, **kwargs):

    # create time list to verify the data covers the whole period
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    frefname = {'mon': 'h0', 'day': 'h1', 'hr': 'h2'}

    if 'model_level' in kwargs:
        model_level = kwargs['model_level']
    else:
        model_level = False

    fdir = '/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/'+case+'/atm/hist/'
    if model_level:
        fname = resolution+'_'+varfname+'_'+case+'.cam.'+frefname[frequency]+'.1919-2005.nc'
    else:
        fname = resolution+'_'+varfname+'_'+case+'.cam.'+frefname[frequency]+'.1919-2005_vertical_interp.nc'

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    dataset = nc.Dataset(fdir+fname)
    time_var = dataset.variables['time']
#    dtime  = nc.num2date(time_var[:],units=time_var.units)
    cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    # print(cftime)
    dtime = []
    for istr in cftime:
        temp = istr.strftime('%Y-%m-%d %H:%M:%S')
        dtime.append(temp)
    dtime = pd.to_datetime(dtime)
    if frequency == 'mon':
        temp = []
        for idx, itime in enumerate(dtime):
            if (itime.month == 3)and(itime.day == 1)and(isleap(itime.year)):
                temp.append(itime - datetime.timedelta(days=2))
            else:
                temp.append(itime - datetime.timedelta(days=1))
        dtime = pd.to_datetime(temp)
    # print(dtime)

    select_dtime = (dtime >= inidate) & (dtime < enddate) & (
        ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
    dtime = dtime[select_dtime]
    # print(select_dtime)
    # print(dtime)

    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    levs = dataset.variables['lev'][:]

    # print(lats)
    # print(lons)

    lat_1 = np.argmin(np.abs(lats - latbounds[0]))
    lat_2 = np.argmin(np.abs(lats - latbounds[1]))
    lon_1 = np.argmin(np.abs(lons - lonbounds[0]))
    lon_2 = np.argmin(np.abs(lons - lonbounds[1]))
    if model_level:
        lev = plev
    else:
        lev = np.argmin(np.abs(levs - plev))

    var = dataset.variables[varname][select_dtime, lev, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]
    var = var

    return var, dtime, levs, lats, lons


'''
# test on the function
varname = 'TREFHT'
iniyear = 2000
endyear = 2005
resolution = 'fv19'
varfname = 'TS'
case = 'f19_f19_AMIP_1979_to_2005'
frequency = 'mon'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readvrcesm(varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1)
var[var.mask] = np.nan
print(lats)
print(lons)
print(time)
print(var.shape)
print(var[0, :, :])
print(np.mean(np.mean(var, axis=1), axis=1))
'''
