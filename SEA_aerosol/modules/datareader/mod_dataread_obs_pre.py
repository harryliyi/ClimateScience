'''
# This is a function to read precip from CRU, GPCP, GPCC, ERA-interim, APHRODITE

# Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as datetime
from calendar import monthrange


################################################################################################
# read monthly data
################################################################################################

'''
This is a function used to read obs data

Description on parameters:
1) project: the name of dataset

2) iniyear/endyear: the time bounds

3) latbounds/lonbounds: the horizontal boundaries of data

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_pre_mon(project, iniyear, endyear, latbounds, lonbounds, oceanmask=0):

    # set up directories dictionary
    dirs = {'CRU': '/scratch/d/dylan/harryli/obsdataset/CRU/',
            'GPCP': '/scratch/d/dylan/harryli/obsdataset/GPCP/',
            'GPCC': '/scratch/d/dylan/harryli/obsdataset/GPCC/',
            'ERA-interim': '/scratch/d/dylan/harryli/obsdataset/ERA_interim/pre/monthly/',
            'APHRODITE': '/scratch/d/dylan/harryli/obsdataset/APHRODITE/MA/'}

    maskdirs = {'ERA-interim': '/scratch/d/dylan/harryli/obsdataset/ERA_interim/'}

    maskfnames = {'ERA-interim': 'era_interim_landsea_mask.nc'}

    maskvarnames = {'ERA-interim': 'lsm'}

    fnames = {'CRU': 'cru_ts3.21.1901.2012.pre.dat.nc',
              'GPCP': 'gpcp_cdr_v23rB1_197901-201608.nc',
              'GPCC': 'precip.mon.total.v7.nc',
              'ERA-interim': 'era_interim_pre_monthly_197901-200512.nc',
              'APHRODITE': 'APHRO_MA_025deg_V1101.1951-2007.nc'}

    varnames = {'CRU': 'pre',
                'GPCP': 'precip',
                'GPCC': 'precip',
                'ERA-interim': 'tp',
                'APHRODITE': 'precip'}

    latnames = {'CRU': 'lat',
                'GPCP': 'latitude',
                'GPCC': 'lat',
                'ERA-interim': 'latitude',
                'APHRODITE': 'latitude'}

    lonnames = {'CRU': 'lon',
                'GPCP': 'longitude',
                'GPCC': 'lon',
                'ERA-interim': 'longitude',
                'APHRODITE': 'longitude'}

    factors = {'CRU': 1.0,
               'GPCP': 1.0,
               'GPCC': 1.0,
               'ERA-interim': 1000.,
               'APHRODITE': 1.0}

    mdays = []

    yearts = range(iniyear, endyear+1, 1)
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    dataset = nc.Dataset(dirs[project]+fnames[project])

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
    select_dtime = (dtime >= inidate) & (dtime < enddate)
    dtime = dtime[select_dtime]
    # print(dtime)
    # print(select_dtime)

    lats = dataset.variables[latnames[project]][:]
    lons = dataset.variables[lonnames[project]][:]

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

    var = dataset.variables[varnames[project]][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
    var = var * factors[project]
    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    if (project == 'ERA-interim') & (oceanmask == 1):
        lsmdata = nc.Dataset(maskdirs[project]+maskfnames[project])
        lsmask = lsmdata.variables[maskvarnames[project]][0, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
        lsmmask_3d = np.broadcast_to(lsmask < 0.5, var.shape)
        var = np.ma.masked_where(lsmmask_3d, var)

    if lat_swap:
        var = var[:, ::-1, :]
        lats = lats[::-1]

    if lon_swap:
        var = var[:, :, ::-1]
        lons = lons[::-1]

    if project == 'GPCC':
        ndays = [monthrange(idate.year, idate.month)[1] for idate in dtime]
        # print(ndays)
        for idx in range(len(dtime)):
            var[idx, :, :] = var[idx, :, :]/ndays[idx]

    if project == 'APHRODITE':
        temp = np.ma.empty(((endyear-iniyear+1)*12, var.shape[1], var.shape[2]))
        temptime = []
        count = 0
        for iyear in range(iniyear, endyear+1):
            for imon in range(12):
                # print(dtime[(dtime.month==(imon+1))&(dtime.year==iyear)])
                temp[count, :, :] = np.ma.mean(
                    var[(dtime.month == (imon+1)) & (dtime.year == iyear), :, :], axis=0)
                count = count + 1
                temptime.append(datetime.datetime.strptime(
                    str(iyear)+'-'+str(imon+1)+'-16', '%Y-%m-%d'))
        var = np.ma.array(temp)
        dtime = pd.to_datetime(temptime)

    return var, dtime, lats, lons


################################################################################################
# read daily data
################################################################################################

'''
This is a function used to read obs data

Description on parameters:
1) project: the name of dataset

2) iniyear/endyear: the time bounds

3) latbounds/lonbounds: the horizontal boundaries of data

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''


def readobs_pre_day(project, iniyear, endyear, latbounds, lonbounds, **kwargs):

    # set up directories dictionary
    dirs = {'ERA-interim': '/scratch/d/dylan/harryli/obsdataset/ERA_interim/pre/daily/',
            'APHRODITE': '/scratch/d/dylan/harryli/obsdataset/APHRODITE/MA/',
            'CPC': '/scratch/d/dylan/harryli/obsdataset/CPC/global_precip_daily/'}

    fnames = {'ERA-interim': 'era_interim_tp_daily_19790101-20051231.nc',
              'APHRODITE': 'APHRO_MA_025deg_V1101.1951-2007.nc',
              'CPC': 'precip.'}

    varnames = {'ERA-interim': 'tp',
                'APHRODITE': 'precip',
                'CPC': 'precip'}

    latnames = {'ERA-interim': 'latitude',
                'APHRODITE': 'latitude',
                'CPC': 'lat'}

    lonnames = {'ERA-interim': 'longitude',
                'APHRODITE': 'longitude',
                'CPC': 'lon'}

    factors = {'ERA-interim': 1000.,
               'APHRODITE': 1.0,
               'CPC': 1.0}

    yearts = range(iniyear, endyear+1, 1)
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    if project == 'CPC':
        fyears = [dirs[project]+fnames[project]+str(iyear)+'.nc' for iyear in yearts]
        dataset = nc.MFDataset(fyears)
    else:
        dataset = nc.Dataset(dirs[project]+fnames[project])

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
    select_dtime = (dtime >= inidate) & (dtime < enddate) & (
        ~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))
    dtime = dtime[select_dtime]
    # print(dtime)
    # print(select_dtime)

    lats = dataset.variables[latnames[project]][:]
    lons = dataset.variables[lonnames[project]][:]

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

    var = dataset.variables[varnames[project]][select_dtime, lat_1: lat_2 + 1, lon_1: lon_2 + 1]
    var = var * factors[project]
    lats = lats[lat_1: lat_2 + 1]
    lons = lons[lon_1: lon_2 + 1]

    if lat_swap:
        var = var[:, ::-1, :]
        lats = lats[::-1]

    if lon_swap:
        var = var[:, :, ::-1]
        lons = lons[::-1]

    return var, dtime, lats, lons


################################################################################################
# read SA-OBS
################################################################################################

'''
This is a function used to read SA-OBS data

Description on parameters:
1) project: the name of dataset

2) iniyear/endyear: the time bounds

3) countries: the list of countries

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''
# create a function to convert degree/minute/second to decimal


def deg2dec(x):
    xlist = x.split(":")
    if xlist[0][0] == "-":
        xdec = float(xlist[0])-float(xlist[1])/60.-float(xlist[2])/60./60.
    else:
        xdec = float(xlist[0])+float(xlist[1])/60.+float(xlist[2])/60./60.
    return xdec


def read_SAOBS_pre(version, iniyear, endyear, countries, missing_ratio, **kwargs):

    # set up SA-OBS pre observations directory filename
    stnsum = "stations.txt"

    countries_dict = {'Thailand': 'TH', 'Vietnam': 'VN', 'Cambodia': 'KH', 'Myanmar': 'MM',
                      'Malaysia': 'MY', 'Philippines': 'PH', 'Indonesia': 'ID', 'Singapore': 'SG'}

    countryids = [countries_dict[icountry] for icountry in countries]

    dirs = {'countries': '/scratch/d/dylan/harryli/obsdataset/SA_OBS/countries/',
            'v1': '/scratch/d/dylan/harryli/obsdataset/SA_OBS/v1/SACA_blend_rr',
            'v2': '/scratch/d/dylan/harryli/obsdataset/SA_OBS/v2/SACA_blend_all'}

    yearts = range(iniyear, endyear+1, 1)
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')
    # print(inidate)

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    # find the stn ids from stations.txt
    fopen = open(dirs[version]+stnsum, "r")
    linereader = fopen.readlines()[19:]

    stnids = np.array([], dtype=int)
    stnnames = np.array([])
    countrynames = np.array([])
    stnlats = np.array([], dtype=float)
    stnlons = np.array([], dtype=float)
    stnhgts = np.array([], dtype=float)
    stnmiss = np.array([], dtype=float)

    # read each line to get a station info
    for lineno in range(len(linereader)):
        linelist = linereader[lineno].split(",")

        if linelist[2] in countryids:
            stnids = np.append(stnids, int(linelist[0]))
            stnnames = np.append(stnnames, " ".join(x for x in linelist[1].split()))
            countrynames = np.append(countrynames, list(countries_dict.keys())[
                                     list(countries_dict.values()).index(linelist[2])])
            stnlats = np.append(stnlats, deg2dec(linelist[3]))
            stnlons = np.append(stnlons, deg2dec(linelist[4]))
            stnhgts = np.append(stnhgts, float(linelist[5]))

        print("Current station "+linelist[0]+" is "+" ".join(x for x in linelist[1].split()) +
              " in "+linelist[2]+" at "+str(deg2dec(linelist[3]))+"/"+str(deg2dec(linelist[4])))

    dataset = {}
    var = []

    # read precip data from each station
    print("Totally "+str(len(stnids))+" stations are found. Their information is shown in following:")
    for idx in range(len(stnids)):
        print("station "+str(idx+1)+"/"+str(len(stnids))+" is: "+stnnames[idx]+" in "+countrynames[idx]+" at "+str(
            stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6))

        # open file from each station
        obsfname = "RR_STAID"+str(stnids[idx]).zfill(6)+".txt"

        currstn = pd.read_csv(dirs[version]+obsfname, sep=',', usecols=[1, 2, 3], names=[
                              'Date', 'precip', 'Q_RR'], dtype={'precip': np.float64, 'Q_RR': np.int32}, header=None, skiprows=20)
        currstn['Date'] = pd.to_datetime(currstn['Date'], format='%Y%m%d')

        currstn = currstn[(currstn['Date'] >= inidate) & (currstn['Date'] < enddate) & (
            ~((currstn['Date'].dt.month == 2) & (currstn['Date'].dt.day == 29))) & (~ np.in1d(currstn['Date'].dt.year, ignore_years))]
        currstn['precip'] = currstn['precip']/10.
        # print(currstn.count())
        currstn.loc[(currstn['Q_RR'] == 9) | (currstn['Q_RR'] == 1), 'precip'] = np.NaN

        print("Current missing ratio is "+str(100.-100. *
                                              currstn.count()['precip']/(currstn.count()['Date']))+'%')
        stnmiss = np.append(stnmiss, 100.-100.*currstn.count()['precip']/(currstn.count()['Date']))
        if stnmiss[-1] < missing_ratio:
            dataset[stnnames[idx]] = currstn['precip'].values
            var.append(currstn['precip'].values)

    dataset['Date'] = currstn['Date'].values
    dataset = pd.DataFrame.from_dict(dataset)
    dataset = dataset.set_index(['Date'])

    stnids = stnids[stnmiss < missing_ratio]
    stnnames = stnnames[stnmiss < missing_ratio]
    countrynames = countrynames[stnmiss < missing_ratio]
    stnlats = stnlats[stnmiss < missing_ratio]
    stnlons = stnlons[stnmiss < missing_ratio]
    stnhgts = stnhgts[stnmiss < missing_ratio]
    stnmiss = stnmiss[stnmiss < missing_ratio]
    var = np.ma.array(var)

    print('Totally '+str(len(stnids))+' stations whose missing ratio is less than ' +
          str(missing_ratio)+'%. Their information is shown in following:')
    for idx in range(len(stnids)):
        print('station '+str(idx+1)+'/'+str(len(stnids))+' is: '+stnnames[idx]+' in '+countrynames[idx]+' at '+str(stnlats[idx])+'/'+str(
            stnlons[idx])+' with an station id '+str(stnids[idx]).zfill(6)+', data missing ratio is '+str(stnmiss[idx])+'%')

    return dataset, var, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss

################################################################################################
# test function
################################################################################################


'''
# test on the function
iniyear = 2000
endyear = 2005
project = 'CRU'

# define the contour plot region
latbounds = [10, 25]
lonbounds = [100, 110]

var, time, lats, lons = readobs_pre_mon(project, iniyear, endyear, latbounds, lonbounds)
print(lats)
print(lons)
print(time)
print(var.shape)
print(np.mean(np.mean(var, axis=1), axis=1))
'''

'''
# test on the function
iniyear = 1980
endyear = 2005
version = 'countries'
countries = ['Thailand', 'Vietnam', 'Myanmar', 'Cambodia']

dataset, var, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_SAOBS_pre(
    version, iniyear, endyear, countries, missing_ratio=5, ignore_years=[1999])
print(var[0, 0:366])
print(dataset)
# print(var.shape)
'''
