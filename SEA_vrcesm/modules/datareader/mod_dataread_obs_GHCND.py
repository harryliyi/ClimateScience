'''
This is a function used to read GHCND data

Description on parameters:
1) project: the name of dataset

2) iniyear/endyear: the time bounds

3) countries: the list of countries

4) returns: var is the returned data, dtime is the series of datetime, lats/lons are latitudes/longitudes
'''

# modules that the function needs
import numpy as np
import pandas as pd
import datetime as datetime

################################################################################################
# create a function to convert degree/minute/second to decimal
################################################################################################


def deg2dec(x):
    xlist = x.split(":")
    if xlist[0][0] == "-":
        xdec = float(xlist[0])-float(xlist[1])/60.-float(xlist[2])/60./60.
    else:
        xdec = float(xlist[0])+float(xlist[1])/60.+float(xlist[2])/60./60.
    return xdec

################################################################################################
# read GHCND daily data
################################################################################################


def read_GHCND_pre(iniyear, endyear, countries, missing_ratio, **kwargs):

    # set up SA-OBS pre observations directory filename
    stnsum = "stations.txt"

    countries_dict = {'Thailand': 'TH', 'Vietnam': 'VM', 'Cambodia': 'CB', 'Myanmar': 'BM', 'LAOS': 'LA',
                      'Malaysia': 'MY', 'Philippines': 'RP', 'Indonesia': 'ID', 'Singapore': 'SN'}

    countryids = [countries_dict[icountry] for icountry in countries]

    indir = '/scratch/d/dylan/harryli/obsdataset/GHCN/daily/GHCND/'

    yearts = range(iniyear, endyear+1, 1)
    inidate = datetime.datetime.strptime(str(iniyear)+'-01-01', '%Y-%m-%d')
    enddate = datetime.datetime.strptime(str(endyear+1)+'-01-01', '%Y-%m-%d')
    dtime = pd.date_range(start=str(iniyear)+'-01-01', end=str(endyear)+'-12-31', freq='D')
    # print(inidate)

    # setup the ignore datetime
    if 'ignore_years' not in kwargs:
        ignore_years = []
    else:
        ignore_years = kwargs['ignore_years']

    dtime = dtime[(~((dtime.month == 2) & (dtime.day == 29))) & (~ np.in1d(dtime.year, ignore_years))]
    ndays = len(dtime)

    # find the stn ids from stations.txt
    stations = pd.read_csv(indir+'stations.csv', sep=',', header=0)
    # print(stations)
    # print(countryids)
    # print(stations['CN'].values)
    # print(np.in1d(stations['CN'].values, countryids))

    stnids = stations.loc[np.in1d(stations['CN'], countryids), 'STAID'].values
    stnnames = stations.loc[np.in1d(stations['CN'], countryids), 'STANAME'].values
    countrynames = stations.loc[np.in1d(stations['CN'], countryids), 'CN'].values
    stnlats = stations.loc[np.in1d(stations['CN'], countryids), 'LAT'].values
    stnlons = stations.loc[np.in1d(stations['CN'], countryids), 'LON'].values
    stnhgts = stations.loc[np.in1d(stations['CN'], countryids), 'HGHT'].values
    stnmiss = []

    dataset = {'Date': dtime}
    dataset = pd.DataFrame.from_dict(dataset)
    dataset.set_index('Date', inplace=True)
    # print(dataset)

    print("Totally "+str(len(stnids))+" stations are found. Their information is shown in following:")
    for idx in range(len(stnids)):
        countrynames[idx] = list(countries_dict.keys())[list(countries_dict.values()).index(countrynames[idx])]
        stnlats[idx] = round(stnlats[idx], 5)
        stnlons[idx] = round(stnlons[idx], 5)
        stnhgts[idx] = round(stnhgts[idx], 5)

        print("station "+str(idx+1)+"/"+str(len(stnids))+": "+stnnames[idx]+" in "+countrynames[idx]+" at "+str(
            stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6))

        currstn = pd.read_csv(indir+stnids[idx]+'.csv', sep=',', usecols=['DATE', 'PRCP'],
                              dtype={'PRCP': np.float64}, header=0)
        currstn['DATE'] = pd.to_datetime(currstn['DATE'], format='%Y-%m-%d')
        currstn['PRCP'] = currstn['PRCP']*25.4  # convert to mm from inches
        currstn = currstn[(currstn['DATE'] >= inidate) & (currstn['DATE'] < enddate) & (
            ~((currstn['DATE'].dt.month == 2) & (currstn['DATE'].dt.day == 29))) & (~ np.in1d(currstn['DATE'].dt.year, ignore_years))]

        print("Current missing ratio is "+str(100.-100. * currstn.count()['PRCP']/ndays)+'%')
        stnmiss.append(round(100.-100. * currstn.count()['PRCP']/ndays, 2))
        currstn = currstn.rename(columns={'PRCP': stnnames[idx], 'DATE': 'Date'})
        # print(currstn)
        currstn.set_index('Date', inplace=True)
        dataset = dataset.join(currstn, how='left')

    # print(dataset[((dataset.index.year == 2004) & (dataset.index.month == 2))])
    # missdata = {'Stations': stnnames,
    #             'Missing Ratio': stnmiss}
    # missdata = pd.DataFrame(missdata)
    # print(missdata)

    return dataset, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss


################################################################################################
# test on the function
################################################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/notes/'
iniyear = 1980
endyear = 2005
version = 'countries'
countries = ['Thailand', 'Vietnam', 'Myanmar', 'Cambodia', 'LAOS']

dataset, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_GHCND_pre(iniyear, endyear, countries, missing_ratio=10, ignore_years=[])

missdata = {'Stations': stnnames,
            '1980-2005 Missing Ratio (%)': stnmiss}

iniyear = 1990
dataset, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_GHCND_pre(iniyear, endyear, countries, missing_ratio=10, ignore_years=[])
missdata['1990-2005 Missing Ratio (%)'] = stnmiss

iniyear = 2000
dataset, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_GHCND_pre(iniyear, endyear, countries, missing_ratio=10, ignore_years=[])
missdata['2000-2005 Missing Ratio (%)'] = stnmiss

missdata = pd.DataFrame(missdata)
print(missdata)
missdata.to_csv(outdir+'GHCND_missingratio.csv', sep=',', index=False)

# print(var[0, 0:366])
# print(dataset)
# print(var.shape)
