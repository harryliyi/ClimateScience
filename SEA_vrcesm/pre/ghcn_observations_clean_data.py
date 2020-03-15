#import libraries
import numpy as np
import pandas as pd
import datetime as datetime

#set up GHCN pre observations directory filename
obsdir = '/scratch/d/dylan/harryli/obsdataset/GHCN/daily/'
obsfname = 'GHCN_daily_1979_2010'

#define country name
countries = ['Vietnam','Cambodia','Laos','Malaysia','Myanmar','Philipines','Singapore','Thailand','Indonesia']

#define inital year and end year
iniyear = 1980
endyear = 2005
inidate = datetime.datetime.strptime(str(iniyear)+'-01-01','%Y-%m-%d')
enddate = datetime.datetime.strptime(str(endyear)+'-12-31','%Y-%m-%d')

#define missing threshold
missthres = 0.15

#define the stn ids
stnids = np.array([],dtype=str)
stnnames = np.array([],dtype=str)
countrynames = np.array([],dtype=str)
stnlats = np.array([],dtype=float)
stnlons = np.array([],dtype=float)
ghcndpre = np.array([],dtype=float)

total_stations = 0
total_days     = (endyear-iniyear+1) * 365


print('--------------------------------------------------------------------')
print('Start the script...')
#read the data from GHCN
#find all stations
for idx,icountry in enumerate(countries):
    
    fname = obsfname+'_'+icountry+'.csv'
    print('Current file name is: '+fname)
    countrynames = np.append(countrynames,icountry)
    
    df = pd.read_csv(obsdir+fname,header=0,delimiter=',')
    df['STATION'] = df['STATION'].astype(str)
    df['NAME'] = df['NAME'].astype(str)
    df['LATITUDE'] = df['LATITUDE'].astype(float)
    df['LONGITUDE'] = df['LONGITUDE'].astype(float)
    df['LONGITUDE'] = df['LONGITUDE'].astype(float)
    df['PRCP'] = df['PRCP'].astype(float)

    countrystations = df['STATION'].unique()
    
    stnids = np.concatenate((stnids,df['STATION'].unique()))
print('--------------------------------------------------------------------')
count_stations = len(stnids)
icount = 0
stnids = np.array([],dtype=str)

#read lats/lons and precip
for idx,icountry in enumerate(countries):

    fname = obsfname+'_'+icountry+'.csv'
    df = pd.read_csv(obsdir+fname,header=0,delimiter=',')
    df['STATION'] = df['STATION'].astype(str)
    df['NAME'] = df['NAME'].astype(str)
    df['LATITUDE'] = df['LATITUDE'].astype(float)
    df['LONGITUDE'] = df['LONGITUDE'].astype(float)
    df['LONGITUDE'] = df['LONGITUDE'].astype(float)
    df['PRCP'] = df['PRCP'].astype(float)
    df['DATE'] = df['DATE'].astype('datetime64[ns]')

    countrystations = df['STATION'].unique()
    countrystnnames = df['NAME'].unique()
    #append station lat/lon
    for ii,istation in enumerate(countrystations):
        icount = icount + 1
        #print(df.index[df['STATION'] == istation][0])
        iname = df.iloc[df.index[df['STATION'] == istation][0],1]
        ilat  = df.iloc[df.index[df['STATION'] == istation][0],2]
        ilon  = df.iloc[df.index[df['STATION'] == istation][0],3]
        print('Working on Station('+str(icount)+'/'+str(count_stations)+'): '+iname+'('+istation+') located in '+str(ilat)+'/'+str(ilon)+' in '+icountry)

        #stnlats = np.append(stnlats,ilat)
        #stnlons = np.append(stnlons,ilon)

        tempdf = df.loc[(df['STATION']==istation)&(df['DATE']>=inidate)&(df['DATE']<=enddate),['DATE','PRCP']].copy()
        tempdf.loc[tempdf['PRCP'].isnull(),['PRCP']] = np.nan

        tempdf   = tempdf.set_index('DATE')
        idxdates = pd.DatetimeIndex(start=inidate, end=enddate, freq='D')
        tempdf   = tempdf.reindex(idxdates)
        #print(tempdf)        
        #print(len(tempdf))
        tempdf = tempdf[~((tempdf.index.month == 2) & (tempdf.index.day == 29))]
        #print(len(tempdf))
        
        dayscount = tempdf['PRCP'].count()
        print('Non-null data in total is '+str(dayscount)+'/'+str(total_days)+'), missing data ratio is '+str(1.*(total_days-dayscount)/total_days))
        if (1.*(total_days-dayscount)/total_days<missthres):
            print('Missing ratio is lower than '+str(missthres)+',data is included')
            ghcndpre = np.concatenate((ghcndpre,tempdf['PRCP']))
            stnids   = np.append(stnids,istation)
            stnnames = np.append(stnnames,iname)
            stnlats  = np.append(stnlats,ilat)
            stnlons  = np.append(stnlons,ilon)
        else:
            print('Missing ratio is too high, data is excluded')

        #print(tempdf.loc[tempdf['PRCP'].isnull(),['DATE','PRCP']])
        #print(tempdf[['DATE','PRCP']])



    print('--------------------------------------------------------------------')

print(ghcndpre[0:100])
ghcndpre = np.reshape(ghcndpre,(len(stnids),365*(endyear-iniyear+1)))
print(np.shape(ghcndpre))
print(ghcndpre[0,0:100])

#    stnlats  = np.concatenate([stnlats,df['LATITUDE'].unique()])
#    stnlons  = np.concatenate([stnlons,df['LONGITUDE'].unique()])

print(len(stnids))
print(len(stnnames))
print(len(stnlats))
print(len(stnlons))

print('--------------------------------------------------------------------')


