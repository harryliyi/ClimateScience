#import libraries
import numpy as np
import netCDF4
import datetime as datetime
import pandas as pd

#set up era interim data directories and filename
eradir = "/scratch/d/dylan/harryli/obsdataset/ERA_interim/pre/daily/"
erafname = "era_interim_tp_daily_19790101-20051231.nc"

#define inital year and end year
iniyear = 1980
endyear = 2005
inidate = datetime.datetime.strptime(str(iniyear)+'-01-01 12:00','%Y-%m-%d %H:%M')
enddate = datetime.datetime.strptime(str(endyear)+'-12-31 12:00','%Y-%m-%d %H:%M')
print(inidate)
print(enddate)

#set up a test lat/lon
stnlat = 16.88
stnlon = 99.15

#open era_interim file
eradata = netCDF4.Dataset(eradir+erafname)

eralats= eradata.variables['latitude'][:]
eralons= eradata.variables['longitude'][:]

eralat_idx= np.abs(eralats - stnlat).argmin()
eralon_idx= np.abs(eralons - stnlon).argmin()

time_var = eradata.variables['time']
dtime = netCDF4.num2date(time_var[:],time_var.units)
dtime = pd.to_datetime(dtime)
select_dtime = (dtime>=inidate)&(dtime<=enddate)&(~((dtime.month==2)&(dtime.day==29)))

erapre = eradata.variables['tp'][select_dtime,eralat_idx, eralon_idx]


print(dtime)
print(len(select_dtime))
dtime = dtime[select_dtime]
print(dtime)
print(len(dtime))
print(len(erapre))
