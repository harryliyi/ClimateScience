#this script is used to compare the prect in mam3 and mam7
#by Harry Li


#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.basemap import Basemap, cm

#set up vrcesm and fv2x2 data directories and filenames
mam3dir = "/scratch/d/djones/harryli/cesm1.2/gpc_cesm1_2_2/archive/F_2000_CAM5_mam3/atm/hist/"
mam7dir = "/scratch/d/djones/harryli/cesm1.2/gpc_cesm1_2_2/archive/F_2000_CAM5_mam7/atm/hist/"
mam3monlyfname = "prec_F_2000_CAM5_mam3.cam.h0.0001-0005.nc"
mam7monlyfname = "prec_F_2000_CAM5_mam7.cam.h0.0001-0005.nc"
mam3dailyfname = "prec_F_2000_CAM5_mam3.cam.h1.0001-0005.nc"
mam7dailyfname = "prec_F_2000_CAM5_mam7.cam.h1.0001-0005.nc"

#set up output directory and output log
outdir = "/scratch/d/djones/harryli/cesm1.2/gpc_cesm1_2_2/archive/F_2000_CAM5_mam7/graphs/pre/"

#define inital year and end year
iniyear = 2
endyear = 5
yearts  = ["0002","0003","0004","0005"]

month = np.arange(1,13,1)
mname = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
monts = np.arange(1,(endyear-iniyear+1)*12+1,1)
monmidts = np.arange(6,(endyear-iniyear+1)*12+1,12)

dayts = np.arange(1,(endyear-iniyear+1)*365+1,1)
daymidts = np.arange(180,(endyear-iniyear+1)*365+1,180)

#define the Southeast region
latbounds = [ 10 , 25 ]
lonbounds = [ 100 , 110 ]

################################################################################################
#S1-Compare seasonality in SEA, and pre distribution in SEA
################################################################################################
#open mam3 and mam7 monthly output
fmam3mon  = Dataset(mam3dir+mam3monlyfname)
fmam7mon  = Dataset(mam7dir+mam7monlyfname)

#read lat/lon grids
lats = fmam3mon.variables['lat'][:]
lons = fmam3mon.variables['lon'][:]

# latitude lower and upper index
latli = np.argmin( np.abs( lats - latbounds[0] ) )
latui = np.argmin( np.abs( lats - latbounds[1] ) ) 

# longitude lower and upper index
lonli = np.argmin( np.abs( lons - lonbounds[0] ) )
lonui = np.argmin( np.abs( lons - lonbounds[1] ) ) 
print(lons)
print(lonui)

#read the monthly data
premam3 = fmam3mon.variables['PRECT'][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ] 
premam7 = fmam7mon.variables['PRECT'][ (iniyear-1)*12 : (endyear)*12 , latli:latui+1 , lonli:lonui+1 ]

#print(premam3)

#calculate mean value among SEA and converted to mm/day
premam3ts = np.mean(np.mean(premam3,axis = 2),axis = 1)*86400*1000
premam7ts = np.mean(np.mean(premam7,axis = 2),axis = 1)*86400*1000
#print(premam3ts)
#print(premam3ts[0::12])

#plot as monthly time series
plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.plot(monts,premam3ts, c='r', label = 'mam3')
plt.plot(monts,premam7ts, c='b', label = 'mam7')
plt.legend(loc='upper left')
plt.xticks(monmidts,yearts)
plt.title("Southeaset Asia monthly total precip time series")
plt.ylabel("Total precip (mm/day)")
plt.xlabel("Time")
plt.savefig(outdir+"F2000_aero_prect_mam3vsmam7_SEA_monts.pdf")


#plot climatology
premam3season = np.array([],dtype=float)
premam7season = np.array([],dtype=float)

for k in np.arange(0,12,1):
    premam3season = np.append(premam3season,np.mean(premam3ts[k::12]))
    premam7season = np.append(premam7season,np.mean(premam7ts[k::12]))
    #print(premam3season)

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.plot(month,premam3season, c='r', label = 'mam3')
plt.plot(month,premam7season, c='b', label = 'mam7')
plt.legend(loc='upper left')
plt.xticks(month,mname)
plt.title("Southeast Asia Total precip climatology")
plt.ylabel("Total precip (mm/day)")
plt.xlabel("Month")
plt.savefig(outdir+"F2000_aero_prect_mam3vsmam7_SEA_monclim.pdf")

#read daily data
fmam3day  = Dataset(mam3dir+mam3dailyfname)
fmam7day  = Dataset(mam7dir+mam7dailyfname)

premam3 = fmam3day.variables['PRECT'][ (iniyear-1)*365 : (endyear)*365 , latli:latui+1 , lonli:lonui+1 ]*1000*86400
premam7 = fmam7day.variables['PRECT'][ (iniyear-1)*365 : (endyear)*365 , latli:latui+1 , lonli:lonui+1 ]*1000*86400

#calculate and plot histogram for all SEA grids
binmax  = np.amax(premam3.flatten())*1./2.
binarray = np.arange(0,binmax,binmax/30)

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)
obsfname = "pre.1704241136.clean.dtb"

y,binEdges=np.histogram(premam3.flatten(),bins=binarray)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,c='r', label = 'mam3')

y,binEdges=np.histogram(premam7.flatten(),bins=binarray)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,c='b', label = 'mam7')

plt.yscale('log')
plt.legend(loc='upper right')

plt.title("Total precip distribution at all stations")
plt.ylabel("Days")
plt.xlabel("Precip(mm/day)")

plt.savefig(outdir+"F2000_aero_prect_mam3vsmam7_SEA_hist_allgrids.pdf")

#plot daily timeseries
premam3ts = np.mean(np.mean(premam3,axis = 2),axis = 1)
premam7ts = np.mean(np.mean(premam7,axis = 2),axis = 1)

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.plot(dayts[0:80],premam3ts[0:80], c='r', label = 'mam3')
plt.plot(dayts[0:80],premam7ts[0:80], c='b', label = 'mam7')
plt.legend(loc='upper left')
#plt.xticks(daymidts[0:2],yearts[0:2])
plt.title("Southeaset Asia daily total precip tiem series")
plt.ylabel("Total precip (mm/day)")
plt.xlabel("Time")
plt.savefig(outdir+"F2000_aero_prect_mam3vsmam7_SEA_dayts_first80.pdf")

plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.plot(dayts,premam3ts, c='r', label = 'mam3')
plt.plot(dayts,premam7ts, c='b', label = 'mam7')
plt.legend(loc='upper left')
plt.xticks(daymidts,yearts)
plt.title("Southeaset Asia daily total precip tiem series")
plt.ylabel("Total precip (mm/day)")
plt.xlabel("Time")
plt.savefig(outdir+"F2000_aero_prect_mam3vsmam7_SEA_dayts.pdf")

################################################################################################
#S2-Compare global prect distribution
################################################################################################
#read the monthly data
premam3 = fmam3mon.variables['PRECT'][ (iniyear-1)*12 : (endyear)*12 , : , : ]*86400*1000
premam7 = fmam7mon.variables['PRECT'][ (iniyear-1)*12 : (endyear)*12 , : , : ]*86400*1000

for k in np.arange(0,12,1):
    premam3contour = np.mean(premam3[k::12,:,:],axis=0)
    premam7contour = np.mean(premam7[k::12,:,:],axis=0)
    premam7_3_diff = premam7contour - premam3contour

    plt.clf()
    map = Basemap(projection='gall',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=0,urcrnrlon=360,resolution = 'l')

    map.drawcoastlines()
    map.drawcountries()

    ny = premam3contour.shape[0];
    nx = premam3contour.shape[1]
    mlons, mlats = map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    x, y = map(mlons, mlats) # compute map proj coordinates.
    # draw filled contours.
    clevs = [0,1,2,3,4,5,6,7,8,10,12.5,15,17.5,20,30,40,60]
    cs = map.contourf(x,y,premam3contour,clevs,cmap=cm.s3pcpn)
    # add colorbar.
    cbar = map.colorbar(cs,location='bottom',pad="5%")
    cbar.set_label('mm')
    # add title
    plt.title("MAM3 mean total precipitation global distribution in "+mname[k])
    plt.savefig(outdir+"F2000_aero_prect_mam3_global_contour_"+str(k+1)+".pdf")

    plt.clf()
    map = Basemap(projection='gall',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=0,urcrnrlon=360,resolution = 'l')
    map.drawcoastlines()
    map.drawcountries()

    ny = premam3contour.shape[0];
    nx = premam3contour.shape[1]
    mlons, mlats = map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    x, y = map(mlons, mlats) # compute map proj coordinates.
    # draw filled contours.
    clevs = [0,1,2,3,4,5,6,7,8,10,12.5,15,17.5,20,30,40,60]
    cs = map.contourf(x,y,premam7contour,clevs,cmap=cm.s3pcpn)
    # add colorbar.
    cbar = map.colorbar(cs,location='bottom',pad="5%")
    cbar.set_label('mm')
    # add title
    plt.title("MAM7 mean total precipitation global distribution in "+mname[k])
    plt.savefig(outdir+"F2000_aero_prect_mam7_global_contour_"+str(k+1)+".pdf")

    plt.clf()
    map = Basemap(projection='gall',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=0,urcrnrlon=360,resolution = 'l')
    map.drawcoastlines()
    map.drawcountries()
    
    ny = premam3contour.shape[0];
    nx = premam3contour.shape[1]
    mlons, mlats = map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    x, y = map(mlons, mlats) # compute map proj coordinates.
    # draw filled contours.
    clevs = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
    cs = map.contourf(x,y,premam7_3_diff,clevs,cmap=cm.GMT_drywet)
    # add colorbar.
    cbar = map.colorbar(cs,location='bottom',pad="5%")
    cbar.set_label('mm')
    # add title
    plt.title("MAM7 vs MAM3 mean total precipitation global distribution in "+mname[k])
    plt.savefig(outdir+"F2000_aero_prect_mam7vsmam3_global_contour_"+str(k+1)+".pdf")

    print("Currentlt plotting for "+mname[k])

premam3contour = np.mean(premam3,axis=0)
premam7contour = np.mean(premam7,axis=0)
premam7_3_diff = premam7contour - premam3contour

#plot mam3 mean global precip
plt.clf()
map = Basemap(projection='gall',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=0,urcrnrlon=360,resolution = 'l')

map.drawcoastlines()
map.drawcountries()

ny = premam3contour.shape[0]; 
nx = premam3contour.shape[1]
mlons, mlats = map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
x, y = map(mlons, mlats) # compute map proj coordinates.
# draw filled contours.
clevs = [0,1,2,3,4,5,6,7,8,10,12.5,15,17.5,20,30,40,60]
cs = map.contourf(x,y,premam3contour,clevs,cmap=cm.s3pcpn)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title("MAM3 mean total precipitation global distribution")
plt.savefig(outdir+"F2000_aero_prect_mam3_global_contour_avg.pdf")


#plot mam7 mean global precip
plt.clf()
map = Basemap(projection='gall',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=0,urcrnrlon=360,resolution = 'l')

map.drawcoastlines()
map.drawcountries()

ny = premam3contour.shape[0];
nx = premam3contour.shape[1]
mlons, mlats = map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
x, y = map(mlons, mlats) # compute map proj coordinates.
# draw filled contours.
clevs = [0,1,2,3,4,5,6,7,8,10,12.5,15,17.5,20,30,40,60]
cs = map.contourf(x,y,premam7contour,clevs,cmap=cm.s3pcpn)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title("MAM7 mean total precipitation global distribution")
plt.savefig(outdir+"F2000_aero_prect_mam7_global_contour_avg.pdf")

#plot mam7 vs mam3 global difference
plt.clf()
map = Basemap(projection='gall',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=0,urcrnrlon=360,resolution = 'l')

map.drawcoastlines()
map.drawcountries()

ny = premam3contour.shape[0];
nx = premam3contour.shape[1]
mlons, mlats = map.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
x, y = map(mlons, mlats) # compute map proj coordinates.
# draw filled contours.
clevs = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
cs = map.contourf(x,y,premam7_3_diff,clevs,cmap=cm.GMT_drywet)
# add colorbar.
cbar = map.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
plt.title("MAM7 vs MAM3 mean total precipitation global distribution")
plt.savefig(outdir+"F2000_aero_prect_mam7vsmam3_global_contour_avg.pdf")


