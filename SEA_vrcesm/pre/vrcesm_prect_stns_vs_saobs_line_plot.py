#This script is used to read daily station precip from SA-OBS and compare them with vrseasia and fv0.9x1.25 output
#Several steps are inplemented:
#S1-read precip data in mainland Southeast Asia from SA-OBS
#S2-read data from vrseasia and fv0.9x1.25 output and compute statics, draw station line plots
#S3-draw scatter plot of statics on map
#
#by Harry Li

#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import pandas as pd

#set up SA-OBS pre observations directory filename
obsdir = "/scratch/d/dylan/harryli/obsdataset/SA_OBS/countries/"
stnsum = "stations.txt"

#set up vrcesm and fv1x1 data directories and filenames
vrcesmdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/vrseasia_AMIP_1979_to_2005/atm/hist/"
fv1x1dir = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f09_f09_AMIP_1979_to_2005/atm/hist/"
ne30dir  = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/ne30_ne30_AMIP_1979_to_2005/atm/hist/"
vrcesmfname = "fv02_prec_vrseasia_AMIP_1979_to_2005.cam.h1.1979-2005.nc"
fv1x1fname = "PREC_f09_f09_AMIP_1979_to_2005.cam.h1.1979-2005.nc"
ne30fname  = "fv09_PREC_ne30_ne30_AMIP_1979_to_2005.cam.h1.1979-2005.nc"

#define inital year and end year
iniyear = 1980
endyear = 2005

#set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_obs/stations/SAOBS_countries/"+str(iniyear)+"-"+str(endyear)+"/"
outlog = open(outdir+"vrcesm_prect_stns_vs_cru_line_plot_output.log", "w")

#define country name
countries = ["Thailand","Vietnam","Cambodia","Myanmar"]
countryids = ["TH","VN","KH","MM"]

#create year series
yearts = np.array(str(iniyear), dtype=np.datetime64)
yearts = yearts + np.arange(endyear-iniyear+1)
yearmidtslabel = yearts[0::2]
#print(yearts)
#print(yearmidts)

#create time series
monts = np.array(str(iniyear)+'-01', dtype=np.datetime64)
monts = monts + np.arange((endyear-iniyear+1)*12)
yearmidts = monts[5::24]
print(yearmidts)

#create time series
date = pd.date_range(str(iniyear)+'-01-01', str(endyear)+'-12-31')
is_leap_day = (date.month == 2) & (date.day == 29)
date = date[~is_leap_day]
print(len(date))

#define selected stations
stnnamesselect = ["SONGKHLA","PHRAE","PHUKET","SURAT THANI","TRANG","KHLONG YAI","BANGKOK","UTTARADIT","TAN SON HOA"]
################################################################################################
#S0-function definition
################################################################################################

#create a function to convert degree/minute/second to decimal
def deg2dec(x):
    xlist =x.split(":")
    if xlist[0][0]=="-":
        xdec = float(xlist[0])-float(xlist[1])/60.-float(xlist[2])/60./60.
    else:
        xdec = float(xlist[0])+float(xlist[1])/60.+float(xlist[2])/60./60.
    return xdec

#define a function to calculate root mean square error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

################################################################################################
#S1-read precip data in mainland Southeast Asia from SA-OBS
################################################################################################

print("Start to read data from SA-OBS...")
outlog.write("Start to read data from SA-OBS...\n")
#find the stn ids from stations.txt
fopen = open(obsdir+stnsum,"r")
linereader = fopen.readlines()[19:]

stnids = np.array([],dtype=int)
stnnames = np.array([])
countrynames = np.array([])
stnlats = np.array([],dtype=float)
stnlons = np.array([],dtype=float)

for lineno in range(len(linereader)):
    linelist = linereader[lineno].split(",")
    #print(linelist)
    #print(" ".join(x for x in linelist[1].split()))
    if linelist[2] in countryids:
        stnids = np.append(stnids,int(linelist[0]))
        stnnames = np.append(stnnames," ".join(x for x in linelist[1].split()))
        countrynames = np.append(countrynames,countries[countryids.index(linelist[2])])
        stnlats = np.append(stnlats,deg2dec(linelist[3]))
        stnlons = np.append(stnlons,deg2dec(linelist[4]))
    print("Current station "+linelist[0]+" is "+" ".join(x for x in linelist[1].split())+" in "+linelist[2]+" at "+str(deg2dec(linelist[3]))+"/"+str(deg2dec(linelist[4])))
    outlog.write("Current station "+linelist[0]+" is "+" ".join(x for x in linelist[1].split())+" in "+linelist[2]+" at "+str(deg2dec(linelist[3]))+"/"+str(deg2dec(linelist[4]))+"\n")


saobspre = np.array([],dtype=float)

#read precip data from each station
print("Totally "+str(len(stnids))+" stations are found. Their information is shown in following:")
for idx in range(len(stnids)):
    print("station "+str(idx+1)+"/"+str(len(stnids))+" is: "+stnnames[idx]+" in "+countrynames[idx]+" at "+str(stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6))
    outlog.write("station "+str(idx+1)+"/"+str(len(stnids))+" is: "+stnnames[idx]+" in "+countrynames[idx]+" at "+str(stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6)+"\n")
    
    #open file from each station
    obsfname = "RR_STAID"+str(stnids[idx]).zfill(6)+".txt"
    fopen = open(obsdir+obsfname,"r")
    linereader = fopen.readlines()[21:]
    
    #read data from station between given years
    lineno = 0
    currpre = np.array([],dtype=float)
    missingcount = 0.
    print(linereader[lineno].split(","))
    #while int(linereader[lineno].split(",")[2][0:4])<=endyear:
    while int(linereader[lineno].split(",")[1][0:4])<=endyear:
        #print(linereader[lineno].split(",")[2])
        if (int(linereader[lineno].split(",")[1][0:4])>=iniyear) and (linereader[lineno].split(",")[1][4:8]!="0229"):
            currpre = np.append(currpre,float(linereader[lineno].split(",")[2]))
            if (int(linereader[lineno].split(",")[3])==9) or (int(linereader[lineno].split(",")[3])==1):
                missingcount = missingcount + 1.
                currpre[-1] = np.NaN
        lineno = lineno +1
    
    #record the data if it cover the whole temperal range and no missing data 
    if (len(currpre)==365*(endyear-iniyear+1)):
        print("Current missing ratio is "+str(missingcount/365/(endyear-iniyear+1)))
        outlog.write("Current missing ratio is "+str(missingcount/365/(endyear-iniyear+1))+"\n")
        saobspre = np.concatenate((saobspre,currpre))
    #print(currpre[0:365])

#convert to mm and regrid saobs data
#print(np.nansum(saobspre))
saobspre = saobspre/10.
#print(np.nansum(saobspre))
saobspre = np.reshape(saobspre,(len(stnids),365*(endyear-iniyear+1)))
print(np.shape(saobspre))

################################################################################################
#S2-read data from vrseasia and fv0.9x1.25 output and compute statics
################################################################################################

#open vrcesm file, fv1x1 and ne30 file
vrcesmdata = Dataset(vrcesmdir+vrcesmfname)
fv1x1data  = Dataset(fv1x1dir+fv1x1fname)
ne30data   = Dataset(ne30dir+ne30fname)

#read lat/lon grids
vrlats = vrcesmdata.variables['lat'][:]
vrlons = vrcesmdata.variables['lon'][:]

fvlats = fv1x1data.variables['lat'][:]
fvlons = fv1x1data.variables['lon'][:]

nelats = ne30data.variables['lat'][:]
nelons = ne30data.variables['lon'][:]

print("Comparing the SA-OBS station observations with vrcesm and fv1x1...")
outlog.write("Comparing the SA-OBS station observations with vrcesm and fv1x1...\n")

#define station daily statistics
vrstnstd = np.array([],dtype=float)
vrstnrse = np.array([],dtype=float)
vrstncor = np.array([],dtype=float)
fvstnstd = np.array([],dtype=float)
fvstnrse = np.array([],dtype=float)
fvstncor = np.array([],dtype=float)
nestnstd = np.array([],dtype=float)
nestnrse = np.array([],dtype=float)
nestncor = np.array([],dtype=float)

#define station monthly statistics
vrstnstdmon = np.array([],dtype=float)
vrstnrsemon = np.array([],dtype=float)
vrstncormon = np.array([],dtype=float)
fvstnstdmon = np.array([],dtype=float)
fvstnrsemon = np.array([],dtype=float)
fvstncormon = np.array([],dtype=float)
nestnstdmon = np.array([],dtype=float)
nestnrsemon = np.array([],dtype=float)
nestncormon = np.array([],dtype=float)

#define all stations precip
sapreall = np.array([],dtype=float)
vrpreall = np.array([],dtype=float)
fvpreall = np.array([],dtype=float)
nepreall = np.array([],dtype=float)

#read data from cesm and compare with SA-OBS for each station
for idx in range(len(stnids)):
    print("Currently dealing with the station "+str(idx+1)+"/"+str(len(stnids))+": "+stnnames[idx]+" in "+countrynames[idx]+" at "+str(stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6))
    outlog.write("Currently dealing with the station "+str(idx+1)+"/"+str(len(stnids))+" : "+stnnames[idx]+" in "+countrynames[idx]+" at "+str(stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6)+"\n")

    #get the station location
    stnlat = stnlats[idx]
    stnlon = stnlons[idx]

    #find the station in vrcesm, fv1x1 and ne30 grids
    vrlat_idx = np.abs(vrlats - stnlat).argmin()
    vrlon_idx = np.abs(vrlons - stnlon).argmin()
    fvlat_idx = np.abs(fvlats - stnlat).argmin()
    fvlon_idx = np.abs(fvlons - stnlon).argmin()
    nelat_idx = np.abs(nelats - stnlat).argmin()
    nelon_idx = np.abs(nelons - stnlon).argmin()

    #get the station precip in vrcesm, fv1x1 and ne30 data and convert to mm/day
    vrpre = vrcesmdata.variables['PRECT'][(iniyear-1979)*365:(endyear-1979+1)*365,vrlat_idx, vrlon_idx]
    fvpre = fv1x1data.variables['PRECT'][(iniyear-1979)*365:(endyear-1979+1)*365,fvlat_idx, fvlon_idx]
    nepre = ne30data.variables['PRECT'][(iniyear-1979)*365:(endyear-1979+1)*365,nelat_idx, nelon_idx]
    vrpre = vrpre*86400*1000
    fvpre = fvpre*86400*1000
    nepre = nepre*86400*1000

    #mask vrcesm, fv0.9x1.25 and ne30 with SA-OBS missing value
    #print(np.nansum(vrpre))
    vrpre[np.isnan(saobspre[idx,:])] = np.NaN
    fvpre[np.isnan(saobspre[idx,:])] = np.NaN
    nepre[np.isnan(saobspre[idx,:])] = np.NaN
    #print(vrpre[(1999-iniyear)*365:(1999-iniyear)*365+365])
    #print(saobspre[idx,(1999-iniyear)*365:(1999-iniyear)*365+365])

    print("SA-OBS precip sum is: "+str(np.nansum(saobspre[idx,:])))
    print("vrseasia precip sum is: "+str(np.nansum(vrpre)))
    print("fv0.9x1.25 precip sum is: "+str(np.nansum(fvpre)))
    print("ne30 precip sum is: "+str(np.nansum(nepre)))

    #select non missing value
    vrprenon = vrpre[~np.isnan(saobspre[idx,:])]
    fvprenon = fvpre[~np.isnan(saobspre[idx,:])]
    neprenon = nepre[~np.isnan(saobspre[idx,:])]
    saprenon = saobspre[idx,~np.isnan(saobspre[idx,:])]

    #calculate the daily statistics for taylor diagram and scatter plot
    vrstnstd  = np.append(vrstnstd,np.std(vrprenon))
    fvstnstd  = np.append(fvstnstd,np.std(fvprenon))
    nestnstd  = np.append(nestnstd,np.std(neprenon))
    vrstncor  = np.append(vrstncor,pearsonr(vrprenon,saprenon)[0])
    fvstncor  = np.append(fvstncor,pearsonr(fvprenon,saprenon)[0])
    nestncor  = np.append(nestncor,pearsonr(neprenon,saprenon)[0])
    vrstnrse  = np.append(vrstnrse,rmse(vrprenon,saprenon))
    fvstnrse  = np.append(fvstnrse,rmse(fvprenon,saprenon))
    nestnrse  = np.append(nestnrse,rmse(neprenon,saprenon))

    #set time range for each dataset
    vr_data = pd.DataFrame({
        'date': date,
        'pre':vrpre 
    })
    #print(np.nansum(vr_data['pre']))
    vr_data.loc[np.isnan(saobspre[idx,:]),'pre']=np.NaN
    #print(vr_data['pre'].sum())

    fv_data = pd.DataFrame({
        'date': date,
        'pre':fvpre
    })
    fv_data.loc[np.isnan(saobspre[idx,:]),'pre'] = np.NaN

    ne_data = pd.DataFrame({
        'date': date,
        'pre':nepre
    })
    ne_data.loc[np.isnan(saobspre[idx,:]),'pre'] = np.NaN

    sa_data = pd.DataFrame({
        'date': date,
        'pre':saobspre[idx,:]
    })
    sa_data.loc[np.isnan(saobspre[idx,:]),'pre'] = np.NaN

    ##################################################################################
    #plot annual maximum precip for each station
    vr_data = vr_data.set_index(['date'])
    vr_data_annmax = vr_data.resample('A').max()
    #print(vr_data_annmax)

    fv_data = fv_data.set_index(['date'])
    fv_data_annmax = fv_data.resample('A').max()

    ne_data = ne_data.set_index(['date'])
    ne_data_annmax = ne_data.resample('A').max()

    sa_data = sa_data.set_index(['date'])
    sa_data_annmax = sa_data.resample('A').max()

    plt.clf()

    plt.plot(yearts,sa_data_annmax['pre'], c='k', label = 'SA-OBS')
    plt.plot(yearts,vr_data_annmax['pre'], c='r', label = 'vrcesm')
    plt.plot(yearts,fv_data_annmax['pre'], c='b', label = 'fv1x1')
    plt.plot(yearts,ne_data_annmax['pre'], c='g', label = 'ne30')
    plt.legend(loc='upper left')

    plt.title("Annual Maximum precip in the station "+str(stnids[idx]).zfill(6)+": "+stnnames[idx]+" in "+countrynames[idx])
    plt.ylabel("Maximum precip (mm/day)")
    plt.xlabel("Year")

    plt.savefig(outdir+"vrcesm_prect_vs_saobs_station_obs_maxts_"+str(idx+1)+"_"+countrynames[idx]+".pdf")

    ##################################################################################
    #plot monthly precip for each station
    vr_data_mon = vr_data.resample('M').mean()
    #print(vr_data_annmax)

    fv_data_mon = fv_data.resample('M').mean()

    ne_data_mon = ne_data.resample('M').mean()

    sa_data_mon = sa_data.resample('M').mean()

    plt.clf()

    plt.plot(monts,sa_data_mon['pre'], c='k', label = 'SA-OBS')
    plt.plot(monts,vr_data_mon['pre'], c='r', label = 'vrcesm')
    plt.plot(monts,fv_data_mon['pre'], c='b', label = 'fv1x1')
    plt.plot(monts,ne_data_mon['pre'], c='g', label = 'ne30')
    plt.legend(loc='upper left')

    plt.title("Monthly total precip in the station "+str(stnids[idx]).zfill(6)+": "+stnnames[idx]+" in "+countrynames[idx])
    plt.ylabel("Total precip (mm/day)")
    plt.xlabel("Time")

    plt.savefig(outdir+"vrcesm_prect_vs_saobs_station_obs_monts_"+str(idx+1)+"_"+countrynames[idx]+".pdf")


    ##################################################################################
    #vr_data.loc[np.isnan(saobspre[idx,:]),'pre']=0.
    #fv_data.loc[np.isnan(saobspre[idx,:]),'pre']=0.
    #sa_data.loc[np.isnan(saobspre[idx,:]),'pre']=0.
    #plot histtogram for each station
    plt.clf()

    #define bin list
    #binmax = np.amax([np.amax(sa_data['pre']),np.amax(vr_data['pre']),np.amax(fv_data['pre'])])*1./2.
    binmax  = np.amax(sa_data['pre'])*1./2.
    binarray = np.arange(0,binmax,binmax/30)
    print(binarray)
    #print(sa_data.loc[~np.isnan(saobspre[idx,:]),'pre'])

    y,binEdges=np.histogram(sa_data.loc[~np.isnan(saobspre[idx,:]),'pre'],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    #print(bincenters)
    plt.plot(bincenters,y,c='k', label = 'SA-OBS')

    y,binEdges=np.histogram(vr_data.loc[~np.isnan(saobspre[idx,:]),'pre'],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,c='r', label = 'vrcesm')

    y,binEdges=np.histogram(fv_data.loc[~np.isnan(saobspre[idx,:]),'pre'],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,c='b', label = 'fv0.9x1.25')

    y,binEdges=np.histogram(ne_data.loc[~np.isnan(saobspre[idx,:]),'pre'],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,c='g', label = 'ne30')

    plt.yscale('log')
    plt.legend(loc='upper right')
    
    plt.title("Total precip distribution in the station "+str(stnids[idx]).zfill(6)+": "+stnnames[idx]+" in "+countrynames[idx])
    plt.ylabel("Days")
    plt.xlabel("Precip(mm/day)")

    plt.savefig(outdir+"vrcesm_prect_vs_saobs_station_obs_hist_"+str(idx+1)+"_"+countrynames[idx]+".pdf")

    #calculate the monthly statistics for taylor diagram and scatter plot
    vrstnstdmon  = np.append(vrstnstdmon,np.std(vr_data_mon['pre']))
    fvstnstdmon  = np.append(fvstnstdmon,np.std(fv_data_mon['pre']))
    nestnstdmon  = np.append(nestnstdmon,np.std(ne_data_mon['pre']))

    vrstncormon  = np.append(vrstncormon,pearsonr(vr_data_mon['pre'],sa_data_mon['pre'])[0])
    fvstncormon  = np.append(fvstncormon,pearsonr(fv_data_mon['pre'],sa_data_mon['pre'])[0])
    nestncormon  = np.append(nestncormon,pearsonr(ne_data_mon['pre'],sa_data_mon['pre'])[0])

    vrstnrsemon  = np.append(vrstnrsemon,rmse(vr_data_mon['pre'],sa_data_mon['pre']))
    fvstnrsemon  = np.append(fvstnrsemon,rmse(fv_data_mon['pre'],sa_data_mon['pre']))
    nestnrsemon  = np.append(nestnrsemon,rmse(ne_data_mon['pre'],sa_data_mon['pre']))

    sapreall = np.concatenate((sapreall,sa_data['pre'][:]))
    vrpreall = np.concatenate((vrpreall,vr_data['pre'][:]))
    fvpreall = np.concatenate((fvpreall,fv_data['pre'][:]))
    nepreall = np.concatenate((nepreall,ne_data['pre'][:]))

    print(np.shape(vrpreall))
    print(np.count_nonzero(~np.isnan(vrpreall)))
    print(np.shape(sapreall))
    print(np.count_nonzero(~np.isnan(sapreall)))


################################################################################################
#S3-darw precip data for mainland Southeast Asia from SA-OBS
################################################################################################   
#deal with daily precip
print("Start to draw scatter plot for daily statics...")
#scatter plot correlation and rmse on map for vrcesm
plt.clf()
map = Basemap(projection='merc', lat_0=17, lon_0=100.5,resolution = 'h', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

map.drawcoastlines()
map.drawcountries()
#map.fillcontinents(color = 'white')
#map.drawmapboundary()

latsmap = stnlats[0:]
lonsmap = stnlons[0:]

x,y = map(lonsmap, latsmap)
map.scatter(x, y, s=10*vrstnrse[0:], marker="o",c=vrstncor[0:],cmap=cm.coolwarm,alpha=0.7, vmin=0., vmax=0.8)

labels = stnnames[0:]
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+10000, ypt+5000, label,fontsize=5)

cbar = map.colorbar(location='bottom',pad="5%")
cbar.set_label('correlation')

plt.title('Daily precip RSME and correlations between vrcesm and SA-OBS')
plt.savefig(outdir+"vrcesm_prect_vs_saobs_station_obs_scatter_daily.pdf")

#scatter plot correlation and rmse on map for fv1x1
plt.clf()
map = Basemap(projection='merc', lat_0=17, lon_0=100.5,resolution = 'h', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

map.drawcoastlines()
map.drawcountries()
#map.fillcontinents(color = 'white')
#map.drawmapboundary()

latsmap = stnlats[0:]
lonsmap = stnlons[0:]

x,y = map(lonsmap, latsmap)
map.scatter(x, y, s=10*fvstnrse[0:], marker="o",c=fvstncor[0:],cmap=cm.coolwarm,alpha=0.7,vmin=0., vmax=0.8)

labels = stnnames[0:]
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+10000, ypt+5000, label,fontsize=5)

cbar = map.colorbar(location='bottom',pad="5%")
cbar.set_label('correlation')

plt.title('Daily precip RSME and correlations between fv0.9x1.25 and SA-OBS')
plt.savefig(outdir+"fv1x1_prect_vs_saobs_station_obs_scatter_daily.pdf")

#scatter plot correlation and rmse on map for ne30
plt.clf()
map = Basemap(projection='merc', lat_0=17, lon_0=100.5,resolution = 'h', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

map.drawcoastlines()
map.drawcountries()
#map.fillcontinents(color = 'white')
#map.drawmapboundary()

latsmap = stnlats[0:]
lonsmap = stnlons[0:]

x,y = map(lonsmap, latsmap)
map.scatter(x, y, s=10*nestnrse[0:], marker="o",c=nestncor[0:],cmap=cm.coolwarm,alpha=0.7, vmin=0., vmax=0.8)

labels = stnnames[0:]
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+10000, ypt+5000, label,fontsize=5)

cbar = map.colorbar(location='bottom',pad="5%")
cbar.set_label('correlation')

plt.title('Daily precip RSME and correlations between ne30 and SA-OBS')
plt.savefig(outdir+"ne30_prect_vs_saobs_station_obs_scatter_daily.pdf")

##################################################################################
#deal with monthly precip
print("Start to draw scatter plot for monthly statics...")
#scatter plot correlation and rmse on map for vrcesm
plt.clf()
map = Basemap(projection='merc', lat_0=17, lon_0=100.5,resolution = 'h', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

map.drawcoastlines()
map.drawcountries()
#map.fillcontinents(color = 'white')
#map.drawmapboundary()

latsmap = stnlats[0:]
lonsmap = stnlons[0:]

x,y = map(lonsmap, latsmap)
map.scatter(x, y, s=30*vrstnrsemon[0:], marker="o",c=vrstncormon[0:],cmap=cm.coolwarm,alpha=0.7,vmin=0., vmax=0.8)

labels = stnnames[0:]
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+10000, ypt+5000, label,fontsize=5)

cbar = map.colorbar(location='bottom',pad="5%")
cbar.set_label('correlation')

plt.title('Monthly precip RSME and correlations between vrcesm and SA-OBS')
plt.savefig(outdir+"vrcesm_prect_vs_saobs_station_obs_scatter_monthly.pdf")

#scatter plot correlation and rmse on map for fv1x1
plt.clf()
map = Basemap(projection='merc', lat_0=17, lon_0=100.5,resolution = 'h', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

map.drawcoastlines()
map.drawcountries()
#map.fillcontinents(color = 'white')
#map.drawmapboundary()

latsmap = stnlats[0:]
lonsmap = stnlons[0:]

x,y = map(lonsmap, latsmap)
map.scatter(x, y, s=30*fvstnrsemon[0:], marker="o",c=fvstncormon[0:],cmap=cm.coolwarm,alpha=0.7, vmin=0., vmax=0.8)

labels = stnnames[0:]
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+10000, ypt+5000, label,fontsize=5)

cbar = map.colorbar(location='bottom',pad="5%")
cbar.set_label('correlation')

plt.title('Monthly precip RSME and correlations between fv0.9x1.25 and SA-OBS')
plt.savefig(outdir+"fv1x1_prect_vs_saobs_station_obs_scatter_monthly.pdf")

#scatter plot correlation and rmse on map for ne30
plt.clf()
map = Basemap(projection='merc', lat_0=17, lon_0=100.5,resolution = 'h', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

map.drawcoastlines()
map.drawcountries()
#map.fillcontinents(color = 'white')
#map.drawmapboundary()

latsmap = stnlats[0:]
lonsmap = stnlons[0:]

x,y = map(lonsmap, latsmap)
map.scatter(x, y, s=30*nestnrsemon[0:], marker="o",c=nestncormon[0:],cmap=cm.coolwarm,alpha=0.7,vmin=0., vmax=0.8)

labels = stnnames[0:]
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+10000, ypt+5000, label,fontsize=5)

cbar = map.colorbar(location='bottom',pad="5%")
cbar.set_label('correlation')

plt.title('Monthly precip RSME and correlations between ne30 and SA-OBS')
plt.savefig(outdir+"ne30_prect_vs_saobs_station_obs_scatter_monthly.pdf")

##################################################################################
print("Start to draw histogram for all stations...")
#draw hist for all stations
plt.clf()

#sapre[np.isnan(sapre)] = 0.
#vrpre[np.isnan(vrpre)] = 0.
#fvpre[np.isnan(fvpre)] = 0.

#define bin list
#binmax = np.amax([np.amax(sa_data['pre']),np.amax(vr_data['pre']),np.amax(fv_data['pre'])])*1./2.
binmax  = np.amax(sapreall[~np.isnan(sapreall)])*1./2.
binarray = np.arange(0,binmax,binmax/30)
#print(binmax)
#print(sapreall[~np.isnan(sapreall)])
#print(binarray)

y,binEdges=np.histogram(sapreall[~np.isnan(sapreall)],bins=binarray)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#print(bincenters)
plt.plot(bincenters,y,c='k', label = 'SA-OBS')

y,binEdges=np.histogram(vrpreall[~np.isnan(vrpreall)],bins=binarray)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,c='r', label = 'vrcesm')

y,binEdges=np.histogram(fvpreall[~np.isnan(fvpreall)],bins=binarray)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,c='b', label = 'fv0.9x1.25')

y,binEdges=np.histogram(nepreall[~np.isnan(nepreall)],bins=binarray)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,c='g', label = 'ne30')

plt.yscale('log')
plt.legend(loc='upper right')

plt.title("Total precip distribution at all stations")
plt.ylabel("Days")
plt.xlabel("Precip(mm/day)")

plt.savefig(outdir+"vrcesm_prect_vs_saobs_station_obs_hist_all.pdf")

##################################################################################
print('Start to plot for selected stations...')
sapreall = np.reshape(sapreall,(len(stnids),365*(endyear-iniyear+1)))
vrpreall = np.reshape(vrpreall,(len(stnids),365*(endyear-iniyear+1)))
fvpreall = np.reshape(fvpreall,(len(stnids),365*(endyear-iniyear+1)))
nepreall = np.reshape(nepreall,(len(stnids),365*(endyear-iniyear+1)))
print(np.shape(vrpreall))
#draw panel plot for selected data
plt.clf()
fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)
for idx,stnname in enumerate(stnnamesselect):
    #get stnid,country name and stnname
    stnidx = np.where(stnnames==stnname)
    stnid = stnids[stnidx[0][0]]
    stnlat = stnlats[stnidx[0][0]]
    stnlon = stnlons[stnidx[0][0]]
    countryname = countrynames[stnidx[0][0]] 
 
    #print(stnidx[0][0])
    #print(stnid)
    #print(stnname)
    #print(countryname)

    #print(np.shape(vrpreall[stnidx[0][0],:]))

    print("station "+str(idx+1)+"/"+str(len(stnnamesselect))+" is: "+stnname+" in "+countryname+" at "+str(stnlat)+"/"+str(stnlon)+" with an station id "+str(stnid).zfill(6))

    #set time range for each dataset
    vr_data = pd.DataFrame({
        'date': date,
        'pre':vrpreall[stnidx[0][0],:]
    })

    fv_data = pd.DataFrame({
        'date': date,
        'pre':fvpreall[stnidx[0][0],:]
    })

    ne_data = pd.DataFrame({
        'date': date,
        'pre':nepreall[stnidx[0][0],:]
    })

    sa_data = pd.DataFrame({
        'date': date,
        'pre':sapreall[stnidx[0][0],:]
    })

    ##################################################################################
    #plot annual maximum precip for each station
    vr_data = vr_data.set_index(['date'])
    vr_data_annmax = vr_data.resample('A').max()
    #print(vr_data_annmax)

    fv_data = fv_data.set_index(['date'])
    fv_data_annmax = fv_data.resample('A').max()

    ne_data = ne_data.set_index(['date'])
    ne_data_annmax = ne_data.resample('A').max()

    sa_data = sa_data.set_index(['date'])
    sa_data_annmax = sa_data.resample('A').max()

    ax1 = fig1.add_subplot(3, 3, idx+1)
    ax1.plot(yearts,sa_data_annmax['pre'], c='k', label = 'SA-OBS')
    ax1.plot(yearts,vr_data_annmax['pre'], c='r', label = 'vrcesm')
    ax1.plot(yearts,fv_data_annmax['pre'], c='b', label = 'fv0.9x1.25')
    ax1.plot(yearts,ne_data_annmax['pre'], c='g', label = 'ne30')
    ax1.legend(loc='upper left',fontsize=5.)

    ax1.set_xticks(yearts)
    ax1.set_yticks(np.arange(0,240,40))

    ax1.set_title(str(stnid).zfill(6)+": "+stnname+" in "+countryname,size=5.)
    if (idx%3)==0:
        ax1.set_ylabel("Maximum precip (mm/day)",fontsize=5.)
        ax1.set_yticklabels(np.arange(0,240,40),fontsize=3.5)
    else:
        ax1.set_yticklabels([],fontsize=4.)
    if (idx>5):
       ax1.set_xlabel("Year",fontsize=5.)
       ax1.set_xticklabels(yearts,fontsize=3.5)
    else:
       ax1.set_xticklabels([],fontsize=4.)

    ##################################################################################
    #plot monthly precip for each station
    vr_data_mon = vr_data.resample('M').mean()

    fv_data_mon = fv_data.resample('M').mean()

    ne_data_mon = ne_data.resample('M').mean()

    sa_data_mon = sa_data.resample('M').mean()

    ax2 = fig2.add_subplot(3, 3, idx+1)
    ax2.plot(monts,sa_data_mon['pre'], c='k', label = 'SA-OBS',linewidth=0.8)
    ax2.plot(monts,vr_data_mon['pre'], c='r', label = 'vrcesm',linewidth=0.8)
    ax2.plot(monts,fv_data_mon['pre'], c='b', label = 'fv0.9x1.25',linewidth=0.8)
    ax2.plot(monts,ne_data_mon['pre'], c='g', label = 'ne30',linewidth=0.8)
    ax2.legend(loc='upper left',fontsize=5.)

    ax2.set_xticks(yearmidts)
    ax2.set_yticks(np.arange(0,30,5))

    ax2.set_title(str(stnid).zfill(6)+": "+stnname+" in "+countryname,size=5.)
    if (idx%3)==0:
        ax2.set_ylabel("Total precip (mm/day)",fontsize=5.)
        ax2.set_yticklabels(np.arange(0,30,5),fontsize=4.)
    else:
        ax2.set_yticklabels([],fontsize=4.)
    if (idx>5):
        ax2.set_xlabel("Time",fontsize=5.)
        ax2.set_xticklabels(yearmidtslabel,fontsize=4.)
    else:
        ax2.set_xticklabels([],fontsize=4.)

    ##################################################################################
    #define bin list
    binmax  = np.amax(sa_data['pre'])*1./2.
    binarray = np.arange(0,binmax,binmax/30)

    ax3 = fig3.add_subplot(3, 3, idx+1)
    y,binEdges=np.histogram(sa_data.loc[~np.isnan(sapreall[stnidx[0][0],:]),'pre'],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    ax3.plot(bincenters,y,c='k', label = 'SA-OBS')

    y,binEdges=np.histogram(vr_data.loc[~np.isnan(sapreall[stnidx[0][0],:]),'pre'],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    ax3.plot(bincenters,y,c='r', label = 'vrcesm')

    y,binEdges=np.histogram(fv_data.loc[~np.isnan(sapreall[stnidx[0][0],:]),'pre'],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    ax3.plot(bincenters,y,c='b', label = 'fv0.9x1.25')

    y,binEdges=np.histogram(ne_data.loc[~np.isnan(sapreall[stnidx[0][0],:]),'pre'],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    ax3.plot(bincenters,y,c='g', label = 'ne30')

    ax3.set_yscale('log')
    ax3.legend(loc='upper right',fontsize=5.)

    ax3.set_xticks(np.arange(0,125,25))
    ax3.set_yticks([0.6,1,10,100,1000,10000])
    ax3.get_yaxis().set_major_formatter(ScalarFormatter())
    #ax3.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax3.yaxis.set_minor_formatter(NullFormatter())
    #formatter = FuncFormatter(log_10_product)
    #ax3.yaxis.set_major_formatter(formatter)
    ax3.set_ylim(0.6, 10000)

    ax3.set_title(str(stnid).zfill(6)+": "+stnname+" in "+countryname,size=5.)
    if (idx%3)==0:
        ax3.set_ylabel("Days",fontsize=5.)
        ax3.set_yticklabels([0.6,1,10,100,1000,10000],fontsize=4.)
    else:
        ax3.set_yticklabels([],fontsize=4.)
    if (idx>5):
        ax3.set_xlabel("Precip(mm/day)",fontsize=5.)
        ax3.set_xticklabels(np.arange(0,125,25),fontsize=4.)
    else:
        ax3.set_xticklabels([],fontsize=4.)

fig1.suptitle("Annual Maximum precip at stations")
fig1.savefig(outdir+"vrcesm_prect_vs_saobs_station_obs_maxts_selected.pdf")

fig2.suptitle("Monthly total precip at stations")
fig2.savefig(outdir+"vrcesm_prect_vs_saobs_station_obs_monts_selected.pdf")

fig3.suptitle("Total precip distribution at stations")
fig3.savefig(outdir+"vrcesm_prect_vs_saobs_station_obs_hist_selected.pdf")

#close output log
outlog.close()
