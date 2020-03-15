#this script is used to read station observations from cru data and compared with vrcesm output
#by Harry Li


#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm

#set up CRU pre observations directory and filename
obsdir = "/scratch/d/dylan/harryli/gpcdata/obsdataset/CRU/pre/"
obsfname = "pre.1704241136.clean.dtb"

#set up vrcesm and fv1x1 data directories and filenames
vrcesmdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/vrseasia_AMIP_1979_to_2005/atm/hist/"
fv1x1dir = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f09_f09_AMIP_1979_to_2005/atm/hist/"
vrcesmfname = "fv02_prec_vrseasia_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
fv1x1fname = "PREC_f09_f09_AMIP_1979_to_2005.cam.h0.1979-2005.nc"

#set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/graphs/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_obs/stations/CRU/"
outlog = open(outdir+"vrcesm_prect_stns_vs_cru_line_plot_output.log", "w")

#define country name
country = ["THAILAND","VIETNAM","CAMBODIA","MYANMAR"]

#define inital year and end year
iniyear = 1995
endyear = 2004

#define days of each month in the leap year and normal year
leapmonthday = np.array([31,29,31,30,31,30,31,31,30,31,30,31],dtype=np.float)
monthday = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.float)
month = np.arange(1,13,1)
mname = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

#create time series
date = np.array(str(iniyear)+'-01', dtype=np.datetime64)
date = date + np.arange(120)

#read observations from CRU
print("Start to read station observations from CRU...")
outlog.write("Start to read station observations from CRU...\n")
fopen = open(obsdir+obsfname,"r")
linereader = fopen.readlines()

#define function to split data
def datasplit(str):
    tempstr = [str[0:4]]
    tempstr.extend([ str[start:start+5] for start in range(4, 60, 5) ])
    return tempstr

#define a function to calculate root mean square error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#read data
lineno = 0
iscountry = False
predata = np.array([],dtype=float)
stndata = []
currpre = np.array([],dtype=float)
currstn = []
while lineno < len(linereader):
    #check if this line is a headline and split the line
    if len(linereader[lineno][64:].strip(' '))>1:
        #print(len(linereader))
        linelist = linereader[lineno].split()
        #print(linelist)
    else:
        #print(linereader[lineno])
        linelist = datasplit(linereader[lineno])
        #print(linelist)
    #print(linelist)
    #print(len(linelist))
    
    #For the data lines, if it is the correct country then read the data from iniyear to endyear
    if len(linereader[lineno][64:].strip(' '))==1:
        if iscountry:
            idxyear = int(linelist[0])
            if idxyear==iniyear:
                currpre = np.append(currpre,np.asfarray(np.array(linelist[1:]),float))
            if (idxyear>iniyear) and (idxyear<endyear):
                currpre = np.append(currpre,np.asfarray(np.array(linelist[1:]),float))
            #At the endyear, if no missing data from iniyear to endyear then record the precip into predata and station info in stndata
            if idxyear==endyear:
                currpre = np.append(currpre,np.asfarray(np.array(linelist[1:]),float))
                if (len(currpre)==(endyear-iniyear+1)*12) and not(np.any(currpre ==-9999)):
                    predata = np.concatenate((predata,currpre))
                    stndata.append(currstn)

    #For the headline, check if the station located in the correct country
    if len(linereader[lineno][64:].strip(' '))>1:
        outlog.write("Current station is "+" ".join(str(x) for x in linelist[4:-3])+" in "+linelist[-3]+"\n")
        print(("Current station is "+" ".join(str(x) for x in linelist[4:-3])+" in "+linelist[-3]))
        if linelist[-3] in country:
            iscountry = True
            currpre = np.array([],dtype=float)
            currstn = linelist
        else:
            iscountry = False
            currpre = np.array([],dtype=float)
            currstn = []
        lineno = lineno + 1
    #print(currpre)
    #print(iscountry)
    lineno = lineno + 1
       
#convert to mm and reshape the data             
predata = predata/10
predata = np.reshape(predata,(len(stndata),12*(endyear-iniyear+1)))

#print(predata)
#print(stndata)
print("the number of predata is "+str(len(predata)))
outlog.write("the number of predata is "+str(len(predata))+"\n")   
print(str(len(stndata))+" stations are selected in "+" ".join(x for x in country)+". The station information is shown as following:")
outlog.write(str(len(stndata))+" stations are selected in "+" ".join(x for x in country)+". The station information is shown as following:\n")
for idx,p in enumerate(stndata):
    print("station  "+str(idx+1)+" is "+" ".join(x for x in stndata[idx]))
    outlog.write("station  "+str(idx+1)+" is "+" ".join(x for x in stndata[idx])+"\n")
    print(predata[idx])
    outlog.write(" ".join(str(x) for x in predata[idx])+"\n")

print("finish reading station observations from CRU...")
outlog.write("finish reading station observations from CRU...\n")

#open vrcesm file and fv1x1 file
vrcesmdata = Dataset(vrcesmdir+vrcesmfname)
fv1x1data  = Dataset(fv1x1dir+fv1x1fname)

#read lat/lon grids
vrlats = vrcesmdata.variables['lat'][:]
vrlons = vrcesmdata.variables['lon'][:]

fvlats = fv1x1data.variables['lat'][:]
fvlons = fv1x1data.variables['lon'][:]

print("Comparing the CRU station observations with vrcesm and fv1x1...")
outlog.write("Comparing the CRU station observations with vrcesm and fv1x1...\n")

#define station statistics
stnname   = np.array([])
allstnlat = np.array([],dtype=float)
allstnlon = np.array([],dtype=float)
vrstnstd = np.array([],dtype=float)
vrstnrse = np.array([],dtype=float)
vrstncor = np.array([],dtype=float)
fvstnstd = np.array([],dtype=float)
fvstnrse = np.array([],dtype=float)
fvstncor = np.array([],dtype=float)

for idx,stninfo in enumerate(stndata):
    print("Currently dealing with the  station "+str(idx+1)+"/"+str(len(stndata))+": "+" ".join(str(x) for x in stninfo[4:-3])+" in "+stninfo[-3])
    outlog.write("Currently dealing with the  station "+str(idx+1)+"/"+str(len(stndata))+": "+" ".join(str(x) for x in stninfo[4:-3])+" in "+stninfo[-3]+"\n")
    
    #get the station location
    stnlat = float(stninfo[1])/100
    stnlon = float(stninfo[2])/100

    #find the station in vrcesm and fv1x1 grids
    vrlat_idx = np.abs(vrlats - stnlat).argmin()
    vrlon_idx = np.abs(vrlons - stnlon).argmin()
    fvlat_idx = np.abs(fvlats - stnlat).argmin()
    fvlon_idx = np.abs(fvlons - stnlon).argmin()

    #get the station precip in vrcesm and fv1x1 data and convert to mm/day
    vrpre = vrcesmdata.variables['PRECT'][(iniyear-1979)*12:(endyear-1979+1)*12,vrlat_idx, vrlon_idx]
    fvpre = fv1x1data.variables['PRECT'][(iniyear-1979)*12:(endyear-1979+1)*12,fvlat_idx, fvlon_idx]
    vrpre = vrpre*86400*1000
    fvpre = fvpre*86400*1000

    #get the station observations in CRU data and convert it to mm/day
    stnpre = predata[idx]
    for iyear in np.arange(iniyear,endyear+1,1):
        #determin if the year is leap year
        #print(iyear)
        if (iyear%4==0) and (iyear%100!=0 or iyear%400==0):
            #print(stnpre[(iyear-iniyear)*12:(iyear-iniyear+1)*12])
            stnpre[(iyear-iniyear)*12:(iyear-iniyear+1)*12] = stnpre[(iyear-iniyear)*12:(iyear-iniyear+1)*12]/leapmonthday
            #print(stnpre[(iyear-iniyear)*12:(iyear-iniyear+1)*12])
        else:
            #print(stnpre[(iyear-iniyear)*12:(iyear-iniyear+1)*12])
            stnpre[(iyear-iniyear)*12:(iyear-iniyear+1)*12] = stnpre[(iyear-iniyear)*12:(iyear-iniyear+1)*12]/monthday
            #print(stnpre[(iyear-iniyear)*12:(iyear-iniyear+1)*12])
    
    #print(predata[idx])
    #print(stnpre)

    plt.clf()

    plt.plot(date,stnpre, c='k', label = 'CRU')
    plt.plot(date,vrpre, c='r', label = 'vrcesm')
    plt.plot(date,fvpre, c='b', label = 'fv1x1')
    plt.legend(loc='upper left')

    plt.title("Total precip in the station "+stninfo[0]+": "+" ".join(str(x) for x in stninfo[4:-3])+" in "+stninfo[-3])
    plt.ylabel("Total precip (mm/day)")
    plt.xlabel("Time")

    plt.savefig(outdir+"vrcesm_prect_vs_cru_station_obs_ts_"+str(idx+1)+"_"+stninfo[-3]+".pdf")

    #plot the long term average
    stnpreseas = np.array([],dtype=float)
    vrpreseas = np.array([],dtype=float)
    fvpreseas = np.array([],dtype=float)
    for k in range(12):
        stnpreseas = np.append(stnpreseas,np.mean(stnpre[k::12]))
        vrpreseas  = np.append(vrpreseas,np.mean(vrpre[k::12]))
        fvpreseas  = np.append(fvpreseas,np.mean(fvpre[k::12]))
    #print(vrpreseas)
    
    plt.clf()
    
    plt.plot(month,stnpreseas, c='k', label = 'CRU')
    plt.plot(month,vrpreseas, c='r', label = 'vrcesm')
    plt.plot(month,fvpreseas, c='b', label = 'fv1x1')
    plt.legend(loc='upper left')

    plt.title("Total precip in the station "+stninfo[0]+": "+" ".join(str(x) for x in stninfo[4:-3])+" in "+stninfo[-3])
    plt.ylabel("Total precip (mm/day)")
    plt.xlabel("Month")
    plt.xticks(np.arange(1,13,1), mname)

    plt.savefig(outdir+"vrcesm_prect_vs_cru_station_obs_seasonality_"+str(idx+1)+"_"+stninfo[-3]+".pdf")

    #calculate the statistics for taylor diagram and scatter plot
    stnname   = np.append(stnname," ".join(str(x) for x in stninfo[4:-3]))
    allstnlat = np.append(allstnlat,stnlat)
    allstnlon = np.append(allstnlon,stnlon)  
    vrstnstd  = np.append(vrstnstd,np.std(vrpre))
    fvstnstd  = np.append(fvstnstd,np.std(fvpre))
    vrstncor  = np.append(vrstncor,pearsonr(vrpre,stnpre)[0])
    fvstncor  = np.append(fvstncor,pearsonr(fvpre,stnpre)[0])
    vrstnrse  = np.append(vrstnrse,rmse(vrpre,stnpre))
    fvstnrse  = np.append(fvstnrse,rmse(fvpre,stnpre))

print(stnname)
print("stations lats are: "+" ".join(str(x) for x in allstnlat))
print("stations lons are: "+" ".join(str(x) for x in allstnlon))
print("vrcesm standard deviations are: "+" ".join(str(x) for x in vrstnstd))
print("vrcesm correlations are: "+" ".join(str(x) for x in vrstncor))
print("vrcesm root-mean-square-errors are: "+" ".join(str(x) for x in vrstnrse))
print("fv1x1 standard deviations are: "+" ".join(str(x) for x in fvstnstd))
print("fv1x1 correlations are: "+" ".join(str(x) for x in fvstncor))
print("fv1x1 root-mean-square-errors are: "+" ".join(str(x) for x in fvstnrse))

outlog.write("stations lats are: "+" ".join(str(x) for x in allstnlat)+"\n")
outlog.write("stations lons are: "+" ".join(str(x) for x in allstnlon)+"\n")
outlog.write("vrcesm standard deviations are: "+" ".join(str(x) for x in vrstnstd)+"\n")
outlog.write("vrcesm correlations are: "+" ".join(str(x) for x in vrstncor)+"\n")
outlog.write("vrcesm root-mean-square-errors are: "+" ".join(str(x) for x in vrstnrse)+"\n")
outlog.write("fv1x1 standard deviations are: "+" ".join(str(x) for x in fvstnstd)+"\n")
outlog.write("fv1x1 correlations are: "+" ".join(str(x) for x in fvstncor)+"\n")
outlog.write("fv1x1 root-mean-square-errors are: "+" ".join(str(x) for x in fvstnrse)+"\n")

#scatter plot correlation and rmse on map for vrcesm
plt.clf()
map = Basemap(projection='merc', lat_0=17, lon_0=100.5,resolution = 'h', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)
 
map.drawcoastlines()
map.drawcountries()
#map.fillcontinents(color = 'white')
#map.drawmapboundary()

latsmap = allstnlat[0:-1]
lonsmap = allstnlon[0:-1]

x,y = map(lonsmap, latsmap)
map.scatter(x, y, s=30*vrstnrse[0:-1], marker="o",c=vrstncor[0:-1],cmap=cm.coolwarm,alpha=0.7)

labels = stnname[0:-1]
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+10000, ypt+5000, label,fontsize=5)

cbar = map.colorbar(location='bottom',pad="5%")
cbar.set_label('correlation')

plt.savefig(outdir+"vrcesm_prect_vs_cru_station_obs_scatter.pdf")

#scatter plot correlation and rmse on map for fv1x1
plt.clf()
map = Basemap(projection='merc', lat_0=17, lon_0=100.5,resolution = 'h', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

map.drawcoastlines()
map.drawcountries()
#map.fillcontinents(color = 'white')
#map.drawmapboundary()

latsmap = allstnlat[0:-1]
lonsmap = allstnlon[0:-1]

x,y = map(lonsmap, latsmap)
map.scatter(x, y, s=30*fvstnrse[0:-1], marker="o",c=fvstncor[0:-1],cmap=cm.coolwarm,alpha=0.7)

labels = stnname[0:-1]
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+10000, ypt+5000, label,fontsize=5)

cbar = map.colorbar(location='bottom',pad="5%")
cbar.set_label('correlation')

plt.savefig(outdir+"fv1x1_prect_vs_cru_station_obs_scatter.pdf")
   
#close output log
outlog.close()
