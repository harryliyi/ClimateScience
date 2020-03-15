#this script is used to compare vrcesm against observations
#here extremes is presented
#by Harry Li


#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter
plt.switch_backend('agg')
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
from scipy.optimize import curve_fit
import multiprocessing

#set up data directories and filenames
case1 = "vrseasia_19501959_OBS"
case2 = "vrseasia_20002010_OBS"
case3 = "vrseasia_20002009_OBS_SUBAERSST_CESM1CAM5_SST"
case4 = "vrseasia_20002009_OBS_AEREMIS1950"
case5 = "vrseasia_20002009_OBS_AEREMIS1950_SUBAERSST_CESM1CAM5_SST"

expdir1 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case1+"/atm/"
expdir2 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case2+"/atm/"
expdir3 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case3+"/atm/"
expdir4 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case4+"/atm/"
expdir5 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case5+"/atm/"

#set up land mask directory
rdir = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"

#set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/pre/extreme/"

#set up variable names and file name
varname = 'PRECT'

varstr  = "Total Precip"
var_res = "fv02"
varfname = 'prect'
var_unit= 'mm/day'

#define inital year and end year
iniyear = 2
endyear = 50

#define the contour plot region
latbounds = [ -20 , 50 ]
lonbounds = [ 40 , 160 ]

#define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

#month series
month = np.arange(1,13,1)
mname = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
mdays = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

#define the season period
inimonth = 6
endmonth = 8

iniday = np.sum(mdays[0:inimonth-1])
endday = np.sum(mdays[0:endmonth])
print(iniday)
print(endday)
ndays  = endday-iniday

#set up percentile
percentile = 99

#set up nbins
nbins = 50

#return levels
#year_returns = np.array([5,10,25,50,100,200,500])
year_return=150

################################################################################################
#S0-Define functions
################################################################################################

#define function
def pdffunc(x, c, loc, scale):
    return 1. / scale * (1. + c * ((x-loc)/scale)) ** (-1. - 1./c)

def cdffunc(x, c, loc, scale):
    return 1. - (1. + c * ((x-loc)/scale)) ** (- 1./c)

def pdffunc_noloc(x, c, scale):
    return 1. / scale * (1. + c * (x/scale)) ** (-1. - 1./c)

def cdffunc_noloc(x, c, scale):
    return 1. - (1. + c * (x/scale)) ** (- 1./c)

#define fitting process and ks test
def kstest(rs,ibins):
    rsmin = np.amin(rs)
    rsmax = np.amax(rs)

    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2.,rsmax,(rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin,rsmax + (rsmax-rsmin)/ibins/2.,(rsmax-rsmin)/ibins)

    hist, bin_edges = np.histogram(rs, bins=xbins, density=True)
    hist_tofitcdf = np.cumsum(hist)*(rsmax-rsmin)/ibins

    popt, pcov = curve_fit(cdffunc_noloc, x, hist_tofitcdf)
    tempscore = ss.kstest(rs,'genpareto',args=[popt[0],0.,popt[1]], alternative = 'two-sided')

    return popt,pcov,tempscore

#check fitting 
def cdfcheck(rs,ibins,popt):
    rsmin = np.amin(rs)
    rsmax = np.amax(rs)

    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2.,rsmax,(rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin,rsmax + (rsmax-rsmin)/ibins/2.,(rsmax-rsmin)/ibins)

    ycdf = cdffunc_noloc(x,popt[0],popt[1])

    return x,xbins,ycdf

def pdfcheck(rs,ibins,popt):
    rsmin = np.amin(rs)
    rsmax = np.amax(rs)

    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2.,rsmax,(rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin,rsmax + (rsmax-rsmin)/ibins/2.,(rsmax-rsmin)/ibins)

    ypdf = pdffunc_noloc(x,popt[0],popt[1])

    return x,xbins,ypdf


def getstats(var1,var2):
    n1=var1.shape[0]
    n2=var2.shape[0]

    var1mean = np.mean(var1,axis = 0)
    var2mean = np.mean(var2,axis = 0)
    var1std  = np.std(var1,axis = 0)
    var2std  = np.std(var2,axis = 0)

    vardiff  = var1mean - var2mean
    varttest = vardiff/np.sqrt(var1std**2/n1+var2std**2/n2)

    return vardiff,abs(varttest)

def getreturns(var):
     varpercent = np.percentile(var,percentile)
     varsub = var[var>varpercent] - varpercent

     var_fit = gpd.fit(varsub)
     var_fit = gpd.fit(varsub)

     var_return = varpercent + var_fit[2] / var_fit[0] * ((m*obsrate) ** var_fit[0] - 1)

     return var_return

def gpdfit(var1,var2,var3,var4,var5,percentile):

    pool = multiprocessing.Pool(processes=400)
#    var = var.reshape(ndays*(endyear-iniyear+1),nlats*nlons)
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    res5 = []

    for ilat in range(nlats):
        res1 = np.concatenate([res1,pool.map(getreturns, (var1[:,ilat,ilon] for ilon in range(nlons)) ) ])
        res2 = np.concatenate([res2,pool.map(getreturns, (var2[:,ilat,ilon] for ilon in range(nlons)) ) ])
        res3 = np.concatenate([res3,pool.map(getreturns, (var3[:,ilat,ilon] for ilon in range(nlons)) ) ])
        res4 = np.concatenate([res4,pool.map(getreturns, (var4[:,ilat,ilon] for ilon in range(nlons)) ) ])
        res5 = np.concatenate([res5,pool.map(getreturns, (var5[:,ilat,ilon] for ilon in range(nlons)) ) ])
        print('GPD fitting: '+str(round(1.*(ilat+1)/nlats,4)*100)+'%')        
    
    res1=np.array(res1)
    var1_return = res1.reshape(nlats,nlons)
    res2=np.array(res2)
    var2_return = res2.reshape(nlats,nlons)
    res3=np.array(res3)
    var3_return = res3.reshape(nlats,nlons)
    res4=np.array(res4)
    var4_return = res4.reshape(nlats,nlons)
    res5=np.array(res5)
    var5_return = res5.reshape(nlats,nlons)
    
    return var1_return,var2_return,var3_return,var4_return,var5_return

def plotextremes(lons,lats,var,relevel,titlestr,fname):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6,linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6,linewidth=0.1)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-125.,125.1,25.)
    cs = map.contourf(x,y,var,clevs,cmap=cm.BrBG,alpha = 0.9,extend="both")

    # add colorbar.
    cbar = map.colorbar(cs,location='bottom',pad="5%")
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+' ['+var_unit+']')

    # add title
    plt.title(titlestr+" JJA "+str(percentile)+"th percentile precip "+relevel+"-year return level",fontsize=10,y=1.08)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_"+str(percentile)+"_th_extremes_SEA_contour_"+str(inimonth)+"to"+str(endmonth)+"_"+relevel+"_"+fname+".pdf")

    plt.close(fig)


#plot for response
def plotdiff(lons,lats,res,relevel,forcingstr,forcingfname):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=6,linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=6,linewidth=0.1)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-90.,90.1,30.)
    norm = mpl.colors.BoundaryNorm(boundaries=clevs, ncolors=256)
    cs = map.pcolormesh(x,y,res,cmap=cm.BrBG,alpha = 0.9,norm=norm,vmax=75.,vmin=-75.)

    # add colorbar.
    cbar = map.colorbar(cs,location='bottom',pad="5%")
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+' ['+var_unit+']')

    plt.title(forcingstr+" "+varstr+" changes",fontsize=7,y=1.08)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_"+str(percentile)+"_th_extremes_SEA_contour_response_"+str(inimonth)+"to"+str(endmonth)+"_"+relevel+"_"+forcingfname+".pdf")


#plot for all responses together
def plotalldiff(lons,lats,res1,res2,res3,relevel):
    fig = plt.figure()
    #total response
    ax1 = fig.add_subplot(311)
    ax1.set_title(relevel+'-year return '+r'$\Delta_{total} Rp$'+str(percentile),fontsize=5,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4,linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4,linewidth=0.1)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-60,60.1,15.)
    norm = mpl.colors.BoundaryNorm(boundaries=clevs, ncolors=256)
    cs = map.pcolormesh(x,y,res1,cmap=cm.BrBG,alpha = 0.9,vmax=75.,vmin=-75.,norm=norm)

    #fast response
    ax2 = fig.add_subplot(312)
    ax2.set_title(relevel+'-year return '+r'$\Delta_{fast} Rp$'+str(percentile),fontsize=5,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4,linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4,linewidth=0.1)
    cs = map.pcolormesh(x,y,res2,cmap=cm.BrBG,alpha = 0.9,vmax=75.,vmin=-75.,norm=norm)

    #slow response
    ax3 = fig.add_subplot(313)
    ax3.set_title(relevel+'-year return '+r'$\Delta_{slow} Rp$'+str(percentile),fontsize=5,pad=3)
    map = Basemap(projection='cyl',llcrnrlat=latbounds[0],urcrnrlat=latbounds[1],llcrnrlon=lonbounds[0],urcrnrlon=lonbounds[1],resolution = 'l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1],20)
    meridians = np.arange(lonbounds[0],lonbounds[1],20)
    map.drawparallels(parallels, labels = [1, 0, 0, 0], fontsize=4,linewidth=0.1)
    map.drawmeridians(meridians, labels = [0, 0, 0, 1], fontsize=4,linewidth=0.1)
    cs = map.pcolormesh(x,y,res3,cmap=cm.BrBG,alpha = 0.9,vmax=75.,vmin=-75.,norm=norm)


    # add colorbar.
#    fig.subplots_adjust(right=0.7,hspace = 0.15)
    cbar_ax = fig.add_axes([0.69, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs,cax = cbar_ax,orientation='vertical',ticks=clevs)
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label(relevel+'-year return Rp'+str(percentile)+' [percent]',fontsize=4,labelpad=0.7)

    # add title
    plt.suptitle("Aerosol Responses "+varstr+" Rp"+str(percentile)+" changes",fontsize=8,y=0.95)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_"+str(percentile)+"_th_extremes_SEA_contour_response_"+str(inimonth)+"to"+str(endmonth)+"_"+relevel+"_aerosolsinone.pdf",bbox_inches='tight')


################################################################################################
#S1-open daily data
################################################################################################

#open data and read grids
fname1 = var_res+"_PREC_"+case1+".cam.h0.0001-0050.nc"
fdata1  = Dataset(expdir1+fname1)

#read lat/lon grids
lats = fdata1.variables['lat'][:]
lons = fdata1.variables['lon'][:]

# latitude/longitude  lower and upper contour index
latli = np.abs(lats - latbounds[0]).argmin()
latui = np.abs(lats - latbounds[1]).argmin()

lonli = np.abs(lons - lonbounds[0]).argmin()
lonui = np.abs(lons - lonbounds[1]).argmin()

lats  = lats[latli:latui+1]
lons  = lons[lonli:lonui+1]

nlats = latui - latli + 1
nlons = lonui - lonli + 1

print(nlats)
print(nlons)


obsrate = (100.-percentile)/100.

var1  = np.zeros(((endyear-iniyear+1)*ndays,nlats,nlons))
var2  = np.zeros(((endyear-iniyear+1)*ndays,nlats,nlons))
var3  = np.zeros(((endyear-iniyear+1)*ndays,nlats,nlons))
var4  = np.zeros(((endyear-iniyear+1)*ndays,nlats,nlons))
var5  = np.zeros(((endyear-iniyear+1)*ndays,nlats,nlons))

print('reading the data...')
for iyear in np.arange(iniyear,endyear+1,1):
    if (iyear<10):
        yearno = '000'+str(iyear)
    else:
        yearno = '00'+str(iyear)
    print('Current year is: '+yearno)

    fname1 = var_res+'_prect_'+case1+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname2 = var_res+'_prect_'+case2+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname3 = var_res+'_prect_'+case3+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname4 = var_res+'_prect_'+case4+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname5 = var_res+'_prect_'+case5+'.cam.h1.'+yearno+'-01-01-00000.nc'

    fdata1  = Dataset(expdir1+fname1)
    fdata2  = Dataset(expdir2+fname2)
    fdata3  = Dataset(expdir3+fname3)
    fdata4  = Dataset(expdir4+fname4)
    fdata5  = Dataset(expdir5+fname5)

    temp1 = fdata1.variables[varname][ iniday : endday , latli:latui+1 , lonli:lonui+1 ] * 86400 * 1000
    temp2 = fdata2.variables[varname][ iniday : endday , latli:latui+1 , lonli:lonui+1 ] * 86400 * 1000
    temp3 = fdata3.variables[varname][ iniday : endday , latli:latui+1 , lonli:lonui+1 ] * 86400 * 1000
    temp4 = fdata4.variables[varname][ iniday : endday , latli:latui+1 , lonli:lonui+1 ] * 86400 * 1000
    temp5 = fdata5.variables[varname][ iniday : endday , latli:latui+1 , lonli:lonui+1 ] * 86400 * 1000

    var1[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp1.copy()
    var2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp2.copy()
    var3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp3.copy()
    var4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp4.copy()
    var5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp5.copy()

################################################################################################
#S2-calculate for extremes
################################################################################################

m = year_return * ndays
obsrate = (100.-percentile)/100.

print('gpd fit ...')
var1_return,var2_return,var3_return,var4_return,var5_return = gpdfit(var1,var2,var3,var4,var5,percentile)

#print('gpd fit for '+case2+'...')
#var2_return = gpdfit(var2,percentile)

#print('gpd fit for '+case3+'...')
#var3_return = gpdfit(var3,percentile)

#print('gpd fit for '+case4+'...')
#var4_return = gpdfit(var4,percentile)

#print('gpd fit for '+case5+'...')
#var5_return = gpdfit(var5,percentile)


################################################################################################
#S2-plot for return levels
################################################################################################

plotextremes(lons,lats,var1_return,str(year_return),case1,'case1')
plotextremes(lons,lats,var2_return,str(year_return),case2,'case2')
plotextremes(lons,lats,var3_return,str(year_return),case3,'case3')
plotextremes(lons,lats,var4_return,str(year_return),case4,'case4')
plotextremes(lons,lats,var5_return,str(year_return),case5,'case5')



################################################################################################
#S3-plot for response
################################################################################################

print('plotting responses...')

#all aerosol foring
forcingstr   = "All aerosol forcings"
forcingfname = "allaerosols"
res          = var2_return - var5_return

plotdiff(lons,lats,res,str(year_return), forcingstr,forcingfname)


#aerosol fast response1
forcingstr   = "Aerosol fast response"
forcingfname = "fastaerosol1"
res          = var2_return - var4_return

plotdiff(lons,lats,res,str(year_return), forcingstr,forcingfname)


#aerosol slow response1
forcingstr   = "Aerosol slow response"
forcingfname = "slowaerosol1"
res          = var4_return - var5_return

plotdiff(lons,lats,res,str(year_return), forcingstr,forcingfname)


#aerosol fast response2
forcingstr   = "Aerosol fast response"
forcingfname = "fastaerosol2"
res          = var3_return - var5_return

plotdiff(lons,lats,res,str(year_return), forcingstr,forcingfname)


#aerosol slow response2
forcingstr   = "Aerosol slow response"
forcingfname = "slowaerosol2"
res          = var2_return - var3_return

plotdiff(lons,lats,res,str(year_return), forcingstr,forcingfname)


#GHG and natural forcing
forcingstr   = "GHG and natural forcings"
forcingfname = "GHGforcings"
res          = var5_return - var1_return

plotdiff(lons,lats,res,str(year_return), forcingstr,forcingfname)


#All forcings
forcingstr   = "All forcings"
forcingfname = "allforcings"
res          = var2_return - var1_return

plotdiff(lons,lats,res,str(year_return), forcingstr,forcingfname)



#################################################################################################
#plot all aerosol respoenses in one figure


res1 = (var2_return - var5_return)/var5_return*100
res2 = ((var2_return + var3_return - var4_return - var5_return)/2)/(var4_return+var5_return)*2*100
res3 = (var4_return + var2_return - var5_return - var3_return)/2/(var5_return+var3_return)*2*100

plotalldiff(lons,lats,res1,res2,res3,str(year_return))



