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
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
from scipy.optimize import curve_fit


#set up data directories and filenames
case1 = "vrseasia_19501959_OBS"
case2 = "vrseasia_20002010_OBS"
case3 = "vrseasia_20002009_OBS_SUBAERSST_CESM1CAM5_SST"
case4 = "vrseasia_20002009_OBS_AEREMIS1950"
case5 = "vrseasia_20002009_OBS_AEREMIS1950_SUBAERSST_CESM1CAM5_SST"

nick1 = r'$S_{CTRL}$'
nick2 = r'$S_{2000}A_{2000}$'
nick3 = r'$S_{PERT}A_{2000}$'
nick4 = r'$S_{2000}A_{1950}$'
nick5 = r'$S_{PERT}A_{1950}$'

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
    rsmax = np.amax(rs) *2/3

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

def gpdfit(var,percentile):

    varpercent = np.percentile(var,percentile)
    varsub = var[var>varpercent] - varpercent

    count = 0
    varlen = len(varsub)
    print(varlen)
    var_ks = 0.
    var_best = []
    var_nbins = 0
     
    for ibins in np.arange(1.*varlen/3,1.*varlen-2,1.*(varlen)*2/3/200):   #(1.*prelen-1.*prelen/3)/1000):
        ibins = int(ibins)

        print('No.'+str(count)+'('+str(ibins)+' bins): ')
        popt,pcov,tempscore = kstest(varsub,ibins)
        if tempscore.pvalue>var_ks:
            var_ks=tempscore.pvalue
            var_best=popt
            var_bestcov=pcov
            var_nbins = ibins
#        print(tempscore)
        count = count + 1

    print('best fit using curve_fit pdf based on hist and Kolmogorov-Smirnov test:')
    print(var_best)
    print(var_ks)

    return var_best,var_ks,var_bestcov,var_nbins,varsub,varpercent


def plothist(var1,var2,var3,var4,var5,opt):

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    binmax   = max(np.amax(var1),np.amax(var2),np.amax(var3),np.amax(var4),np.amax(var5)) *2/3
    binmin   = min(np.amin(var1),np.amin(var2),np.amin(var3),np.amin(var4),np.amin(var5))
    binarray = np.arange(binmin,binmax,(binmax-binmin)/nbins)

    #case1
    binmax   = np.amax(var1) *2/3
    binmin   = np.amin(var1)
    binarray = np.arange(binmin,binmax,(binmax-binmin)/nbins)
    y,binEdges=np.histogram(var1,bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])    
    ax.plot(bincenters,y,c='black',linestyle='solid',linewidth=2, label = nick1)

    #case2
    binmax   = np.amax(var2) *2/3
    binmin   = np.amin(var2)
    binarray = np.arange(binmin,binmax,(binmax-binmin)/nbins)
    y,binEdges=np.histogram(var2,bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])    
    ax.plot(bincenters,y,c='red',linestyle='solid',linewidth=2, label = nick2)

    #case3
    binmax   = np.amax(var3) *2/3
    binmin   = np.amin(var3)
    binarray = np.arange(binmin,binmax,(binmax-binmin)/nbins)
    y,binEdges=np.histogram(var3,bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])    
    ax.plot(bincenters,y,c='green',linestyle='solid',linewidth=2, label = nick3)

    #case4
    binmax   = np.amax(var4) *2/3
    binmin   = np.amin(var4)
    binarray = np.arange(binmin,binmax,(binmax-binmin)/nbins)
    y,binEdges=np.histogram(var4,bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])    
    ax.plot(bincenters,y,c='blue',linestyle='solid',linewidth=2, label = nick4)

    #case5
    binmax   = np.amax(var5) *2/3
    binmin   = np.amin(var5)
    binarray = np.arange(binmin,binmax,(binmax-binmin)/nbins)
    y,binEdges=np.histogram(var5,bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])    
    ax.plot(bincenters,y,c='brown',linestyle='solid',linewidth=2, label = nick5)
    
    ax.set_yscale('log')
    ax.legend(loc='upper right',fontsize=6)

    ax.set_ylabel('Days',fontsize=7)
    ax.set_xlabel(varstr+' ['+var_unit+']',fontsize=7)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.minorticks_off()

    if (opt==1):
        plt.suptitle("Southeast Asia "+varstr+" exceeds "+str(percentile)+" percentile histogram",fontsize=10,y=0.95)
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_"+str(percentile)+"th_hist_month_"+str(inimonth)+"to"+str(endmonth)+".pdf")
    else:
        plt.suptitle("Southeast Asia "+varstr+" histogram",fontsize=10,y=0.95)
        plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_SEA_hist_month_"+str(inimonth)+"to"+str(endmonth)+".pdf")
    plt.close(fig)

    
def checkfit(varsub,var_best,var_ks,case,fname):

    plt.clf()
    fig = plt.figure()

    #pdf check
    ax1 = plt.subplot(3,1,1)
    x,xbins,ypdf = pdfcheck(varsub,nbins,var_best)
    anderson = ss.anderson_ksamp([varsub,gpd.rvs(var_best[0],0.,var_best[1],size=5000)])
    ax1.plot(x, ypdf,linestyle='solid',linewidth=1.5,label='GPD Fit')
    ax1.hist(varsub,bins=xbins,alpha=0.5,density=True,label=case)
    ax1.set_title('Anderson sig level='+str(anderson.significance_level)+', ks pval='+str(var_ks),fontsize=5)
    ax1.legend(loc='upper right',fontsize=5)
    ax1.set_ylabel('Frequency',fontsize=5)
    ax1.set_xlabel(varstr+' ['+var_unit+']',fontsize=5)
    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.yaxis.set_tick_params(labelsize=5)

    #qq plot
    ax2 = fig.add_subplot(3,1,2)
    res2 = ss.probplot(varsub, dist=ss.genpareto, sparams=(var_best[0],0.,var_best[1]), plot=ax2)
    ax2.set_title(case,fontsize=5)
    ax2.set_ylabel('Quantiles',fontsize=5)
#    ax2.set_xlabel('Quantiles',fontsize=5)
    ax2.xaxis.set_tick_params(labelsize=5)
    ax2.yaxis.set_tick_params(labelsize=5)

    #cdf check
    x,xbins,ycdf = cdfcheck(varsub,nbins,var_best)
    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(x, ycdf,linestyle='solid',linewidth=1.5,label='GPD Fit')
    ax3.hist(varsub,bins=xbins,alpha=0.5,density=True,histtype='step', cumulative=True,label=case)
#    ax3.set_title('ks pval='+str(var_ks),fontsize=5)
    ax3.legend(loc='lower right',fontsize=5)
    ax3.set_ylabel('CDF',fontsize=5)
    ax3.set_xlabel(varstr+' ['+var_unit+']',fontsize=5)
    ax3.xaxis.set_tick_params(labelsize=5)
    ax3.yaxis.set_tick_params(labelsize=5)

    fig.suptitle(case+' mainland SEA '+str(percentile)+'th percentile precip', y=0.95,fontsize=8)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+str(percentile)+"th_extremes_fit_check_"+str(inimonth)+"to"+str(endmonth)+"_"+fname+".pdf",bbox_inches='tight')
    plt.close(fig)  


def get_returns(year_returns,var_best,varpercent):

    var_return     = np.zeros(len(year_returns))
    var_return_up  = np.zeros(len(year_returns))
    var_return_bot = np.zeros(len(year_returns))

    for idx,iyear in enumerate(year_returns):
        m = iyear * ndays
        obsrate = (100.-percentile)/100.

        if (var_best[0]!=0):
            var_return[idx] = varpercent + var_best[1] / var_best[0] * ((m*obsrate) ** var_best[0] - 1)
            var_sigma =  (var_return[idx] - varpercent) * var_best[0] / ((m*obsrate) - 1)
            var_return_up[idx]  = var_return[idx] + var_sigma
            var_return_bot[idx] = var_return[idx] - var_sigma

    return var_return,var_return_up,var_return_bot

################################################################################################
#S1-open daily data
################################################################################################

#open data and read grids
fname1 = var_res+"_PREC_"+case1+".cam.h0.0001-0050.nc"
fdata1  = Dataset(expdir1+fname1)

#read lat/lon grids
lats = fdata1.variables['lat'][:]
lons = fdata1.variables['lon'][:]

# latitude/longitude  lower and upper regional index
reg_latli = np.abs(lats - reg_lats[0]).argmin()
reg_latui = np.abs(lats - reg_lats[1]).argmin()

reg_lonli = np.abs(lons - reg_lons[0]).argmin()
reg_lonui = np.abs(lons - reg_lons[1]).argmin()

nlats = reg_latui - reg_latli + 1
nlons = reg_lonui - reg_lonli + 1

print(nlats)
print(nlons)

#read land mask
rfname = "USGS-gtopo30_0.23x0.31_remap_c061107.nc"
rdata  = Dataset(rdir+rfname)
landfrac=rdata.variables['LANDFRAC'][:,:]
landfrac_reg = landfrac[reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ]
print(landfrac_reg)

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

    temp1 = fdata1.variables[varname][ iniday : endday , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ] * 86400 * 1000
    temp2 = fdata2.variables[varname][ iniday : endday , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ] * 86400 * 1000
    temp3 = fdata3.variables[varname][ iniday : endday , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ] * 86400 * 1000
    temp4 = fdata4.variables[varname][ iniday : endday , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ] * 86400 * 1000
    temp5 = fdata5.variables[varname][ iniday : endday , reg_latli:reg_latui+1 , reg_lonli:reg_lonui+1 ] * 86400 * 1000

    for iday in range(temp1.shape[0]):
        temp1[iday,landfrac_reg<0.5] = np.NaN 
        temp2[iday,landfrac_reg<0.5] = np.NaN
        temp3[iday,landfrac_reg<0.5] = np.NaN
        temp4[iday,landfrac_reg<0.5] = np.NaN
        temp5[iday,landfrac_reg<0.5] = np.NaN        

    var1[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp1.copy()
    var2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp2.copy()
    var3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp3.copy()
    var4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp4.copy()
    var5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays,:,:] = temp5.copy()

################################################################################################
#S2-plot histogram
################################################################################################

print(len(var1.flatten()))
var1_nomiss = var1.flatten()[~np.isnan(var1.flatten())]
var2_nomiss = var2.flatten()[~np.isnan(var2.flatten())]
var3_nomiss = var3.flatten()[~np.isnan(var3.flatten())]
var4_nomiss = var4.flatten()[~np.isnan(var4.flatten())]
var5_nomiss = var5.flatten()[~np.isnan(var5.flatten())]
print(len(var1_nomiss))

plothist(var1_nomiss,var2_nomiss,var3_nomiss,var4_nomiss,var5_nomiss,0)


################################################################################################
#S3-calculate for extremes
################################################################################################

print('gpd fit for '+case1+'...')
var1_best,var1_ks,var1_bestcov,var1_nbins,var1sub,var1percent = gpdfit(var1_nomiss,percentile)

print('gpd fit for '+case2+'...')
var2_best,var2_ks,var2_bestcov,var2_nbins,var2sub,var2percent = gpdfit(var2_nomiss,percentile)

print('gpd fit for '+case3+'...')
var3_best,var3_ks,var3_bestcov,var3_nbins,var3sub,var3percent = gpdfit(var3_nomiss,percentile)

print('gpd fit for '+case4+'...')
var4_best,var4_ks,var4_bestcov,var4_nbins,var4sub,var4percent = gpdfit(var4_nomiss,percentile)

print('gpd fit for '+case5+'...')
var5_best,var5_ks,var5_bestcov,var5_nbins,var5sub,var5percent = gpdfit(var5_nomiss,percentile)


plothist(var1sub+var1percent,var2sub+var2percent,var3sub+var3percent,var4sub+var4percent,var5sub+var5percent,1)

checkfit(var1sub,var1_best,var1_ks,case1,'case1')
checkfit(var2sub,var2_best,var2_ks,case2,'case2')
checkfit(var3sub,var3_best,var3_ks,case3,'case3')
checkfit(var4sub,var4_best,var4_ks,case4,'case4')
checkfit(var5sub,var5_best,var5_ks,case5,'case5')


################################################################################################
#S5-calculate N-year return level
################################################################################################ 

year_returns = np.arange(5.,504.,5.)

var1_return,var1_return_up,var1_return_bot = get_returns(year_returns,var1_best,var1percent)
var2_return,var2_return_up,var2_return_bot = get_returns(year_returns,var2_best,var2percent)
var3_return,var3_return_up,var3_return_bot = get_returns(year_returns,var3_best,var3percent)
var4_return,var4_return_up,var4_return_bot = get_returns(year_returns,var4_best,var4percent)
var5_return,var5_return_up,var5_return_bot = get_returns(year_returns,var5_best,var5percent)

plt.clf()
fig = plt.figure()

plt.plot(year_returns,var1_return, c='k', label = nick1)
plt.plot(year_returns,var2_return, c='r', label = nick2)
plt.plot(year_returns,var3_return, c='b', label = nick3)
plt.plot(year_returns,var4_return, c='g', label = nick4)
plt.plot(year_returns,var5_return, c='m', label = nick5)

#plt.plot(year_returns,var1_return_up, c='k', linestyle = 'dashed')
#plt.plot(year_returns,var2_return_up, c='r', linestyle = 'dashed')
#plt.plot(year_returns,var3_return_up, c='b', linestyle = 'dashed')
#plt.plot(year_returns,var4_return_up, c='g', linestyle = 'dashed')
#plt.plot(year_returns,var5_return_up, c='m', linestyle = 'dashed')

#plt.plot(year_returns,var1_return_bot, c='k', linestyle = 'dashed')
#plt.plot(year_returns,var2_return_bot, c='r', linestyle = 'dashed')
#plt.plot(year_returns,var3_return_bot, c='b', linestyle = 'dashed')
#plt.plot(year_returns,var4_return_bot, c='g', linestyle = 'dashed')
#plt.plot(year_returns,var5_return_bot, c='m', linestyle = 'dashed')

plt.legend(loc='upper left',fontsize=5)

plt.title("SEA Precip extreme return levels",fontsize=10,y=-1.05)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.ylabel(varstr+' '+var_unit,fontsize=7)
plt.xlabel('Return years',fontsize=7)

plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+str(percentile)+"th_extremes_return_levels_"+str(inimonth)+"to"+str(endmonth)+".pdf")
plt.close(fig)







