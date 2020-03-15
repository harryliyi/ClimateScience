# This script is used to read daily station precip from SA-OBS and compare the extremes with vrseasia and fv0.9x1.25 output
# Several steps are inplemented:
# S1-read precip data in mainland Southeast Asia from SA-OBS
# S2-read data from vrseasia and fv0.9x1.25 output and compute statics
# S3-test the mean residual life plot for SA-OBS data
#
# by Harry Li

#import libraries
import datetime as datetime
import math as math
import pandas as pd
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import matplotlib.cm as cm
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
from scipy.optimize import curve_fit
from netCDF4 import Dataset
import netCDF4
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# set up SA-OBS pre observations directory filename
obsdir = "/scratch/d/dylan/harryli/obsdataset/SA_OBS/countries/"
stnsum = "stations.txt"

# set up vrcesm and fv1x1 data directories and filenames
vrcesmdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/vrseasia_AMIP_1979_to_2005/atm/hist/"
fv1x1dir = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f09_f09_AMIP_1979_to_2005/atm/hist/"
ne30dir = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/ne30_ne30_AMIP_1979_to_2005/atm/hist/"
vrcesmfname = "fv02_prec_vrseasia_AMIP_1979_to_2005.cam.h1.1979-2005.nc"
# vrcesmfname = "fv09_prec_vrseasia_AMIP_1979_to_2005.cam.h1.1979-2005.nc"
fv1x1fname = "PREC_f09_f09_AMIP_1979_to_2005.cam.h1.1979-2005.nc"
ne30fname = "fv09_PREC_ne30_ne30_AMIP_1979_to_2005.cam.h1.1979-2005.nc"

# set up era interim data directories and filename
eradir = "/scratch/d/dylan/harryli/obsdataset/ERA_interim/pre/daily/"
erafname = "era_interim_tp_daily_19790101-20051231.nc"

# define inital year and end year
iniyear = 1980
endyear = 2005
inidate = datetime.datetime.strptime(str(iniyear)+'-01-01 12:00', '%Y-%m-%d %H:%M')
enddate = datetime.datetime.strptime(str(endyear)+'-12-31 12:00', '%Y-%m-%d %H:%M')

# set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/extremes/SAOBS_countries/" + \
    str(iniyear)+"-"+str(endyear)+"/vrseasia_regrid025x025_erainterim/"
outlog = open(outdir+"vrcesm_prect_stns_vs_saobs_extreme_output.log", "w")

# define country name
countries = ["Thailand", "Vietnam", "Cambodia", "Myanmar"]
countryids = ["TH", "VN", "KH", "MM"]

# create year series
yearts = np.array(str(iniyear), dtype=np.datetime64)
yearts = yearts + np.arange(endyear-iniyear+1)
yearmidtslabel = yearts[0::2]
# print(yearts)
# print(yearmidts)

# create time series
monts = np.array(str(iniyear)+'-01', dtype=np.datetime64)
monts = monts + np.arange((endyear-iniyear+1)*12)
yearmidts = monts[5::24]
print(yearmidts)

# create time series
date = pd.date_range(str(iniyear)+'-01-01', str(endyear)+'-12-31')
is_leap_day = (date.month == 2) & (date.day == 29)
date = date[~is_leap_day]
print(len(date))

# set up percentile
percentile = 97

# set up nbins
nbins = 50

################################################################################################
# S0-function definition
################################################################################################

# create a function to convert degree/minute/second to decimal


def deg2dec(x):
    xlist = x.split(":")
    if xlist[0][0] == "-":
        xdec = float(xlist[0])-float(xlist[1])/60.-float(xlist[2])/60./60.
    else:
        xdec = float(xlist[0])+float(xlist[1])/60.+float(xlist[2])/60./60.
    return xdec

# define a function to calculate root mean square error


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# define function


def pdffunc(x, c, loc, scale):
    return 1. / scale * (1. + c * ((x-loc)/scale)) ** (-1. - 1./c)


def cdffunc(x, c, loc, scale):
    return 1. - (1. + c * ((x-loc)/scale)) ** (- 1./c)


def pdffunc_noloc(x, c, scale):
    return 1. / scale * (1. + c * (x/scale)) ** (-1. - 1./c)


def cdffunc_noloc(x, c, scale):
    return 1. - (1. + c * (x/scale)) ** (- 1./c)

# define fitting process and ks test


def kstest(rs, ibins):
    rsmin = np.amin(rs)
    rsmax = np.amax(rs)

    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2., rsmax, (rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin, rsmax + (rsmax-rsmin)/ibins/2., (rsmax-rsmin)/ibins)

    hist, bin_edges = np.histogram(rs, bins=xbins, density=True)
    hist_tofitcdf = np.cumsum(hist)*(rsmax-rsmin)/ibins

    popt, pcov = curve_fit(cdffunc_noloc, x, hist_tofitcdf)
    tempscore = ss.kstest(rs, 'genpareto', args=[popt[0], 0., popt[1]], alternative='two-sided')

    return popt, pcov, tempscore

# check fitting


def cdfcheck(rs, ibins, popt):
    rsmin = np.amin(rs)
    rsmax = np.amax(rs)

    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2., rsmax, (rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin, rsmax + (rsmax-rsmin)/ibins/2., (rsmax-rsmin)/ibins)

    ycdf = cdffunc_noloc(x, popt[0], popt[1])

    return x, xbins, ycdf


def pdfcheck(rs, ibins, popt):
    rsmin = np.amin(rs)
    rsmax = np.amax(rs)

    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2., rsmax, (rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin, rsmax + (rsmax-rsmin)/ibins/2., (rsmax-rsmin)/ibins)

    ypdf = pdffunc_noloc(x, popt[0], popt[1])

    return x, xbins, ypdf

################################################################################################
# S1-read precip data in mainland Southeast Asia from SA-OBS
################################################################################################


print("Start to read data from SA-OBS...")
outlog.write("Start to read data from SA-OBS...\n")
# find the stn ids from stations.txt
fopen = open(obsdir+stnsum, "r")
linereader = fopen.readlines()[19:]

stnids = np.array([], dtype=int)
stnnames = np.array([])
countrynames = np.array([])
stnlats = np.array([], dtype=float)
stnlons = np.array([], dtype=float)

for lineno in range(len(linereader)):
    linelist = linereader[lineno].split(",")
    # print(linelist)
    #print(" ".join(x for x in linelist[1].split()))
    if linelist[2] in countryids:
        stnids = np.append(stnids, int(linelist[0]))
        stnnames = np.append(stnnames, " ".join(x for x in linelist[1].split()))
        countrynames = np.append(countrynames, countries[countryids.index(linelist[2])])
        stnlats = np.append(stnlats, deg2dec(linelist[3]))
        stnlons = np.append(stnlons, deg2dec(linelist[4]))
    print("Current station "+linelist[0]+" is "+" ".join(x for x in linelist[1].split()) +
          " in "+linelist[2]+" at "+str(deg2dec(linelist[3]))+"/"+str(deg2dec(linelist[4])))
    outlog.write("Current station "+linelist[0]+" is "+" ".join(x for x in linelist[1].split()) +
                 " in "+linelist[2]+" at "+str(deg2dec(linelist[3]))+"/"+str(deg2dec(linelist[4]))+"\n")


saobspre = np.array([], dtype=float)

# read precip data from each station
print("Totally "+str(len(stnids))+" stations are found. Their information is shown in following:")
for idx in range(len(stnids)):
    print("station "+str(idx+1)+"/"+str(len(stnids))+" is: "+stnnames[idx]+" in "+countrynames[idx]+" at "+str(
        stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6))
    outlog.write("station "+str(idx+1)+"/"+str(len(stnids))+" is: "+stnnames[idx]+" in "+countrynames[idx]+" at "+str(
        stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6)+"\n")

    # open file from each station
    obsfname = "RR_STAID"+str(stnids[idx]).zfill(6)+".txt"
    fopen = open(obsdir+obsfname, "r")
    linereader = fopen.readlines()[21:]

    # read data from station between given years
    lineno = 0
    currpre = np.array([], dtype=float)
    missingcount = 0.
    print(linereader[lineno].split(","))
    # while int(linereader[lineno].split(",")[2][0:4])<=endyear:
    while int(linereader[lineno].split(",")[1][0:4]) <= endyear:
        # print(linereader[lineno].split(",")[2])
        if (int(linereader[lineno].split(",")[1][0:4]) >= iniyear) and (linereader[lineno].split(",")[1][4:8] != "0229"):
            currpre = np.append(currpre, float(linereader[lineno].split(",")[2]))
            if (int(linereader[lineno].split(",")[3]) == 9) or (int(linereader[lineno].split(",")[3]) == 1):
                missingcount = missingcount + 1.
                currpre[-1] = np.NaN
        lineno = lineno + 1

    # record the data if it cover the whole temperal range and report missing data rate
    if (len(currpre) == 365*(endyear-iniyear+1)):
        print("Current missing ratio is "+str(missingcount/365/(endyear-iniyear+1)))
        outlog.write("Current missing ratio is "+str(missingcount/365/(endyear-iniyear+1))+"\n")
        saobspre = np.concatenate((saobspre, currpre))
    # print(currpre[0:365])

# convert to mm and regrid saobs data
# print(np.nansum(saobspre))
saobspre = saobspre/10.
# print(np.nansum(saobspre))
saobspre = np.reshape(saobspre, (len(stnids), 365*(endyear-iniyear+1)))
print(np.shape(saobspre))

################################################################################################
# S2-read data from vrseasia and fv0.9x1.25 output and compute statics
################################################################################################

# open vrcesm file, fv1x1 and ne30 file
vrcesmdata = Dataset(vrcesmdir+vrcesmfname)
fv1x1data = Dataset(fv1x1dir+fv1x1fname)
ne30data = Dataset(ne30dir+ne30fname)

# open era_interim file
eradata = Dataset(eradir+erafname)

# find era time series index
time_var = eradata.variables['time']
dtime = netCDF4.num2date(time_var[:], time_var.units)
dtime = pd.to_datetime(dtime)
select_dtime = (dtime >= inidate) & (dtime <= enddate) & (~((dtime.month == 2) & (dtime.day == 29)))
print(select_dtime)
print(dtime[select_dtime])

# read lat/lon grids
vrlats = vrcesmdata.variables['lat'][:]
vrlons = vrcesmdata.variables['lon'][:]

fvlats = fv1x1data.variables['lat'][:]
fvlons = fv1x1data.variables['lon'][:]

nelats = ne30data.variables['lat'][:]
nelons = ne30data.variables['lon'][:]

eralats = eradata.variables['latitude'][:]
eralons = eradata.variables['longitude'][:]
print(eralats)

print("Reading from vrcesm, fv1x1 and ne30...")
outlog.write("Reading from vrcesm, fv1x1 and ne30...\n")

# define all stations precip
sapreall = np.array([], dtype=float)
vrpreall = np.array([], dtype=float)
fvpreall = np.array([], dtype=float)
nepreall = np.array([], dtype=float)

erapreall = np.array([], dtype=float)

# read data from cesm and compare with SA-OBS for each station
for idx in range(len(stnids)):
    print("Currently dealing with the station "+str(idx+1)+"/"+str(len(stnids))+": " +
          stnnames[idx]+" in "+countrynames[idx]+" at "+str(stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6))
    outlog.write("Currently dealing with the station "+str(idx+1)+"/"+str(len(stnids))+" : " +
                 stnnames[idx]+" in "+countrynames[idx]+" at "+str(stnlats[idx])+"/"+str(stnlons[idx])+" with an station id "+str(stnids[idx]).zfill(6)+"\n")

    # get the station location
    stnlat = stnlats[idx]
    stnlon = stnlons[idx]

    # find the station in vrcesm, fv1x1 and ne30 grids
    vrlat_idx = np.abs(vrlats - stnlat).argmin()
    vrlon_idx = np.abs(vrlons - stnlon).argmin()
    fvlat_idx = np.abs(fvlats - stnlat).argmin()
    fvlon_idx = np.abs(fvlons - stnlon).argmin()
    nelat_idx = np.abs(nelats - stnlat).argmin()
    nelon_idx = np.abs(nelons - stnlon).argmin()

    eralat_idx = np.abs(eralats - stnlat).argmin()
    eralon_idx = np.abs(eralons - stnlon).argmin()

    # get the station precip in vrcesm, fv1x1 and ne30 data and convert to mm/day
    vrpre = vrcesmdata.variables['PRECT'][(iniyear-1979)*365:(endyear-1979+1)*365, vrlat_idx, vrlon_idx]
    fvpre = fv1x1data.variables['PRECT'][(iniyear-1979)*365:(endyear-1979+1)*365, fvlat_idx, fvlon_idx]
    nepre = ne30data.variables['PRECT'][(iniyear-1979)*365:(endyear-1979+1)*365, nelat_idx, nelon_idx]
    vrpre = vrpre*86400*1000
    fvpre = fvpre*86400*1000
    nepre = nepre*86400*1000

    # print(fvlats[fvlat_idx])
    # print(fvlons[fvlon_idx])

    erapre = eradata.variables['tp'][select_dtime, eralat_idx, eralon_idx]
    erapre = erapre*1000
    # print(eralats[eralat_idx])
    # print(eralons[eralon_idx])

    # mask vrcesm, fv0.9x1.25 and ne30 with SA-OBS missing value
    # print(np.nansum(vrpre))
    vrpre[np.isnan(saobspre[idx, :])] = np.NaN
    fvpre[np.isnan(saobspre[idx, :])] = np.NaN
    nepre[np.isnan(saobspre[idx, :])] = np.NaN
    erapre[np.isnan(saobspre[idx, :])] = np.NaN
    # print(vrpre[(1999-iniyear)*365:(1999-iniyear)*365+365])
    # print(saobspre[idx,(1999-iniyear)*365:(1999-iniyear)*365+365])

    print("SA-OBS precip sum is: "+str(np.nansum(saobspre[idx, :])))
    print("vrseasia precip sum is: "+str(np.nansum(vrpre)))
    print("fv0.9x1.25 precip sum is: "+str(np.nansum(fvpre)))
    print("ne30 precip sum is: "+str(np.nansum(nepre)))
    print("ERA interim precip sum is: "+str(np.nansum(erapre)))

    # set time range for each dataset
    vr_data = pd.DataFrame({
        'date': date,
        'pre': vrpre
    })
    # print(np.nansum(vr_data['pre']))
    # print(vr_data['pre'].sum())

    fv_data = pd.DataFrame({
        'date': date,
        'pre': fvpre
    })

    ne_data = pd.DataFrame({
        'date': date,
        'pre': nepre
    })

    era_data = pd.DataFrame({
        'date': date,
        'pre': erapre
    })

    sa_data = pd.DataFrame({
        'date': date,
        'pre': saobspre[idx, :]
    })

    # concatenate to all pre data array
    sapreall = np.concatenate((sapreall, sa_data['pre'][:]))
    vrpreall = np.concatenate((vrpreall, vr_data['pre'][:]))
    fvpreall = np.concatenate((fvpreall, fv_data['pre'][:]))
    nepreall = np.concatenate((nepreall, ne_data['pre'][:]))
    erapreall = np.concatenate((erapreall, era_data['pre'][:]))

    # print(np.shape(vrpreall))
    # print(np.count_nonzero(~np.isnan(vrpreall)))
    # print(np.shape(sapreall))
    # print(np.count_nonzero(~np.isnan(sapreall)))
    # print(np.shape(erapreall))
    # print(np.count_nonzero(~np.isnan(erapreall)))

################################################################################################
# S3-test the mean residual life plot for SA-OBS
################################################################################################
print("Plotting for mean residual life plot...")
outlog.write("Plotting for mean residual life plot...\n")

# plot for SA-obs
sapreall_nomiss = sapreall[~np.isnan(sapreall)]
sapremax = np.amax(sapreall_nomiss)
test_thresholds = np.arange(0, sapremax, sapremax/300)

# create array for mean excess and 95% confidence bounds
test_me = np.zeros(len(test_thresholds))
test_meup = np.zeros(len(test_thresholds))
test_mebot = np.zeros(len(test_thresholds))
test_menum = np.zeros(len(test_thresholds))

for idx, ithres in enumerate(test_thresholds):
    sapresub = sapreall_nomiss[sapreall_nomiss > ithres] - ithres
    test_me[idx] = np.mean(sapresub)
    sapresub_std = np.std(sapresub)
    test_menum[idx] = len(sapresub)
    test_meup[idx] = test_me[idx] + 1.96*(sapresub_std/np.sqrt(len(sapresub)))
    test_mebot[idx] = test_me[idx] - 1.96*(sapresub_std/np.sqrt(len(sapresub)))

# print(test_me)
# print(test_meup)

# plot for mean residual life plot
plt.clf()

print(test_thresholds[(np.abs(test_menum - 2000)).argmin()])
print(ss.percentileofscore(sapreall_nomiss, test_thresholds[(np.abs(test_menum - 2000)).argmin()]))
saprepercent = np.percentile(sapreall_nomiss, percentile)

plt.plot(test_thresholds, test_me, linestyle='solid', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_meup, linestyle='dashed', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_mebot, linestyle='dashed', linewidth=0.7, c='k')

plt.suptitle("Mean residual life plot for SA-OBS over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(saprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Mean Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"saobs_prect_"+str(percentile)+"th_mean_residual_life_plot_mainlandSEA.pdf")

plt.clf()
plt.plot(test_thresholds, test_menum, linestyle='solid', linewidth=0.7, c='k')

plt.suptitle("Number of Excess for SA-OBS over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(saprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Number of  Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"saobs_prect_"+str(percentile)+"th_excess_number_plot_mainlandSEA.pdf")

print(saprepercent)

# confirm that threshold should be 99 percentile, since obs has more than 2000 points over 99 percentile and mean excess life plot is linear

# plot for vrcesm
vrpreall_nomiss = vrpreall[~np.isnan(vrpreall)]
vrpremax = np.amax(vrpreall_nomiss)
test_thresholds = np.arange(0, vrpremax, vrpremax/300)

# create array for mean excess and 95% confidence bounds
test_me = np.zeros(len(test_thresholds))
test_meup = np.zeros(len(test_thresholds))
test_mebot = np.zeros(len(test_thresholds))
test_menum = np.zeros(len(test_thresholds))

for idx, ithres in enumerate(test_thresholds):
    vrpresub = vrpreall_nomiss[vrpreall_nomiss > ithres] - ithres
    test_me[idx] = np.mean(vrpresub)
    vrpresub_std = np.std(vrpresub)
    test_menum[idx] = len(vrpresub)
    test_meup[idx] = test_me[idx] + 1.96*(vrpresub_std/np.sqrt(len(vrpresub)))
    test_mebot[idx] = test_me[idx] - 1.96*(vrpresub_std/np.sqrt(len(vrpresub)))

# print(test_me)
# print(test_meup)

# plot for mean residual life plot
plt.clf()

vrprepercent = np.percentile(vrpreall_nomiss, percentile)

plt.plot(test_thresholds, test_me, linestyle='solid', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_meup, linestyle='dashed', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_mebot, linestyle='dashed', linewidth=0.7, c='k')

plt.suptitle("Mean residual life plot for vrcesm over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(vrprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Mean Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"vrcesm_prect_"+str(percentile)+"th_mean_residual_life_plot_mainlandSEA.pdf")

plt.clf()
plt.plot(test_thresholds, test_menum, linestyle='solid', linewidth=0.7, c='k')

plt.suptitle("Number of Excess for vrcesm over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(vrprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Number of  Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"vrcesm_prect_"+str(percentile)+"th_excess_number_plot_mainlandSEA.pdf")


# plot for fv0.9x1.25
fvpreall_nomiss = fvpreall[~np.isnan(fvpreall)]
fvpremax = np.amax(fvpreall_nomiss)
test_thresholds = np.arange(0, fvpremax, fvpremax/300)

# create array for mean excess and 95% confidence bounds
test_me = np.zeros(len(test_thresholds))
test_meup = np.zeros(len(test_thresholds))
test_mebot = np.zeros(len(test_thresholds))
test_menum = np.zeros(len(test_thresholds))

for idx, ithres in enumerate(test_thresholds):
    fvpresub = fvpreall_nomiss[fvpreall_nomiss > ithres] - ithres
    test_me[idx] = np.mean(fvpresub)
    fvpresub_std = np.std(fvpresub)
    test_menum[idx] = len(fvpresub)
    test_meup[idx] = test_me[idx] + 1.96*(fvpresub_std/np.sqrt(len(fvpresub)))
    test_mebot[idx] = test_me[idx] - 1.96*(fvpresub_std/np.sqrt(len(fvpresub)))

# print(test_me)
# print(test_meup)

# plot for mean residual life plot
plt.clf()

fvprepercent = np.percentile(fvpreall_nomiss, percentile)

plt.plot(test_thresholds, test_me, linestyle='solid', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_meup, linestyle='dashed', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_mebot, linestyle='dashed', linewidth=0.7, c='k')

plt.suptitle("Mean residual life plot for fv0.9x1.25 over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(fvprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Mean Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"fv1x1_prect_"+str(percentile)+"th_mean_residual_life_plot_mainlandSEA.pdf")

plt.clf()
plt.plot(test_thresholds, test_menum, linestyle='solid', linewidth=0.7, c='k')

plt.suptitle("Number of Excess for fv0.9x1.25 over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(fvprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Number of  Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"fv1x1_prect_"+str(percentile)+"th_excess_number_plot_mainlandSEA.pdf")


# plot for ne30
nepreall_nomiss = nepreall[~np.isnan(nepreall)]
nepremax = np.amax(nepreall_nomiss)
test_thresholds = np.arange(0, nepremax, nepremax/300)

# create array for mean excess and 95% confidence bounds
test_me = np.zeros(len(test_thresholds))
test_meup = np.zeros(len(test_thresholds))
test_mebot = np.zeros(len(test_thresholds))
test_menum = np.zeros(len(test_thresholds))

for idx, ithres in enumerate(test_thresholds):
    nepresub = nepreall_nomiss[nepreall_nomiss > ithres] - ithres
    test_me[idx] = np.mean(nepresub)
    nepresub_std = np.std(nepresub)
    test_menum[idx] = len(nepresub)
    test_meup[idx] = test_me[idx] + 1.96*(nepresub_std/np.sqrt(len(nepresub)))
    test_mebot[idx] = test_me[idx] - 1.96*(nepresub_std/np.sqrt(len(nepresub)))

# print(test_me)
# print(test_meup)

# plot for mean residual life plot
plt.clf()

neprepercent = np.percentile(nepreall_nomiss, percentile)

plt.plot(test_thresholds, test_me, linestyle='solid', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_meup, linestyle='dashed', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_mebot, linestyle='dashed', linewidth=0.7, c='k')

plt.suptitle("Mean residual life plot for ne30 over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(neprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Mean Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"ne30_prect_"+str(percentile)+"th_mean_residual_life_plot_mainlandSEA.pdf")

plt.clf()
plt.plot(test_thresholds, test_menum, linestyle='solid', linewidth=0.7, c='k')

plt.suptitle("Number of Excess for ne30 over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(neprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Number of  Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"ne30_prect_"+str(percentile)+"th_excess_number_plot_mainlandSEA.pdf")


# plot for era interim
erapreall_nomiss = erapreall[~np.isnan(erapreall)]
erapremax = np.amax(erapreall_nomiss)
test_thresholds = np.arange(0, erapremax, erapremax/300)

# create array for mean excess and 95% confidence bounds
test_me = np.zeros(len(test_thresholds))
test_meup = np.zeros(len(test_thresholds))
test_mebot = np.zeros(len(test_thresholds))
test_menum = np.zeros(len(test_thresholds))

for idx, ithres in enumerate(test_thresholds):
    erapresub = erapreall_nomiss[erapreall_nomiss > ithres] - ithres
    test_me[idx] = np.mean(erapresub)
    erapresub_std = np.std(erapresub)
    test_menum[idx] = len(erapresub)
    test_meup[idx] = test_me[idx] + 1.96*(erapresub_std/np.sqrt(len(erapresub)))
    test_mebot[idx] = test_me[idx] - 1.96*(erapresub_std/np.sqrt(len(erapresub)))

# print(test_me)
# print(test_meup)

# plot for mean residual life plot
plt.clf()

eraprepercent = np.percentile(erapreall_nomiss, percentile)

plt.plot(test_thresholds, test_me, linestyle='solid', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_meup, linestyle='dashed', linewidth=0.7, c='k')
plt.plot(test_thresholds, test_mebot, linestyle='dashed', linewidth=0.7, c='k')

plt.suptitle("Mean residual life plot for ERA interim over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(eraprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Mean Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"era_prect_"+str(percentile)+"th_mean_residual_life_plot_mainlandSEA.pdf")

plt.clf()
plt.plot(test_thresholds, test_menum, linestyle='solid', linewidth=0.7, c='k')

plt.suptitle("Number of Excess for ERA interim over mainland SEA", fontsize=15)
plt.title(str(percentile)+"th percentile: "+str(eraprepercent)+",  2000 obs needs threshold below: " +
          str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=10)
plt.ylabel("Number of  Excess")
plt.xlabel("Thresholds")
plt.savefig(outdir+"era_prect_"+str(percentile)+"th_excess_number_plot_mainlandSEA.pdf")


################################################################################################
# S4-use Generalized Pareto Distribution to fit each data set
################################################################################################

sapresub = sapreall_nomiss[sapreall_nomiss > saprepercent] - saprepercent
vrpresub = vrpreall_nomiss[vrpreall_nomiss > vrprepercent] - vrprepercent
fvpresub = fvpreall_nomiss[fvpreall_nomiss > fvprepercent] - fvprepercent
nepresub = nepreall_nomiss[nepreall_nomiss > neprepercent] - neprepercent
erapresub = erapreall_nomiss[erapreall_nomiss > eraprepercent] - eraprepercent

sa_fit = gpd.fit(sapresub)
sa_fit = gpd.fit(sapresub)
vr_fit = gpd.fit(vrpresub)
fv_fit = gpd.fit(fvpresub)
ne_fit = gpd.fit(nepresub)
era_fit = gpd.fit(erapresub)

print('scipy.stats.gpd fit: ')

print(sa_fit)
print(vr_fit)
print(fv_fit)
print(ne_fit)
print(era_fit)

sapre_ks = 0.
vrpre_ks = 0.
fvpre_ks = 0.
nepre_ks = 0.
erapre_ks = 0.

sapre_best = []
vrpre_best = []
fvpre_best = []
nepre_best = []
erapre_best = []

sapre_nbins = 0
vrpre_nbins = 0
fvpre_nbins = 0
nepre_nbins = 0
erapre_nbins = 0


count = 0
prelen = len(sapresub)
print(prelen)
for ibins in np.arange(1.*prelen/3, 1.*prelen-2, 1.):  # (1.*prelen-1.*prelen/3)/1000):
    ibins = int(ibins)

    print('No.'+str(count)+'('+str(ibins)+' bins): ')
    popt, pcov, tempscore = kstest(sapresub, ibins)
    if tempscore.pvalue > sapre_ks:
        sapre_ks = tempscore.pvalue
        sapre_best = popt
        sapre_bestcov = pcov
        sapre_nbins = ibins
    print(tempscore)

    popt, pcov, tempscore = kstest(vrpresub, ibins)
    if tempscore.pvalue > vrpre_ks:
        vrpre_ks = tempscore.pvalue
        vrpre_best = popt
        vrpre_bestcov = pcov
        vrpre_nbins = ibins
    print(tempscore)

    popt, pcov, tempscore = kstest(fvpresub, ibins)
    if tempscore.pvalue > fvpre_ks:
        fvpre_ks = tempscore.pvalue
        fvpre_best = popt
        fvpre_bestcov = pcov
        fvpre_nbins = ibins
    print(tempscore)

    popt, pcov, tempscore = kstest(nepresub, ibins)
    if tempscore.pvalue > nepre_ks:
        nepre_ks = tempscore.pvalue
        nepre_best = popt
        nepre_bestcov = pcov
        nepre_nbins = ibins
    print(tempscore)

    popt, pcov, tempscore = kstest(erapresub, ibins)
    if tempscore.pvalue > erapre_ks:
        erapre_ks = tempscore.pvalue
        erapre_best = popt
        erapre_bestcov = pcov
        erapre_nbins = ibins
    print(tempscore)

    count = count + 1
    #print('No.'+str(count)+'('+str(ibins)+' bins): '+str(tempscore.statistic)+',  '+str(tempscore.pvalue))
    # print(popt)

print('best fit using curve_fit pdf based on hist and Kolmogorov-Smirnov test:')
print(sapre_best)
print(sapre_ks)
print(vrpre_best)
print(vrpre_ks)
print(fvpre_best)
print(fvpre_ks)
print(nepre_best)
print(nepre_ks)
print(erapre_best)
print(erapre_ks)

# confirm the fitting
#pdf and hist
plt.clf()
fig = plt.figure(1)
fig.suptitle(str(iniyear)+' to '+str(endyear)+' mainland SEA ' +
             str(percentile)+'th percentile precip', y=1.05, fontsize=12)

ax = plt.subplot(2, 2, 1)
x, xbins, ypdf = pdfcheck(sapresub, nbins, sapre_best)
anderson = ss.anderson_ksamp([sapresub, gpd.rvs(sapre_best[0], 0., sapre_best[1], size=5000)])
ax.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
ax.hist(sapresub, bins=xbins, alpha=0.5, density=True, label='SA-OBS')
ax.set_title('SA-OBS, Anderson sig level='+str(anderson.significance_level), fontsize=8)
ax.legend(loc='upper right', fontsize=8)

ax = plt.subplot(2, 2, 2)
x, xbins, ypdf = pdfcheck(vrpresub, nbins, vrpre_best)
anderson = ss.anderson_ksamp([vrpresub, gpd.rvs(vrpre_best[0], 0., vrpre_best[1], size=5000)])
ax.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
ax.hist(vrpresub, bins=xbins, alpha=0.5, density=True, label='vrcesm')
ax.set_title('vrcesm, Anderson sig level='+str(anderson.significance_level), fontsize=8)
ax.legend(loc='upper right', fontsize=8)

ax = plt.subplot(2, 2, 3)
x, xbins, ypdf = pdfcheck(fvpresub, nbins, fvpre_best)
anderson = ss.anderson_ksamp([fvpresub, gpd.rvs(fvpre_best[0], 0., fvpre_best[1], size=5000)])
ax.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
ax.hist(fvpresub, bins=xbins, alpha=0.5, density=True, label='fv0.9x1.25')
ax.set_title('fv0.9x1.25, Anderson sig level='+str(anderson.significance_level), fontsize=8)
ax.legend(loc='upper right', fontsize=8)

ax = plt.subplot(2, 2, 4)
x, xbins, ypdf = pdfcheck(nepresub, nbins, nepre_best)
anderson = ss.anderson_ksamp([nepresub, gpd.rvs(nepre_best[0], 0., nepre_best[1], size=5000)])
ax.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
ax.hist(nepresub, bins=xbins, alpha=0.5, density=True, label='ne30')
ax.set_title('ne30, Anderson sig level='+str(anderson.significance_level), fontsize=8)
ax.legend(loc='upper right', fontsize=8)

fig.tight_layout()
plt.savefig(outdir+"vrcesm_prect_"+str(percentile)+"th_extremes_GPD_fit_pdf.pdf")

# qqplot
plt.clf()
fig = plt.figure(2)
ax1 = fig.add_subplot(2, 2, 1)
res1 = ss.probplot(sapresub, dist=ss.genpareto, sparams=(sapre_best[0], 0., sapre_best[1]), plot=ax1)
ax1.set_title('SA-OBS', fontsize=8)
ax1.legend(loc='upper right', fontsize=8)

ax2 = fig.add_subplot(2, 2, 2)
res2 = ss.probplot(vrpresub, dist=ss.genpareto, sparams=(vrpre_best[0], 0., vrpre_best[1]), plot=ax2)
ax2.set_title('vrcesm', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)

ax3 = fig.add_subplot(2, 2, 3)
res3 = ss.probplot(fvpresub, dist=ss.genpareto, sparams=(fvpre_best[0], 0., fvpre_best[1]), plot=ax3)
ax3.set_title('fv0.9x1.25', fontsize=8)
ax3.legend(loc='upper right', fontsize=8)

ax4 = fig.add_subplot(2, 2, 4)
res4 = ss.probplot(nepresub, dist=ss.genpareto, sparams=(nepre_best[0], 0., nepre_best[1]), plot=ax4)
ax4.set_title('ne30', fontsize=8)
ax4.legend(loc='upper right', fontsize=8)

fig.tight_layout()
plt.savefig(outdir+"vrcesm_prect_"+str(percentile)+"th_extremes_GPD_fit_qqplot.pdf")

# cdf and ks test
plt.clf()
fig = plt.figure(3)

x, xbins, ycdf = cdfcheck(sapresub, nbins, sapre_best)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, ycdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
ax1.hist(sapresub, bins=xbins, alpha=0.5, density=True, histtype='step', cumulative=True, label='SA-OBS')
ax1.set_title('SA-OBS, ks pval='+str(sapre_ks), fontsize=8)
ax1.legend(loc='lower right', fontsize=8)

x, xbins, ycdf = cdfcheck(vrpresub, nbins, vrpre_best)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, ycdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
ax2.hist(vrpresub, bins=xbins, alpha=0.5, density=True, histtype='step', cumulative=True, label='vrcesm')
ax2.set_title('vrcesm, ks pval='+str(vrpre_ks), fontsize=8)
ax2.legend(loc='lower right', fontsize=8)

x, xbins, ycdf = cdfcheck(fvpresub, nbins, fvpre_best)
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, ycdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
ax3.hist(fvpresub, bins=xbins, alpha=0.5, density=True, histtype='step', cumulative=True, label='fv0.9x1.25')
ax3.set_title('fv0.9x1.25, ks pval='+str(vrpre_ks), fontsize=8)
ax3.legend(loc='lower right', fontsize=8)

x, xbins, ycdf = cdfcheck(nepresub, nbins, nepre_best)
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x, ycdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
ax4.hist(nepresub, bins=xbins, alpha=0.5, density=True, histtype='step', cumulative=True, label='ne30')
ax4.set_title('ne30, ks pval='+str(vrpre_ks), fontsize=8)
ax4.legend(loc='lower right', fontsize=8)

fig.tight_layout()
plt.savefig(outdir+"vrcesm_prect_"+str(percentile)+"th_extremes_GPD_fit_cdf.pdf")

################################################################################################
# S5-calculate N-year return level
################################################################################################

year_returns = np.arange(5., 1005., 5.)

sa_return = np.zeros(len(year_returns))
vr_return = np.zeros(len(year_returns))
fv_return = np.zeros(len(year_returns))
ne_return = np.zeros(len(year_returns))
era_return = np.zeros(len(year_returns))

sa_return_up = np.zeros(len(year_returns))
vr_return_up = np.zeros(len(year_returns))
fv_return_up = np.zeros(len(year_returns))
ne_return_up = np.zeros(len(year_returns))
era_return_up = np.zeros(len(year_returns))

sa_return_bot = np.zeros(len(year_returns))
vr_return_bot = np.zeros(len(year_returns))
fv_return_bot = np.zeros(len(year_returns))
ne_return_bot = np.zeros(len(year_returns))
era_return_bot = np.zeros(len(year_returns))

for idx, iyear in enumerate(year_returns):
    m = iyear * 365.
    obsrate = (100.-percentile)/100.

    if (sapre_best[0] != 0):
        sa_return[idx] = saprepercent + sapre_best[1] / sapre_best[0] * ((m*obsrate) ** sapre_best[0] - 1)
        sa_sigma = (sa_return[idx] - saprepercent) * sapre_best[0] / ((m*obsrate) - 1)
        sa_return_up[idx] = sa_return[idx] + sa_sigma
        sa_return_bot[idx] = sa_return[idx] - sa_sigma

        #print((m*obsrate) ** sa_fit[0])
        # print(m*obsrate)
        # print(sa_fit[0])
        # print(sa_fit[2])
        # print(sa_sigma)
        # print(sa_return_up)
    else:
        sa_return[idx] = saprepercent + sa_fit[2] * math.log(m*obsrate)
        sa_sigma = (sa_return[idx] - saprepercent) / (math.log(m*obsrate))
        sa_return_up[idx] = sa_return[idx] + sa_sigma
        sa_return_bot[idx] = sa_return[idx] - sa_sigma

    if (vrpre_best[0] != 0):
        vr_return[idx] = vrprepercent + vrpre_best[1] / vrpre_best[0] * ((m*obsrate) ** vrpre_best[0] - 1)
        vr_sigma = (vr_return[idx] - vrprepercent) * vrpre_best[0] / ((m*obsrate) - 1)
        vr_return_up[idx] = vr_return[idx] + vr_sigma
        vr_return_bot[idx] = vr_return[idx] - vr_sigma
    else:
        vr_return[idx] = vrprepercent + vr_fit[2] * math.log(m*obsrate)
        vr_sigma = (vr_return[idx] - vrprepercent) / (math.log(m*obsrate))
        vr_return_up[idx] = vr_return[idx] + vr_sigma
        vr_return_bot[idx] = vr_return[idx] - vr_sigma

    if (fvpre_best[0] != 0):
        fv_return[idx] = fvprepercent + fvpre_best[1] / fvpre_best[0] * ((m*obsrate) ** fvpre_best[0] - 1)
        fv_sigma = (fv_return[idx] - fvprepercent) * fvpre_best[0] / ((m*obsrate) - 1)
        fv_return_up[idx] = fv_return[idx] + fv_sigma
        fv_return_bot[idx] = fv_return[idx] - fv_sigma
    else:
        fv_return[idx] = fvprepercent + fv_fit[2] * math.log(m*obsrate)
        fv_sigma = (fv_return[idx] - fvprepercent) / (math.log(m*obsrate))
        fv_return_up[idx] = fv_return[idx] + fv_sigma
        fv_return_bot[idx] = fv_return[idx] - fv_sigma

    if (nepre_best[0] != 0):
        ne_return[idx] = neprepercent + nepre_best[1] / nepre_best[0] * ((m*obsrate) ** nepre_best[0] - 1)
        ne_sigma = (ne_return[idx] - neprepercent) * nepre_best[0] / ((m*obsrate) - 1)
        ne_return_up[idx] = ne_return[idx] + ne_sigma
        ne_return_bot[idx] = ne_return[idx] - ne_sigma
    else:
        ne_return[idx] = neprepercent + ne_fit[2] * math.log(m*obsrate)
        ne_sigma = (ne_return[idx] - neprepercent) / (math.log(m*obsrate))
        ne_return_up[idx] = ne_return[idx] + ne_sigma
        ne_return_bot[idx] = ne_return[idx] - ne_sigma

    if (erapre_best[0] != 0):
        era_return[idx] = eraprepercent + erapre_best[1] / erapre_best[0] * ((m*obsrate) ** erapre_best[0] - 1)
        era_sigma = (era_return[idx] - eraprepercent) * erapre_best[0] / ((m*obsrate) - 1)
        era_return_up[idx] = era_return[idx] + era_sigma
        era_return_bot[idx] = era_return[idx] - era_sigma
    else:
        era_return[idx] = eraprepercent + era_fit[2] * math.log(m*obsrate)
        era_sigma = (era_return[idx] - eraprepercent) / (math.log(m*obsrate))
        era_return_up[idx] = era_return[idx] + era_sigma
        era_return_bot[idx] = era_return[idx] - era_sigma

plt.clf()
plt.figure(3)

plt.plot(year_returns, sa_return, c='k', label='SA-OBS')
plt.plot(year_returns, vr_return, c='r', label='vrcesm')
plt.plot(year_returns, fv_return, c='b', label='fv0.9x1.25')
plt.plot(year_returns, ne_return, c='g', label='ne30')
plt.plot(year_returns, era_return, c='m', label='ERA-interim')

plt.plot(year_returns, sa_return_up, c='k', linestyle='dashed')
plt.plot(year_returns, vr_return_up, c='r', linestyle='dashed')
plt.plot(year_returns, fv_return_up, c='b', linestyle='dashed')
plt.plot(year_returns, ne_return_up, c='g', linestyle='dashed')
plt.plot(year_returns, era_return_up, c='m', linestyle='dashed')

plt.plot(year_returns, sa_return_bot, c='k', linestyle='dashed')
plt.plot(year_returns, vr_return_bot, c='r', linestyle='dashed')
plt.plot(year_returns, fv_return_bot, c='b', linestyle='dashed')
plt.plot(year_returns, ne_return_bot, c='g', linestyle='dashed')
plt.plot(year_returns, era_return_bot, c='m', linestyle='dashed')

plt.legend(loc='upper left')

plt.title("Precip extreme return levels")

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.ylabel("Precip(mm/day)", fontsize=8)
plt.xlabel("Return years", fontsize=8)

plt.savefig(outdir+"vrcesm_prect_"+str(percentile)+"th_extremes_return_levels.pdf")

plt.clf()
plt.figure(4)

plt.plot(sa_return, sa_return, c='k', label='reference')
plt.plot(sa_return, vr_return, c='r', label='vrcesm')
plt.plot(sa_return, fv_return, c='b', label='fv0.9x1.25')
plt.plot(sa_return, ne_return, c='g', label='ne30')
plt.plot(sa_return, era_return, c='m', label='ERA-interim')
plt.legend(loc='upper left')

plt.title("Precip extreme return levels vs SA-OBS")

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.ylabel("Model return years", fontsize=8)
plt.xlabel("SA-OBS return years", fontsize=8)

plt.savefig(outdir+"vrcesm_prect_"+str(percentile)+"th_extremes_return_levels_refSAOBS.pdf")

# plot the histogram and EV fits for R10num
#fig = plt.figure(1)
#fig.suptitle('Southeast Asia PRECT summer R10num EVA(1920-2005)', fontsize=12)
#ax = plt.subplot(3,1,1)
#fit = gev.fit(np.reshape(numR10en,ennum*len(yeart)))
#print('EV fit for 1920-2005 is',fit)
#x = np.arange(0.02, 50, 0.5)
#ax.plot(x, gev.pdf(x, fit[0],fit[1],fit[2]),'r--',label='GEV Fit')
# ax.hist(np.reshape(numR10en,ennum*len(yeart)),bins=20,alpha=0.5,normed=True,label='Empirical')
#ax.set_title('Southeast Asia PRECT JJA numR10 GEV fit(1920-2005)')
#ax.legend(loc='upper right')
