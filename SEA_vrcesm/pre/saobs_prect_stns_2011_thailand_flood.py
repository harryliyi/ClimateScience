# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-plot basic analysis
# S3-calculate and plot extreme
#
# Written by Harry Li

#import libraries
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
from scipy.stats import gamma as gamma
from scipy.stats import lognorm as lognorm
import scipy.stats as ss
from mod_dataread_obs_pre import readobs_pre_day, read_SAOBS_pre
from mod_stats_clim import mon2clim
from mod_stats_clustering import kmeans_cluster
from mod_plt_bars import plot_bars
from mod_plt_lines import plot_lines
from mod_plt_findstns import data_findstns
from mod_dataread_vrcesm import readvrcesm
from mod_dataread_cordex_sea import readcordex
import datetime as datetime
import math as math
import pandas as pd
import matplotlib.cm as cm
import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')


############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_obs/stations/SAOBS_countries/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# time bounds
iniyear = 1955
endyear = 2015
yearts = np.arange(iniyear, endyear+1)
# yearts    = np.delete(yearts,9,None)
print(yearts)
year_label = [str(iyear) for iyear in yearts]
year_ticks = np.arange(182, (endyear-iniyear+1)*365, 365)

# define regions
latbounds = [-15, 25]
lonbounds = [90, 145]

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]

# set data frequency
frequency = 'day'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# set up ndays roling sum
ndays = 10

# set up percentile to calculate the extremes
percentile = 97

# plot parameters
nbins = 30

############################################################################
# read data
############################################################################

# read Observations

print('Reading SA-OBS data...')

# read SA-OBS
version = 'countries'
countries = ['Thailand']
dataset1, obs_var1, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_SAOBS_pre(
    version, iniyear, endyear, countries, missing_ratio=10)

print(dataset1.shape[0])

dataset1 = dataset1.rolling(ndays).sum()
print(dataset1.resample('A').max())
temp = dataset1.max(axis=1)
maxts = temp.resample('A').max()
print(maxts)
maxevent = maxts['2011-12-31']

print(maxevent)

# plot precip ts for station
# print(dataset1['NAKHON SI THAMMARAT'].resample('D').sum())
stndata = dataset1['NAKHON SI THAMMARAT'].resample('D').sum().values
# print(stndata)
plt.clf()
fig = plt.figure(1)
title = str(iniyear)+' to '+str(endyear) + \
    ' NAKHON SI THAMMARAT '+str(ndays)+'-day precipitation'
fname = 'saobs_prect_Thailand_stn_002879_pdf.pdf'
plt.plot(np.arange(len(stndata)), stndata, linestyle='solid', linewidth=.5, color='k')
plt.xlabel('Year', fontsize=8)
plt.ylabel(str(ndays)+'-day precip', fontsize=8)
plt.xticks(year_ticks[::int(len(year_ticks)/8.)], year_label[::int(len(year_label)/8.)], fontsize=6)
plt.yticks(fontsize=6)
plt.title(title, fontsize=9)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)


rs = dataset1['NAKHON SI THAMMARAT'].values.flatten()
# rs = dataset1.values.flatten()
rs = rs[~np.isnan(rs)]
nonzeros = np.count_nonzero(rs)
rslen = len(rs)
rs = rs[rs > 0]
print(len(rs))

# plot pdf for var and use Gamma function to fit
rsmin = 0.
rsmax = np.amax(rs)*3/4.

x = np.arange(rsmin+(rsmax-rsmin)/nbins/2., rsmax, (rsmax-rsmin)/nbins)
xbins = np.arange(rsmin, rsmax + (rsmax-rsmin)/nbins/2., (rsmax-rsmin)/nbins)
hist = np.histogram(rs, bins=xbins, density=True)

gamma_fit = gamma.fit(rs, 0.2, loc=0, scale=85)
gamma_fit = gamma.fit(rs, 0.2, loc=0, scale=85)
print('Gamma fitting results: ')
print(gamma_fit)

plt.clf()
fig = plt.figure(1)
title = str(iniyear)+' to '+str(endyear) + \
    ' Thailand '+str(ndays)+'-day precipitation histogram'
fname = 'saobs_prect_Thailand_Gamma_fit_pdf.pdf'
ypdf = gamma.pdf(x, gamma_fit[0], gamma_fit[1], gamma_fit[2])
print(ypdf)
print(hist)
plt.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='Gamma Fit')
plt.hist(rs, bins=xbins, alpha=0.5, density=True, label='SA-OBS')
plt.legend(loc='upper right', fontsize=8)
plt.xlabel(str(ndays)+'-day precipitation', fontsize=8)
plt.ylabel('Frequency', fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.title(title, fontsize=9)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)

gamma_return = 1. / \
    (1-gamma.cdf(maxevent, gamma_fit[0], gamma_fit[1], gamma_fit[2]))/(365*nonzeros/rslen)
print('Gamma fitting return year is: ')
print(gamma_return)


plt.clf()
fig = plt.figure(1)
title = str(iniyear)+' to '+str(endyear) + \
    ' Thailand '+str(ndays)+'-day precipitation Gamma fitting qq plot'
fname = 'saobs_prect_Thailand_Gamma_qqplot_pdf.pdf'
res = ss.probplot(rs, dist=ss.gamma, sparams=(gamma_fit[0], gamma_fit[1], gamma_fit[2]))
# print(res[0][0])
print(res[1])
plt.scatter(res[0][0], res[0][1], c='b', s=3)
plt.plot(res[0][0], res[0][0], linestyle='solid', linewidth=1.5, color='r')
plt.xlabel('Theoretical quantiles', fontsize=8)
plt.ylabel('Emperical', fontsize=8)
# plt.xticks(,fontsize=6)
# plt.yticks(,fontsize=6)
plt.suptitle(title, fontsize=9)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)


# plot for pdf for var and use Lognormal to fit
lognorm_fit = lognorm.fit(rs, 0.2, loc=0, scale=85)
lognorm_fit = lognorm.fit(rs, 0.2, loc=0, scale=85)
print('Lognormal fitting results: ')
print(lognorm_fit)

plt.clf()
fig = plt.figure(1)
title = str(iniyear)+' to '+str(endyear) + \
    ' Thailand '+str(ndays)+'-day precipitation histogram'
fname = 'saobs_prect_Thailand_Lognormal_fit_pdf.pdf'
ypdf = lognorm.pdf(x, lognorm_fit[0], lognorm_fit[1], lognorm_fit[2])
# print(ypdf)
# print(hist)
plt.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='Lognormal Fit')
plt.hist(rs, bins=xbins, alpha=0.5, density=True, label='SA-OBS')
plt.legend(loc='upper right', fontsize=8)
plt.xlabel(str(ndays)+'-day precipitation', fontsize=8)
plt.ylabel('Frequency', fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.title(title, fontsize=9)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)

lognorm_return = 1. / \
    (1-lognorm.cdf(maxevent, lognorm_fit[0], lognorm_fit[1], lognorm_fit[2]))/(365*nonzeros/rslen)
print('Lognormal fitting return year is: ')
print(lognorm_return)


plt.clf()
fig = plt.figure(1)
title = str(iniyear)+' to '+str(endyear) + \
    ' Thailand '+str(ndays)+'-day precipitation Lognormal fitting qq plot'
fname = 'saobs_prect_Thailand_lognorm_qqplot_pdf.pdf'
res = ss.probplot(rs, dist=ss.lognorm, sparams=(lognorm_fit[0], lognorm_fit[1], lognorm_fit[2]))
# print(res[0][0])
print(res[1])
plt.scatter(res[0][0], res[0][1], c='b', s=3)
plt.plot(res[0][0], res[0][0], linestyle='solid', linewidth=1.5, color='r')
plt.xlabel('Theoretical quantiles', fontsize=8)
plt.ylabel('Emperical', fontsize=8)
# plt.xticks(,fontsize=6)
# plt.yticks(,fontsize=6)
plt.suptitle(title, fontsize=9)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)


# plot pdf for var and use GPD function to fit
rspercent = np.percentile(rs, percentile)
# print(rspercent)
rs = rs[rs > rspercent] - rspercent

rsmin = 0.
rsmax = np.amax(rs)*3/4.

x = np.arange(rsmin+(rsmax-rsmin)/nbins/2., rsmax, (rsmax-rsmin)/nbins)
xbins = np.arange(rsmin, rsmax + (rsmax-rsmin)/nbins/2., (rsmax-rsmin)/nbins)

gpd_fit = gpd.fit(rs, 0.1, loc=0, scale=85)
gpd_fit = gpd.fit(rs, 0.1, loc=0, scale=85)

rs_mean = np.mean(rs)
rs_var = np.var(rs)
xi = 0.5*(1-rs_mean*rs_mean/rs_var)
sigma = rs_mean*(1-xi)

print('GPD fitting results: ')
print(gpd_fit)
print([xi, sigma])

plt.clf()
fig = plt.figure(1)
title = str(iniyear)+' to '+str(endyear) + \
    ' Thailand '+str(percentile)+'th '+str(ndays)+'-day precipitation histogram'
fname = 'saobs_prect_Thailand_GPD_fit_pdf.pdf'
ypdf = gpd.pdf(x, xi, 0, sigma)
plt.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
plt.hist(rs, bins=xbins, alpha=0.5, density=True, label='SA-OBS')
plt.legend(loc='upper right', fontsize=8)
plt.xlabel(str(percentile)+'th '+str(ndays)+'-day precipitation', fontsize=8)
plt.ylabel('Frequency', fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.title(title, fontsize=9)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)

gpd_return = 1./(1-gpd.cdf(maxevent, xi, 0, sigma))/(365*(100-percentile)/100)
print('GPD fitting return year is: ')
print(gpd_return)

plt.clf()
fig = plt.figure(1)
title = str(iniyear)+' to '+str(endyear) + \
    ' Thailand '+str(ndays)+'-day precipitation GPD fitting qq plot'
fname = 'saobs_prect_Thailand_GPD_qqplot_pdf.pdf'
res = ss.probplot(rs, dist=gpd, sparams=(gpd_fit[0], 0., gpd_fit[2]))
# print(res[0][0])
print(res[1])
plt.scatter(res[0][0], res[0][1], c='b', s=3)
plt.plot(res[0][0], res[0][0], linestyle='solid', linewidth=1.5, color='r')
plt.xlabel('Theoretical quantiles', fontsize=8)
plt.ylabel('Emperical', fontsize=8)
# plt.xticks(,fontsize=6)
# plt.yticks(,fontsize=6)
plt.suptitle(title, fontsize=9)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)


# plot pdf for var and use GEV function to fit
temp = dataset1['NAKHON SI THAMMARAT'].resample('A').max()
#temp = dataset1.resample('A').max()
rs = temp.values.flatten()
print(len(rs))
gev_fit = gev.fit(rs, 0.1, loc=100, scale=85)
gev_fit = gev.fit(rs, 0.1, loc=100, scale=85)

print('GEV fitting results: ')
print(gev_fit)

plt.clf()
fig = plt.figure(1)
title = str(iniyear)+' to '+str(endyear) + \
    ' Thailand '+str(percentile)+'th '+str(ndays)+'-day precipitation histogram'
fname = 'saobs_prect_Thailand_GEV_fit_pdf.pdf'
ypdf = gev.pdf(x, gev_fit[0], gev_fit[1], gev_fit[2])
hist = np.histogram(rs, bins=xbins, density=True)
# print(ypdf)
# print(hist)
plt.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='GEV Fit')
plt.hist(rs, bins=xbins, alpha=0.5, density=True, label='SA-OBS')
plt.legend(loc='upper right', fontsize=8)
plt.xlabel(str(percentile)+'th '+str(ndays)+'-day precipitation', fontsize=8)
plt.ylabel('Frequency', fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.title(title, fontsize=9)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)

gev_return = 1./(1-gev.cdf(maxevent, gev_fit[0], gev_fit[1], gev_fit[2]))
print('GEV fitting return year is: ')
print(gev_return)

plt.clf()
fig = plt.figure(1)
title = str(iniyear)+' to '+str(endyear) + \
    ' Thailand '+str(ndays)+'-day precipitation GEV fitting qq plot'
fname = 'saobs_prect_Thailand_GEV_qqplot_pdf.pdf'
res = ss.probplot(rs, dist=gev, sparams=(gev_fit[0], gev_fit[1], gev_fit[2]))
# print(res[0][0])
print(res[1])
plt.scatter(res[0][0], res[0][1], c='b', s=3)
plt.plot(res[0][0], res[0][0], linestyle='solid', linewidth=1.5, color='r')
plt.xlabel('Theoretical quantiles', fontsize=8)
plt.ylabel('Emperical', fontsize=8)
# plt.xticks(,fontsize=6)
# plt.yticks(,fontsize=6)
plt.suptitle(title, fontsize=9)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)
