# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_cesm import readcesm, readcesm_3D
import collections
from netCDF4 import Dataset

import matplotlib.cm as cm
import numpy as np
import pandas as pd

from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.1
plt.switch_backend('agg')

# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = '/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/'


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/watertag/drywet/'

# set up variable names and file name
varname = 'Moist Sources'
var_longname = "moist"
varstr = "wt"
var_unit = 'mm/day'

# define inital year and end year
iniyear = 1980
endyear = 2005
yearts = np.arange(iniyear, endyear+1)
nyears = len(yearts)

# define the contour plot region
latbounds = [10, 40]
lonbounds = [80, 170]

# latbounds = [-20, 40]
# lonbounds = [60, 150]

# define top layer
ptop = 200
plevel = 500
plevs = [200, 300, 400, 500, 700, 850, 925]

# contants
oro_water = 997
g = 9.8
r_earth = 6371000

# define Southeast region
reg_lats = [10, 20]
reg_lons = [100, 110]


# define Wet-Dry yyears
month_idx = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
monsoon_period_str = 'JAS'
if monsoon_period_str == 'JJA':
    monsoon_period = [6, 7, 8]
    monsoon_dayidx = [151, 243]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1987, 1993, 2002]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'JJAS':
    monsoon_period = [6, 7, 8, 9]
    monsoon_dayidx = [151, 273]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1987, 1993, 2002]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'MJJAS':
    monsoon_period = [5, 6, 7, 8, 9]
    monsoon_dayidx = [120, 273]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1987, 1993, 2000]
    years_dry = [1988, 1992, 1995, 1998, 1999, 2005]
if monsoon_period_str == 'JAS':
    monsoon_period = [7, 8, 9]
    monsoon_dayidx = [181, 273]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1987, 1991, 1993, 2000]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'AS':
    monsoon_period = [8, 9]
    monsoon_dayidx = [212, 273]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1987, 1991, 1996, 2000]
    years_dry = [1980, 1990, 1995, 1998, 2005]
if monsoon_period_str == 'Sep':
    monsoon_period = [9]
    monsoon_dayidx = [243, 273]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1981, 1983, 1987, 1996, 2000]
    years_dry = [1980, 1990, 1995, 2005]
if monsoon_period_str == 'Oct':
    monsoon_period = [10]
    monsoon_dayidx = [273, 304]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1981, 1986, 1989, 2001]
    years_dry = [1990, 1992, 1993, 1995, 2004]
if monsoon_period_str == 'May':
    monsoon_period = [5]
    monsoon_dayidx = [120, 151]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1981, 1982, 1985, 1990, 2000]
    years_dry = [1983, 1987, 1988, 1992, 1998]
if monsoon_period_str == 'AM':
    monsoon_period = [4, 5]
    monsoon_dayidx = [90, 151]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1981, 1985, 1996, 2000]
    years_dry = [1983, 1987, 1992, 1998, 2005]
if monsoon_period_str == 'MJ':
    monsoon_period = [5, 6]
    monsoon_dayidx = [120, 181]
    ndays = monsoon_dayidx[-1] - monsoon_dayidx[0]
    years_wet = [1981, 1985, 1997, 2001]
    years_dry = [1987, 1988, 1992, 2005]
outdir = outdir + monsoon_period_str+'/py_moist_diff_contour_wtLandMask_10N_20N_TCs/'

nyears_wet = len(years_wet)
nyears_dry = len(years_dry)

# set data frequency
frequency = 'day'

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
############################################################################
# define tagged regions
############################################################################

# define regions (10N-30N, 80E-170E band)

reg_name = 'SEAband'
reg_list = [15, 22, 27, 30]

var_longname = 'Moisture Sources'
varstr = 'wt'

# first entry for each tag is latitude 1, second entry is latitude 2, third entry is land or ocean (land=1, ocean=0), fourth entry is tag number
lats = collections.OrderedDict([
    ('Southern Indian Ocean', [-40, -30, 0, 5]),
    ('Southern Coast Indian Ocean', [-30, -10, 0, 6]),
    ('South Western Indian Ocean', [-30, -10, 0, 7]),
    ('South Central Indian Ocean', [-30, -10, 0, 8]),
    ('South East Indian Ocean', [-30, -10, 0, 9]),
    ('Eastern Indian Ocean', [-10, 10, 0, 10]),
    ('Equatorial Coastal Indian Ocean', [-10, 10, 0, 11]),
    ('Equatorial Central Indian Ocean', [-10, 10, 0, 12]),
    ('Equatorial East Indian Ocean', [-10, 10, 0, 13]),
    ('Arabian Sea', [10, 30, 0, 14]),
    ('Bay of Bengal', [10, 30, 0, 15]),
    ('Mediterranean Sea', [30, 50, 0, 16]),
    ('Persian Gulf', [10, 30, 0, 17]),
    ('Europe', [40, 70, 1, 18]),
    ('North Asia', [50, 80, 1, 19]),
    ('Indian Sub Continent', [10, 30, 1, 20]),
    ('Southeast Asia subregion', [10, 30, 1, 21]),
    ('East coast China Ocean', [10, 40, 0, 22]),
    ('East China land', [20, 50, 1, 23]),
    ('Mid Asia and Western China', [30, 50, 1, 24]),
    ('Indonesia Ocean', [-10, 10, 0, 25]),
    ('Indonesia Land', [-10, 10, 1, 26]),
    ('Southern Japanese Ocean', [10, 30, 0, 27]),
    ('Japanese Ocean', [30, 60, 0, 28]),
    ('Australia', [-40, -10, 1, 29]),
    ('Northwest Subtropical Pacific_W', [10, 30, 0, 30]),
    ('Northwest Subtropical Pacific_E', [10, 30, 0, 31]),
    ('West Equatorial Pacific_W', [-10, 10, 0, 32]),
    ('West Equatorial Pacific_E', [-10, 10, 0, 33]),
    ('North Pacific', [30, 60, 0, 34]),
    ('East Equatorial Pacific', [-10, 10, 0, 35]),
    ('Northeast Subtropical Pacific', [10, 30, 0, 36]),
    ('Southeast Subtropical Pacific', [-30, -10, 0, 37]),
    ('Southwest Subtropical Pacific', [-30, -10, 0, 38]),
    ('East Pacific', [-30, 30, 0, 39]),
    ('Southern Pacific', [-60, -30, 0, 40])])

print('total region numbers(lats):')
lats_l = list(lats.keys())
print(len(lats_l))
print('************************************')

lons = collections.OrderedDict([
    ('Southern Indian Ocean', [30, 110, 0, 5]),
    ('Southern Coast Indian Ocean', [30, 50, 0, 6]),
    ('South Western Indian Ocean', [50, 70, 0, 7]),
    ('South Central Indian Ocean', [70, 90, 0, 8]),
    ('South East Indian Ocean', [90, 110, 0, 9]),
    ('Eastern Indian Ocean', [90, 110, 0, 10]),
    ('Equatorial Coastal Indian Ocean', [30, 50, 0, 11]),
    ('Equatorial Central Indian Ocean', [50, 70, 0, 12]),
    ('Equatorial East Indian Ocean', [70, 90, 0, 13]),
    ('Arabian Sea', [50, 80, 0, 14]),
    ('Bay of Bengal', [80, 100, 0, 15]),
    ('Mediterranean Sea', [350, 40, 0, 16]),
    ('Persian Gulf', [30, 50, 0, 17]),
    ('Europe', [350, 50, 1, 18]),
    ('North Asia', [50, 130, 1, 19]),
    ('Indian Sub Continent', [70, 90, 1, 20]),
    ('Southeast Asia subregion', [90, 110, 1, 21]),
    ('East coast China Ocean', [110, 130, 0, 22]),
    ('East China land', [110, 130, 1, 23]),
    ('Mid Asia and Western China', [50, 110, 1, 24]),
    ('Indonesia Ocean', [110, 150, 0, 25]),
    ('Indonesia Land', [110, 150, 1, 26]),
    ('Southern Japanese Ocean', [130, 150, 0, 27]),
    ('Japanese Ocean', [130, 150, 0, 28]),
    ('Australia', [110, 150, 1, 29]),
    ('Northwest Subtropical Pacific_W', [150, 170, 0, 30]),
    ('Northwest Subtropical Pacific_E', [170, 190, 0, 31]),
    ('West Equatorial Pacific_W', [150, 170, 0, 32]),
    ('West Equatorial Pacific_E', [170, 190, 0, 33]),
    ('North Pacific', [150, 230, 0, 34]),
    ('East Equatorial Pacific', [190, 230, 0, 35]),
    ('Northeast Subtropical Pacific', [190, 230, 0, 36]),
    ('Southeast Subtropical Pacific', [190, 230, 0, 37]),
    ('Southwest Subtropical Pacific', [150, 190, 0, 38]),
    ('East Pacific', [230, 270, 0, 39]),
    ('Southern Pacific', [150, 270, 0, 40])])

print('total region numbers(lons):')
lons_l = list(lons.keys())
print(len(lons_l))
print('************************************')

tagnum = len(lats_l)  # all tag numbers

print('all tag regions:')
for i in np.arange(0, len(lons_l)):
    print(str(i+5)+': '+lons_l[i])
print('************************************')

tagnum = len(lats_l)  # all tag numbers
regnum = len(reg_list)  # selected source region numbers

################################################################################################
# S0-Define functions
################################################################################################


def cal_diff(var1, var2, std1, std2, n1, n2):
    res = var1-var2
    SE = np.sqrt((std1**2/n1) + (std2**2/n2))
    SE[SE == 0] = 99999
    res_sig = res/SE
    res_sig = np.abs(res_sig)

    return res, res_sig


################################################################################################
# S1-read climatological data
################################################################################################
# read vrcesm
print('Reading CESM data...')

# define file directory
filedir = "/scratch/d/dylan/harryli/gpcdata/icam5_iclm4_v12/archive/SEA_wt_1920today/atm/hist/"

# read lats lons, land fraction
filename = filedir+"SEA_wt_1920today.cam.h0.1980-01.nc"
nvarFile = Dataset(filename)
org_lats = nvarFile.variables['lat'][:]
org_lons = nvarFile.variables['lon'][:]

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(org_lats - latbounds[0]))
model_latu = np.argmin(np.abs(org_lats - latbounds[1]))
model_lonl = np.argmin(np.abs(org_lons - lonbounds[0]))
model_lonr = np.argmin(np.abs(org_lons - lonbounds[1]))

model_lats = org_lats[model_latl: model_latu+1]
model_lons = org_lons[model_lonl: model_lonr+1]
print(model_lons)

# find regional lat/lon boundaries
org_latl = np.argmin(np.abs(org_lats - reg_lats[0]))
org_latu = np.argmin(np.abs(org_lats - reg_lats[1]))
org_lonl = np.argmin(np.abs(org_lons - reg_lons[0]))
org_lonr = np.argmin(np.abs(org_lons - reg_lons[1]))

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))

lattest = np.argmin(np.abs(model_lats - 5))
lontest = np.argmin(np.abs(model_lons - 98))

print(org_lats[org_latl])
print(org_lats[org_latu])
print(org_lons[org_lonl])
print(org_lons[org_lonr])

print('region of interest boundary indices:')
print('lats: '+str(org_latl)+','+str(org_latu))
print('lons: '+str(org_lonl)+','+str(org_lonr))
print('************************************')

# define water tags
SOURCES = ['Convective rain', 'Large-scale rain', 'Convective snow', 'Large-scale sow']
sourceL = ['PRECRC_', 'PRECRL_', 'PRECSC_', 'PRECSL_']
sourceS = ['r', 'R', 's', 'S']

# create new array for moist sources timexlons for wet and dry years
wt_wet = np.zeros((ndays*len(years_wet), len(model_lons)))
wt_dry = np.zeros((ndays*len(years_dry), len(model_lons)))

# wet and dry year counter
count_wet = 0
count_dry = 0

for y in range(iniyear, endyear + 1):
    dataLoc = filedir+'SEA_wt_1920today.cam.h1.'+str(y)+'-01-01-00000.nc'
    filename = Dataset(dataLoc)
    if np.in1d(y, years_wet):
        print('Current year ('+str(y)+') is a Wet year')
        for tagind in range(len(reg_list)):
            tag = reg_list[tagind]
            tag_latl = np.argmin(np.abs(model_lats - lats[lats_l[tag-5]][0]))
            tag_latu = np.argmin(np.abs(model_lats - lats[lats_l[tag-5]][1]))
            tag_lonl = np.argmin(np.abs(model_lons - lons[lons_l[tag-5]][0]))
            tag_lonr = np.argmin(np.abs(model_lons - lons[lons_l[tag-5]][1]))
            print('tag region: '+lats_l[tag-5])
            print('tag lons: '+str(model_lats[tag_latl])+','+str(model_lats[tag_latu]))
            print('tag lons: '+str(model_lons[tag_lonl])+','+str(model_lons[tag_lonr]))
            for s in range(len(SOURCES)):
                print(sourceL[s] + 'wt' + str(tag) + sourceS[s])
                tagstep = filename.variables[sourceL[s] + 'wt' +
                                             str(tag) + sourceS[s]][monsoon_dayidx[0]:monsoon_dayidx[1], org_latl:org_latu + 1, org_lonl:org_lonr + 1]
                print(np.mean(np.mean(tagstep, axis=1), axis=1))
                tagstep = np.mean(np.mean(tagstep, axis=1), axis=1) * 86400 * 1000
                for lon_idx in range(tag_lonl, tag_lonr):
                    wt_wet[ndays*count_wet: ndays*(count_wet+1), lon_idx] = wt_wet[ndays *
                                                                                   count_wet: ndays*(count_wet+1), lon_idx] + tagstep
        count_wet = count_wet + 1

    if np.in1d(y, years_dry):
        print('Current year ('+str(y)+') is a Dry year')
        for tagind in range(len(reg_list)):
            tag = reg_list[tagind]
            tag_latl = np.argmin(np.abs(model_lats - lats[lats_l[tag-5]][0]))
            tag_latu = np.argmin(np.abs(model_lats - lats[lats_l[tag-5]][1]))
            tag_lonl = np.argmin(np.abs(model_lons - lons[lons_l[tag-5]][0]))
            tag_lonr = np.argmin(np.abs(model_lons - lons[lons_l[tag-5]][1]))
            print('tag region: '+lats_l[tag-5])
            print('tag lons: '+str(model_lats[tag_latl])+','+str(model_lats[tag_latu]))
            print('tag lons: '+str(model_lons[tag_lonl])+','+str(model_lons[tag_lonr]))
            for s in range(len(SOURCES)):
                tagstep = filename.variables[sourceL[s] + 'wt' +
                                             str(tag) + sourceS[s]][monsoon_dayidx[0]:monsoon_dayidx[1], org_latl:org_latu + 1, org_lonl:org_lonr + 1]
                tagstep = np.mean(np.mean(tagstep, axis=1), axis=1) * 86400 * 1000
                for lon_idx in range(tag_lonl, tag_lonr):
                    wt_dry[ndays*count_dry: ndays*(count_dry+1), lon_idx] = wt_dry[ndays *
                                                                                   count_dry: ndays*(count_dry+1), lon_idx] + tagstep
        count_dry = count_dry + 1


print('finished reading...')


################################################################################################

# calculate wet/dry year mean
wt_wet_mean = np.zeros((ndays, len(model_lons)))
wt_dry_mean = np.zeros((ndays, len(model_lons)))
wt_wet_std = np.zeros((ndays, len(model_lons)))
wt_dry_std = np.zeros((ndays, len(model_lons)))

for idx in range(ndays):
    wt_wet_mean[idx, :] = np.mean(wt_wet[idx:: ndays, :], axis=0)
    wt_wet_std[idx, :] = np.std(wt_wet[idx:: ndays, :], axis=0)
    wt_dry_mean[idx, :] = np.mean(wt_dry[idx:: ndays, :], axis=0)
    wt_dry_std[idx, :] = np.std(wt_dry[idx:: ndays, :], axis=0)

print(wt_wet_mean[:, 1])

res_diff, res_sig = cal_diff(wt_wet_mean, wt_dry_mean, wt_wet_std, wt_dry_std, len(years_wet), len(years_dry))


days = range(ndays)
model_sig = np.zeros((ndays, len(model_lons)))
model_sig[:, :] = 0.5

plot_data = [wt_wet_mean, wt_dry_mean, res_diff]
plot_days = [days, days, days]
plot_lons = [model_lons, model_lons, model_lons]
plot_test = [model_sig, model_sig, res_sig]
colors = [cm.Blues, cm.Greens, cm.BrBG]


legends = ['a) Wet', 'b) Dry', 'c) Wet-Dry']

title = ' CESM differences in moisture sources between Wet and Dry years'
fname = 'icam5_wt_SEA_daily_contour_diff_drywet_'+monsoon_period_str

print('plotting for season/month: '+monsoon_period_str+'...')
# plot for wet-dry contours
plt.clf()
fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ss in range(3):
    axes[ss].set_title(legends[ss], fontsize=7, pad=0.3)

    x, y = np.meshgrid(model_lons, days)
    if ss < 2:
        clevs = np.arange(0., 2., 0.2)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=colors[ss], alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_ylabel('Days', fontsize=7, labelpad=1.2)
        axes[ss].set_xticks([])
        axes[ss].yaxis.set_tick_params(labelsize=6)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-0.8, 1., 0.2)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=colors[ss], alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_xlabel('Longitude', fontsize=7, labelpad=1.2)
        axes[ss].set_ylabel('Days', fontsize=7, labelpad=1.2)
        axes[ss].xaxis.set_tick_params(labelsize=6)
        axes[ss].yaxis.set_tick_params(labelsize=6)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=4)
    cbar.ax.set_title('mm/day', fontsize=5, loc='left')

plt.savefig(outdir+fname+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=7, y=0.95)
plt.savefig(outdir+fname+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)

# plot for wet-dry contours with sig
plt.clf()
fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ss in range(3):
    axes[ss].set_title(legends[ss], fontsize=7, pad=0.3)

    x, y = np.meshgrid(model_lons, days)
    if ss < 2:
        clevs = np.arange(0., 2., 0.2)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=colors[ss], alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_ylabel('Days', fontsize=7, labelpad=1.2)
        axes[ss].set_xticks([])
        axes[ss].yaxis.set_tick_params(labelsize=6)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-0.8, 1., 0.2)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=colors[ss], alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_xlabel('Longitude', fontsize=7, labelpad=1.2)
        axes[ss].set_ylabel('Days', fontsize=7, labelpad=1.2)
        axes[ss].xaxis.set_tick_params(labelsize=6)
        axes[ss].yaxis.set_tick_params(labelsize=6)
        temptest = plot_test[ss]
        levels = [0., 2.01, temptest.max()]
        axes[ss].contourf(x, y, plot_test[ss], levels=levels, colors='none', hatches=['', '//////'], alpha=0)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=4)
    cbar.ax.set_title('mm/day', fontsize=5, loc='left')

plt.savefig(outdir+fname+'_wtsig.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=7, y=0.95)
plt.savefig(outdir+fname+'_wtsig.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)


# plot for wet and dry contours
plt.clf()
fig, axes = plt.subplots(2, 1)
axes = axes.flatten()

for ss in range(2):
    axes[ss].set_title(legends[ss], fontsize=7, pad=0.3)

    x, y = np.meshgrid(model_lons, days)
    if ss < 1:
        clevs = np.arange(0., 2., 0.2)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=colors[ss], alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_ylabel('Days', fontsize=8, labelpad=1.2)
        axes[ss].set_xticks([])
        axes[ss].yaxis.set_tick_params(labelsize=7)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(0., 2., 0.2)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=colors[ss], alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_xlabel('Longitude', fontsize=8, labelpad=1.2)
        axes[ss].set_ylabel('Days', fontsize=8, labelpad=1.2)
        axes[ss].xaxis.set_tick_params(labelsize=7)
        axes[ss].yaxis.set_tick_params(labelsize=7)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.set_title('mm/day', fontsize=6, loc='left')

plt.savefig(outdir+fname+'_nodiff.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=8, y=0.95)
plt.savefig(outdir+fname+'_nodiff.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
