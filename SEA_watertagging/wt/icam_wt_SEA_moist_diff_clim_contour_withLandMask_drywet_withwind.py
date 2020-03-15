#########################################################################
# This script is used to calculate moisture source contribution
# fraction from different source regions to Southeast Asia (SEA)
# in different months and seasons
#
# by Harry Li (2018.06)
# refers to Ellen Dyer's code
#########################################################################
# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_cesm import readcesm, readcesm_3D

from matplotlib.patches import Polygon
from netCDF4 import MFDataset
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import collections
import os
import glob
import string
from math import *
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.1
matplotlib.use("Agg")


def drawMapRect(lats, lons, m, color):
    x, y = m(lons, lats)
    xy = list(zip(x, y))
    # print(xy)
    poly = Polygon(xy, hatch='x', alpha=0.5, fill=False)
    plt.gca().add_patch(poly)


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
    ('Arabian Sea', [50, 80, 0, 14]), ('Bay of Bengal', [80, 100, 0, 15]),
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

# define file directory
filedir = "/scratch/d/dylan/harryli/gpcdata/icam5_iclm4_v12/archive/SEA_wt_1920today/atm/hist/"
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/watertag/'

# set up data directories and filenames
case = "SEA_wt_1920today"
expdir1 = '/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/'

##############################################################################
# read moisture sources
##############################################################################

# read lats lons, land fraction
filename = filedir+"SEA_wt_1920today.cam.h0.1980-01.nc"
nvarFile = Dataset(filename)
LON = nvarFile.variables['lon'][:]
lat = nvarFile.variables['lat'][:]
LNDFRC = nvarFile.variables['LANDFRAC'][0, :, :]
lndfrc, lon = shiftgrid(180., LNDFRC[:, :], LON, start=False)

# define month array
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '01', '02']
mons = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]) - 1
numMon = len(mons)

# Region of Interest
las = [10, 10, 20, 20]
los = [100, 110, 110, 100]

# find region boundary indice
la_drc_1 = np.argmin(np.abs(lat - las[0]))
la_drc_2 = np.argmin(np.abs(lat - las[2]))
lo_drc_1 = np.argmin(np.abs(LON - los[0]))
lo_drc_2 = np.argmin(np.abs(LON - los[2]))

print('region of interest boundary indices:')
print('lats: '+str(la_drc_1)+','+str(la_drc_2))
print('lons: '+str(lo_drc_1)+','+str(lo_drc_2))
print('************************************')

# define water tags
SOURCES = ['Convective rain', 'Large-scale rain', 'Convective snow', 'Large-scale sow']
sourceL = ['PRECRC_', 'PRECRL_', 'PRECSC_', 'PRECSL_']
sourceS = ['r', 'R', 's', 'S']

# define years
year1 = 1980
year2 = 2005

# set plot months and seasons
monplt = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
monnam = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monind = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
seaplt = ['DJF', 'MAM', 'JJA', 'SON']
seaind = np.array([[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])

# define Wet-Dry yyears
monsoon_period_str = 'JAS'
if monsoon_period_str == 'JJA':
    monsoon_period = [6, 7, 8]
    years_wet = [1987, 1993, 2002]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'JJAS':
    monsoon_period = [6, 7, 8, 9]
    years_wet = [1987, 1993, 2002]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'MJJAS':
    monsoon_period = [5, 6, 7, 8, 9]
    years_wet = [1987, 1993, 2000]
    years_dry = [1988, 1992, 1995, 1998, 1999, 2005]
if monsoon_period_str == 'JAS':
    monsoon_period = [7, 8, 9]
    years_wet = [1987, 1991, 1993, 2000]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'AS':
    monsoon_period = [8, 9]
    years_wet = [1987, 1991, 1996, 2000]
    years_dry = [1980, 1990, 1995, 1998, 2005]
if monsoon_period_str == 'Sep':
    monsoon_period = [9]
    years_wet = [1981, 1983, 1987, 1996, 2000]
    years_dry = [1980, 1990, 1995, 2005]
if monsoon_period_str == 'Oct':
    monsoon_period = [10]
    years_wet = [1981, 1986, 1989, 2001]
    years_dry = [1990, 1992, 1993, 1995, 2004]
if monsoon_period_str == 'May':
    monsoon_period = [5]
    years_wet = [1981, 1982, 1985, 1990, 2000]
    years_dry = [1983, 1987, 1988, 1992, 1998]
if monsoon_period_str == 'AM':
    monsoon_period = [4, 5]
    years_wet = [1981, 1985, 1996, 2000]
    years_dry = [1983, 1987, 1992, 1998, 2005]
if monsoon_period_str == 'MJ':
    monsoon_period = [5, 6]
    years_wet = [1981, 1985, 1997, 2001]
    years_dry = [1987, 1988, 1992, 2005]
outdir = outdir + 'drywet/'+monsoon_period_str+'/py_moist_diff_contour_wtLandMask_10N_20N/'

nyears_wet = len(years_wet)
nyears_dry = len(years_dry)

localPrecip_wet = np.zeros((tagnum, len(monsoon_period)))
globalPrecip_wet = np.zeros(len(monsoon_period))
localPrecip_dry = np.zeros((tagnum, len(monsoon_period)))
globalPrecip_dry = np.zeros(len(monsoon_period))
print('Loading data...')

for m in range(len(monsoon_period)):
    mm = monsoon_period[m]-1
    for y in range(year1, year2 + 1):
        dataLoc = filedir+'SEA_wt_1920today.cam.h0.'+str(y)+'-'+months[mons[mm]]+'.nc'
        filename = Dataset(dataLoc)
        if np.in1d(y, years_wet):
            print('Current year ('+str(y)+') is a Wet year')
            h20step = filename.variables['PRECT'][0, la_drc_1:la_drc_2 + 1, lo_drc_1:lo_drc_2 + 1]
            globalPrecip_wet[m] = globalPrecip_wet[m] + np.mean(h20step) * 86400 * 1000
            for s in range(len(SOURCES)):
                for tag in range(5, 5+tagnum):
                    tagind = tag - 5
                    tagstep = filename.variables[sourceL[s] + 'wt' +
                                                 str(tag) + sourceS[s]][0, la_drc_1:la_drc_2 + 1, lo_drc_1:lo_drc_2 + 1]
                    localPrecip_wet[tagind, m] = localPrecip_wet[tagind, m] + np.mean(tagstep) * 86400 * 1000

        if np.in1d(y, years_dry):
            print('Current year ('+str(y)+') is a Dry year')
            h20step = filename.variables['PRECT'][0, la_drc_1:la_drc_2 + 1, lo_drc_1:lo_drc_2 + 1]
            globalPrecip_dry[m] = globalPrecip_dry[m] + np.mean(h20step) * 86400 * 1000
            for s in range(len(SOURCES)):
                for tag in range(5, 5+tagnum):
                    tagind = tag - 5
                    tagstep = filename.variables[sourceL[s] + 'wt' +
                                                 str(tag) + sourceS[s]][0, la_drc_1:la_drc_2 + 1, lo_drc_1:lo_drc_2 + 1]
                    localPrecip_dry[tagind, m] = localPrecip_dry[tagind, m] + np.mean(tagstep) * 86400 * 1000

    print('Current month is: '+months[mm])

print('************************************')

globalPrecip_wet = np.sum(globalPrecip_wet)
localPrecip_wet = np.sum(localPrecip_wet, axis=1)

globalPrecip_wet = (globalPrecip_wet)/(nyears_wet)/len(monsoon_period)
localPrecip_wet = (localPrecip_wet)/(nyears_wet)/len(monsoon_period)

globalPrecip_dry = np.sum(globalPrecip_dry)
localPrecip_dry = np.sum(localPrecip_dry, axis=1)

globalPrecip_dry = (globalPrecip_dry)/(nyears_dry)/len(monsoon_period)
localPrecip_dry = (localPrecip_dry)/(nyears_dry)/len(monsoon_period)


##############################################################################
# read winds
##############################################################################
print('Reading wind data...')

# define inital year and end year
iniyear = 1980
endyear = 2005

# define top layer
ptop = 200
plevel = 850
plevs = [200, 300, 400, 500, 700, 850, 925]

# define the contour plot region
latbounds = [-50, 70]
lonbounds = [30, 270]

# set data frequency
frequency = 'mon'

# month series
monthts = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# read PS
varname = 'PS'

resolution = 'fv19'
varfname = 'PS'
model_ps, model_time,  model_lats, model_lons = readcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)


model_ps = model_ps/100

varname = 'U'
varfname = 'U'
model_u, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

varname = 'V'
varfname = 'V'
model_v, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

model_lev = np.argmin(np.abs(model_levs - plevel))

for idx in range(len(model_levs)):
    temp_lev = model_levs[idx]
    temp = model_u[:, idx, :, :]
    temp[model_ps < temp_lev] = np.nan
    model_u[:, idx, :, :] = temp
    temp = model_v[:, idx, :, :]
    temp[model_ps < temp_lev] = np.nan
    model_v[:, idx, :, :] = temp

##############################################################################
# plot for each month
##############################################################################

legends = ['Wet years', 'Dry years', 'Differences between wet and dry years']
colors = [cm.Blues, cm.Greens, cm.BrBG, cm.Purples, cm.Reds]

print('Start to analyze for '+monsoon_period_str+'...')

select_el = np.in1d(model_time.month, monsoon_period) & np.in1d(model_time.year, years_wet)
select_la = np.in1d(model_time.month, monsoon_period) & np.in1d(model_time.year, years_dry)

print(model_time[select_el])

model_u_el = np.nanmean(model_u[select_el, model_lev, :, :], axis=0)
model_v_el = np.nanmean(model_v[select_el, model_lev, :, :], axis=0)

model_u_la = np.nanmean(model_u[select_la, model_lev, :, :], axis=0)
model_v_la = np.nanmean(model_v[select_la, model_lev, :, :], axis=0)

model_u_diff = model_u_el - model_u_la
model_v_diff = model_v_el - model_v_la


globalPrecip_awet = globalPrecip_wet
localPrecip_awet = np.zeros((tagnum))
globalPrecip_adry = globalPrecip_dry
localPrecip_adry = np.zeros((tagnum))

for t in range(0, tagnum):
    localPrecip_awet[t] = localPrecip_wet[t]
    localPrecip_adry[t] = localPrecip_dry[t]

localratio_wet = np.zeros((tagnum))
localratio_dry = np.zeros((tagnum))
print('local moisture (with real tag name)')
for t in range(0, tagnum):
    localratio_wet[t] = localPrecip_wet[t]
    print(str(t+5)+': '+str(localratio_wet[t]))
for t in range(0, tagnum):
    localratio_dry[t] = localPrecip_dry[t]
    print(str(t+5)+': '+str(localratio_dry[t]))

print('Wet years:')
print('global precip: '+str(globalPrecip_wet))
print('total local precip: '+str(np.sum(localPrecip_wet)))
print('SUM = '+str(np.sum(localratio_wet[:])))

print('Dry years:')
print('global precip: '+str(globalPrecip_dry))
print('total local precip: '+str(np.sum(localPrecip_dry)))
print('SUM = '+str(np.sum(localratio_dry[:])))

print('Plotting for '+monsoon_period_str+'...')
print('************************************')

# create global map fpr plotting
POLY_el = np.zeros((len(lat), len(LON)))
POLY_la = np.zeros((len(lat), len(LON)))
for count in range(0, tagnum):
    wtag_num = count + 5
    la1 = np.argmin(np.abs(lat - lats[lats_l[count]][0]))
    la2 = np.argmin(np.abs(lat - lats[lats_l[count]][1]))
    lo1 = np.argmin(np.abs(LON - lons[lons_l[count]][0]))
    lo2 = np.argmin(np.abs(LON - lons[lons_l[count]][1]))

    for la in range(la1, la2):
        if (lo2 > lo1):
            for lo in range(lo1, lo2):
                if np.abs(LNDFRC[la, lo] - lats[lats_l[count]][2]) < 0.5:
                    POLY_el[la, lo] = localratio_wet[count]
                    POLY_la[la, lo] = localratio_dry[count]
        else:
            for lo in range(lo1, len(LON)):
                if np.abs(LNDFRC[la, lo] - lats[lats_l[count]][2]) < 0.5:
                    POLY_el[la, lo] = localratio_wet[count]
                    POLY_la[la, lo] = localratio_dry[count]
            for lo in range(0, lo2):
                if np.abs(LNDFRC[la, lo] - lats[lats_l[count]][2]) < 0.5:
                    POLY_el[la, lo] = localratio_wet[count]
                    POLY_la[la, lo] = localratio_dry[count]

POLY_diff = POLY_el - POLY_la
plot_data = [POLY_el, POLY_la, POLY_diff]
plot_u = [model_u_el, model_u_la, model_u_diff]
plot_v = [model_v_el, model_v_la, model_v_diff]

# poly_el, lon = shiftgrid(180., POLY_el[:, :], LON, start=False)
# poly_la, lon = shiftgrid(180., POLY_la[:, :], LON, start=False)

plt.clf()
fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ss in range(3):
    axes[ss].set_title(legends[ss], fontsize=5, pad=-0.3)
    m = Basemap(projection='cyl', llcrnrlat=-50, urcrnrlat=70, llcrnrlon=30, urcrnrlon=270, ax=axes[ss])
    parallels = np.arange(90, -90, -20.0)
    meridians = np.arange(30, 271, 40.0)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=3, linewidth=0.1)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=3, linewidth=0.1)
    m.drawcoastlines()

    x, y = np.meshgrid(LON[:], lat[:])
    xx, yy = np.meshgrid(model_lons, model_lats)
    windu = plot_u[ss]
    windv = plot_v[ss]

    if ss < 2:
        clevs = np.arange(0, 2.2, 0.2)
        csm = m.pcolormesh(x, y, plot_data[ss], cmap=colors[ss], vmax=clevs[-1], vmin=clevs[0], ax=axes[ss])
        cq = m.quiver(xx[::2, ::2], yy[::2, ::2], windu[::2, ::2], windv[::2, ::2], scale=2., scale_units='xy', ax=axes[ss])
        qk = axes[ss].quiverkey(cq, 0.9, 1.05, 20, '20 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 4})
        cbar = fig.colorbar(csm, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[::2]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=3)
        cbar.ax.set_title('mm/day', fontsize=4)
    else:
        clevs = np.arange(-0.8, 0.9, 0.1)
        csm = m.pcolormesh(x, y, plot_data[ss], cmap=colors[ss], vmax=clevs[-1], vmin=clevs[0], ax=axes[ss])
        cq = m.quiver(xx[::2, ::2], yy[::2, ::2], windu[::2, ::2], windv[::2, ::2], scale=1., scale_units='xy', ax=axes[ss])
        qk = axes[ss].quiverkey(cq, 0.9, 1.05, 5, '5 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 4})
        cbar = fig.colorbar(csm, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs[::2]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=3)
        cbar.ax.set_title('mm/day', fontsize=4)
# drawMapRect(las, los, m, 'white')
# plt.show()
plt.savefig(outdir+'icam_SEA_'+str(year1)+'_'+str(year2)+'_moist_contrib_clim_contour_withLandMask_drywet_wtwind_' + monsoon_period_str+'.png', bbox_inches='tight', dpi=600)
plt.savefig(outdir+'icam_SEA_'+str(year1)+'_'+str(year2)+'_moist_contrib_clim_contour_withLandMask_drywet_wtwind_' + monsoon_period_str+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# axes[ss].set_title(legends[ss], fontsize=5, pad=-0.3)
m = Basemap(projection='cyl', llcrnrlat=-50, urcrnrlat=70, llcrnrlon=30, urcrnrlon=270, ax=ax)
parallels = np.arange(90, -90, -20.0)
meridians = np.arange(30, 271, 40.0)
m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6, linewidth=0.1)
m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6, linewidth=0.1)
m.drawcoastlines()

x, y = np.meshgrid(LON[:], lat[:])
xx, yy = np.meshgrid(model_lons, model_lats)
windu = plot_u[2]
windv = plot_v[2]


clevs = np.arange(-0.8, 0.9, 0.1)
csm = m.pcolormesh(x, y, plot_data[2], cmap=colors[2], vmax=clevs[-1], vmin=clevs[0], ax=ax)
cq = m.quiver(xx[::2, ::2], yy[::2, ::2], windu[::2, ::2], windv[::2, ::2], scale=1., scale_units='xy', ax=ax)
qk = ax.quiverkey(cq, 0.9, 1.02, 5, '5 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 7})

fig.subplots_adjust(top=0.85, bottom=0.18, wspace=0.2, hspace=0.2)
cbar_ax = fig.add_axes([0.18, 0.2, 0.7, 0.02])
cbar = fig.colorbar(csm, cax=cbar_ax, orientation='horizontal')
ticks = clevs[::2]
ticks = np.round(ticks, 2)
ticklabels = [str(itick) for itick in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)
cbar.ax.tick_params(labelsize=7)
cbar.set_label('mm/day', fontsize=7, labelpad=0.7)
# drawMapRect(las, los, m, 'white')
# plt.show()

plt.suptitle('', fontsize=7, y=0.95)
plt.savefig(outdir+'icam_SEA_'+str(year1)+'_'+str(year2)+'_moist_contrib_clim_contour_withLandMask_drywet_wtwind_' + monsoon_period_str+'_diff.png', bbox_inches='tight', dpi=600)
plt.savefig(outdir+'icam_SEA_'+str(year1)+'_'+str(year2)+'_moist_contrib_clim_contour_withLandMask_drywet_wtwind_' + monsoon_period_str+'_diff.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)

print('************************************')
