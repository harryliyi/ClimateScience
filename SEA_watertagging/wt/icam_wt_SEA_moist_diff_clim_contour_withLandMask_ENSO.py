#########################################################################
# This script is used to calculate moisture source contribution
# fraction from different source regions to Southeast Asia (SEA)
# in different months and seasons
#
# by Harry Li (2018.06)
# refers to Ellen Dyer's code
#########################################################################

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
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/watertag/new_py_moist_ENSOdiff_contour_wtLandMask_10N_20N/'

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

# define ENSO yyears
years_elweak = [1980, 1983, 1987, 1988, 1992, 1995, 1998, 2003, 2005]
years_elmod = [1983, 1987, 1988, 1992, 1998, 2003]
years_laweak = [1984, 1985, 1989, 1996, 1999, 2000, 2001]
years_lamod = [1989, 1999, 2000]

nyears_el = len(years_elweak)
nyears_la = len(years_laweak)

localPrecip_el = np.zeros((tagnum, 12))
globalPrecip_el = np.zeros(12)
localPrecip_la = np.zeros((tagnum, 12))
globalPrecip_la = np.zeros(12)
print('Loading data...')

for m in range(0, 12):
    for y in range(year1, year2 + 1):
        dataLoc = filedir+'SEA_wt_1920today.cam.h0.'+str(y)+'-'+months[mons[m]]+'.nc'
        filename = Dataset(dataLoc)
        if np.in1d(y, years_elweak):
            print('Current year ('+str(y)+') is a El Nino year')
            h20step = filename.variables['PRECT'][0, la_drc_1:la_drc_2 + 1, lo_drc_1:lo_drc_2 + 1]
            globalPrecip_el[m] = globalPrecip_el[m] + np.mean(h20step) * 86400 * 1000
            for s in range(len(SOURCES)):
                for tag in range(5, 5+tagnum):
                    tagind = tag - 5
                    tagstep = filename.variables[sourceL[s] + 'wt' +
                                                 str(tag) + sourceS[s]][0, la_drc_1:la_drc_2 + 1, lo_drc_1:lo_drc_2 + 1]
                    localPrecip_el[tagind, m] = localPrecip_el[tagind, m] + np.mean(tagstep) * 86400 * 1000

        if np.in1d(y, years_laweak):
            print('Current year ('+str(y)+') is a La Nina year')
            h20step = filename.variables['PRECT'][0, la_drc_1:la_drc_2 + 1, lo_drc_1:lo_drc_2 + 1]
            globalPrecip_la[m] = globalPrecip_la[m] + np.mean(h20step) * 86400 * 1000
            for s in range(len(SOURCES)):
                for tag in range(5, 5+tagnum):
                    tagind = tag - 5
                    tagstep = filename.variables[sourceL[s] + 'wt' +
                                                 str(tag) + sourceS[s]][0, la_drc_1:la_drc_2 + 1, lo_drc_1:lo_drc_2 + 1]
                    localPrecip_la[tagind, m] = localPrecip_la[tagind, m] + np.mean(tagstep) * 86400 * 1000

    print('Current month is: '+months[m])

print('************************************')

globalPrecip_el = (globalPrecip_el)/(nyears_el)
localPrecip_el = (localPrecip_el)/(nyears_el)

globalPrecip_la = (globalPrecip_la)/(nyears_la)
localPrecip_la = (localPrecip_la)/(nyears_la)

##############################################################################
# plot for each month
##############################################################################

legends = ['Under El Nino events', 'Under La Nina events', 'Differences in ENSO events']
colors = [cm.Blues, cm.Greens, cm.BrBG, cm.Purples, cm.Reds]

for mm in range(0, 12):
    print('Start to analyze for month: '+monnam[mm])
    globalPrecip_ael = globalPrecip_el[mm]
    localPrecip_ael = np.zeros((tagnum))
    globalPrecip_ala = globalPrecip_la[mm]
    localPrecip_ala = np.zeros((tagnum))

    for t in range(0, tagnum):
        localPrecip_ael[t] = localPrecip_el[t, mm]
        localPrecip_ala[t] = localPrecip_la[t, mm]

    localratio_el = np.zeros((tagnum))
    localratio_la = np.zeros((tagnum))
    print('local moisture (with real tag name)')
    for t in range(0, tagnum):
        localratio_el[t] = localPrecip_ael[t]
        print(str(t+5)+': '+str(localratio_el[t]))
    for t in range(0, tagnum):
        localratio_la[t] = localPrecip_ala[t]
        print(str(t+5)+': '+str(localratio_la[t]))

    print('El Nino years:')
    print('global precip: '+str(globalPrecip_ael))
    print('total local precip: '+str(np.sum(localPrecip_ael)))
    print('SUM = '+str(np.sum(localratio_el[:])))

    print('La Nina years:')
    print('global precip: '+str(globalPrecip_ala))
    print('total local precip: '+str(np.sum(localPrecip_ala)))
    print('SUM = '+str(np.sum(localratio_la[:])))

    print('Plotting for month: '+monnam[mm]+'...')
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
                        POLY_el[la, lo] = localratio_el[count]
                        POLY_la[la, lo] = localratio_la[count]
            else:
                for lo in range(lo1, len(LON)):
                    if np.abs(LNDFRC[la, lo] - lats[lats_l[count]][2]) < 0.5:
                        POLY_el[la, lo] = localratio_el[count]
                        POLY_la[la, lo] = localratio_la[count]
                for lo in range(0, lo2):
                    if np.abs(LNDFRC[la, lo] - lats[lats_l[count]][2]) < 0.5:
                        POLY_el[la, lo] = localratio_el[count]
                        POLY_la[la, lo] = localratio_la[count]

    POLY_diff = POLY_el - POLY_la
    plot_data = [POLY_el, POLY_la, POLY_diff]

    # poly_el, lon = shiftgrid(180., POLY_el[:, :], LON, start=False)
    # poly_la, lon = shiftgrid(180., POLY_la[:, :], LON, start=False)

    plt.clf()
    fig, axes = plt.subplots(3, 1)
    axes = axes.flatten()

    for ss in range(3):
        axes[ss].set_title(legends[ss], fontsize=5, pad=-0.3)
        m = Basemap(projection='cyl', llcrnrlat=-50, urcrnrlat=70, llcrnrlon=30, urcrnrlon=270, ax=axes[ss])
        parallels = np.arange(90, -90, -20.0)
        meridians = np.arange(30, 270, 40.0)
        m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=3, linewidth=0.1)
        m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=3, linewidth=0.1)
        m.drawcoastlines()

        x, y = np.meshgrid(LON[:], lat[:])
        if ss < 2:
            csm = m.pcolormesh(x, y, plot_data[ss], cmap=colors[ss], vmax=2., vmin=0., ax=axes[ss])
            cbar = fig.colorbar(csm, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
            ticks = np.arange(0, 2.2, 0.2)
            ticks = np.round(ticks, 2)
            ticklabels = [str(itick) for itick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
        else:
            csm = m.pcolormesh(x, y, plot_data[ss], cmap=colors[ss], vmax=1.2, vmin=-1.2, ax=axes[ss])
            cbar = fig.colorbar(csm, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
            ticks = np.arange(-1.2, 1.4, 0.2)
            ticks = np.round(ticks, 2)
            ticklabels = [str(itick) for itick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=3)
        cbar.ax.set_title('mm/day', fontsize=4)
    # drawMapRect(las, los, m, 'white')
    # plt.show()
    plt.savefig(outdir+'icam_SEA_'+str(year1)+'_'+str(year2)+'_moist_contrib_clim_contour_withLandMask_' +
                monplt[mm]+'.png', bbox_inches='tight', dpi=600)
    plt.savefig(outdir+'icam_SEA_'+str(year1)+'_'+str(year2)+'_moist_contrib_clim_contour_withLandMask_' +
                monplt[mm]+'.pdf', bbox_inches='tight', dpi=600)

    print('************************************')
