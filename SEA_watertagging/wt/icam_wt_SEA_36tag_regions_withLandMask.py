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
    poly = Polygon(xy, edgecolor='black', fill=False, linewidth=1.)
    plt.gca().add_patch(poly)
    plt.text(x[0], y[0]+5, 'SEA', color='black', fontsize=5, weight='bold')


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
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/climatology/watertag/'

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


##############################################################################
# plot for each month
##############################################################################

# create global map fpr plotting
POLY = np.zeros((len(lat), len(LON)))
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
                    POLY[la, lo] = lons[lons_l[count]][3]
        else:
            for lo in range(lo1, len(LON)):
                if np.abs(LNDFRC[la, lo] - lats[lats_l[count]][2]) < 0.5:
                    POLY[la, lo] = lons[lons_l[count]][3]
            for lo in range(0, lo2):
                if np.abs(LNDFRC[la, lo] - lats[lats_l[count]][2]) < 0.5:
                    POLY[la, lo] = lons[lons_l[count]][3]

poly, lon = shiftgrid(180., POLY[:, :], LON, start=False)

plt.clf()
m = Basemap(projection='cyl', llcrnrlat=-70, urcrnrlat=80, llcrnrlon=10, urcrnrlon=290)
parallels = np.arange(90, -90, -20.0)
meridians = np.arange(-10, 310, 40.0)
m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=5, linewidth=0.3)
m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=5, linewidth=0.3)
m.drawcoastlines(linewidth=0.6)

x, y = np.meshgrid(LON[:], lat[:])
csm = m.pcolormesh(x, y, POLY[:, :], cmap=cm.nipy_spectral, edgecolor='face', linewidth=0, rasterized=True, vmax=40., vmin=5., alpha=0.7)
# csm = m.pcolormesh(x, y, POLY[:, :], cmap=cm.Spectral, vmax=40., vmin=1.)
# csc = m.contour(x, y, POLY[:, :], colors='black', levels=np.arange(4, 41, 1), linewidths=0.5, interpolation='none')
# cbar = m.colorbar(csm, shrink=0.5)
drawMapRect(las, los, m, 'white')
# plt.show()
plt.savefig(outdir+'icam_wt_SEA_36tag_regions_wtLandMask_pynew_10N_20N.png', bbox_inches='tight', dpi=600)
plt.suptitle('36 tagged regions', fontsize=7)
plt.savefig(outdir+'icam_wt_SEA_36tag_regions_wtLandMask_pynew_10N_20N.pdf', bbox_inches='tight', dpi=600)
