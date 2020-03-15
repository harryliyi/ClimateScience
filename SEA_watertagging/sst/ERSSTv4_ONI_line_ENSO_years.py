# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import netCDF4 as nc
import datetime as datetime
from mpl_toolkits.basemap import Basemap
import pandas as pd
plt.switch_backend('agg')


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/sst/'

# set up variable names and file name
varname = 'SST_cpl'
var_longname = 'Sea Surface Temperature'
varstr = 'sst'
var_unit = r'$^{\circ}C$'


# define inital year and end year
iniyear = 1979
endyear = 2006

# define the contour plot region
latbounds = [-30, 30]
lonbounds = [30, 300]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'mon'

# define pressure level
plevel = 500

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# define ENSO yyears
years_elweak = [1980, 1983, 1987, 1988, 1992, 1995, 1998, 2003, 2005]
years_elmod = [1983, 1987, 1988, 1992, 1998, 2003]
years_laweak = [1984, 1985, 1989, 1996, 1999, 2000, 2001]
years_lamod = [1989, 1999, 2000]

years_elweakpre = [iyear-1 for iyear in years_elweak]
years_laweakpre = [iyear-1 for iyear in years_laweak]


############################################################################
# ERSST v4 ONI data starting from 1979 DJF
############################################################################

oni = np.array([0.0, 0.1, 0.2, 0.3, 0.2, 0.0, 0.0, 0.2, 0.3, 0.5, 0.5, 0.6,
                0.6, 0.5, 0.3, 0.4, 0.5, 0.5, 0.3, 0.0, -0.1, 0.0, 0.1, 0.0,
                -0.3, -0.5, -0.5, -0.4, -0.3, -0.3, -0.3, -0.2, -0.2, -0.1, -0.2, -0.1,
                0.0, 0.1, 0.2, 0.5, 0.7, 0.7, 0.8, 1.1, 1.6, 2.0, 2.2, 2.2,
                2.2, 1.9, 1.5, 1.3, 1.1, 0.7, 0.3, -0.1, -0.5, -0.8, -1.0, -0.9,
                -0.6, -0.4, -0.3, -0.4, -0.5, -0.4, -0.3, -0.2, -0.2, -0.6, -0.9, -1.1,
                -1.0, -0.8, -0.8, -0.8, -0.8, -0.6, -0.5, -0.5, -0.4, -0.3, -0.3, -0.4,
                -0.5, -0.5, -0.3, -0.2, -0.1, 0.0, 0.2, 0.4, 0.7, 0.9, 1.1, 1.2,
                1.2, 1.2, 1.1, 0.9, 1.0, 1.2, 1.5, 1.7, 1.6, 1.5, 1.3, 1.1,
                0.8, 0.5, 0.1, -0.3, -0.9, -1.3, -1.3, -1.1, -1.2, -1.5, -1.8, -1.8,
                -1.7, -1.4, -1.1, -0.8, -0.6, -0.4, -0.3, -0.3, -0.2, -0.2, -0.2, -0.1,
                0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.3, 0.4, 0.4,
                0.4, 0.3, 0.2, 0.3, 0.5, 0.6, 0.7, 0.6, 0.6, 0.8, 1.2, 1.5,
                1.7, 1.6, 1.5, 1.3, 1.1, 0.7, 0.4, 0.1, -0.1, -0.2, -0.3, -0.1,
                0.1, 0.3, 0.5, 0.7, 0.7, 0.6, 0.3, 0.3, 0.2, 0.1, 0.0, 0.1,
                0.1, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.6, 0.7, 1.0, 1.1,
                1.0, 0.7, 0.5, 0.3, 0.1, 0.0, -0.2, -0.5, -0.8, -1.0, -1.0, -1.0,
                -0.9, -0.8, -0.6, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.4, -0.4, -0.5,
                -0.5, -0.4, -0.1, 0.3, 0.8, 1.2, 1.6, 1.9, 2.1, 2.3, 2.4, 2.4,
                2.2, 1.9, 1.4, 1.0, 0.5, -0.1, -0.8, -1.1, -1.3, -1.4, -1.5, -1.6,
                -1.5, -1.3, -1.1, -1.0, -1.0, -1.0, -1.1, -1.1, -1.2, -1.3, -1.5, -1.7,
                -1.7, -1.4, -1.1, -0.8, -0.7, -0.6, -0.6, -0.5, -0.5, -0.6, -0.7, -0.7,
                -0.7, -0.5, -0.4, -0.3, -0.3, -0.1, -0.1, -0.1, -0.2, -0.3, -0.3, -0.3,
                -0.1, 0.0, 0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.1,
                0.9, 0.6, 0.4, 0.0, -0.3, -0.2, 0.1, 0.2, 0.3, 0.3, 0.4, 0.4,
                0.4, 0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7,
                0.6, 0.6, 0.4, 0.4, 0.3, 0.1, -0.1, -0.1, -0.1, -0.3, -0.6, -0.8,
                -0.8, -0.7, -0.5, -0.3, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9,
                ])

print(len(oni))

ts = np.arange((endyear-iniyear+1)*12)+1

title = 'ONI'
fname = 'ERSSTv4_ONI_ENSO_years'

plt.clf()
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)

ax.plot(ts, oni, linewidth=1., color='black')
ax.fill_between(ts, 0, oni, color='lightskyblue')
ax.axhline(y=0., color='black', linewidth=1.)
ax.axhline(y=0.5, color='red', linewidth=1.)
plt.text(ts[-1]+3, 0.75, 'Weak', color='red', fontsize=7)
ax.axhline(y=1., color='red', linewidth=1.)
plt.text(ts[-1]+3, 1.25, 'Moderate', color='red', fontsize=7)
ax.axhline(y=1.5, color='red', linewidth=1.)
plt.text(ts[-1]+3, 1.75, 'Strong', color='red', fontsize=7)
ax.axhline(y=2., color='red', linewidth=1.)
plt.text(ts[-1]+3, 2.25, 'Very Strong', color='red', fontsize=7)
ax.axhline(y=-0.5, color='blue', linewidth=1.)
plt.text(ts[-1]+3, -0.75, 'Weak', color='blue', fontsize=7)
ax.axhline(y=-1., color='blue', linewidth=1.)
plt.text(ts[-1]+3, -1.25, 'Moderate', color='blue', fontsize=7)
ax.axhline(y=-1.5, color='blue', linewidth=1.)
plt.text(ts[-1]+3, -1.75, 'Strong', color='blue', fontsize=7)

plt.text(3, 2.35, 'Red = El Niño', color='Red', fontsize=7)
plt.text(3, 2.15, 'Blue = La Niña', color='Blue', fontsize=7)

ax.set_xlim(0, ts[-1])
ax.set_ylim(-2, 2.5)

xticks = ts[5::12]
xticknames = np.arange(iniyear, endyear + 1, 1)
ax.set_xticks(xticks)
ax.set_xticklabels(xticknames, fontsize=5, rotation=90)

yticks = np.arange(-2, 2.7, 0.5)
yticknames = yticks
ax.set_yticks(yticks)
ax.set_yticklabels(yticknames, fontsize=5, rotation=90)

# ax2 = ax.twinx()
# yticks = np.arange(-2, 2.7, 0.5)
# yticknames = ['Strong', 'Moderate', 'Weak', 'Weak', 'Moderate', 'Strong']
# ax2.set_yticks(yticks)
# ax2.set_yticklabels(yticknames, fontsize=5)

ax.set_xlabel('Years', fontsize=7)
ax.set_ylabel('Oceanic Niño Index', fontsize=7)

plt.savefig(outdir+fname+'_wtsig.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=7, y=0.9)
plt.savefig(outdir+fname+'_wtsig.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
