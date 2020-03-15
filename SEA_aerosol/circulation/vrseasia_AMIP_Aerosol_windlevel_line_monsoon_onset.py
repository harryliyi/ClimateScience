# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# set up data directories and filenames
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

# set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/circulation/onset/"

# set up variable names and file name
varname = 'U850'
varreader = 'U'
varfname = "85hPa_uwind"
varstr = "850hPa zonal wind"
var_unit = 'm/s'
var_res = "fv09"
prelev = 850

fname1 = var_res+"_WIND_"+case1+".cam.h0.0001-0050.nc"
fname2 = var_res+"_WIND_"+case2+".cam.h0.0001-0050.nc"
fname3 = var_res+"_WIND_"+case3+".cam.h0.0001-0050.nc"
fname4 = var_res+"_WIND_"+case4+".cam.h0.0001-0050.nc"
fname5 = var_res+"_WIND_"+case5+".cam.h0.0001-0050.nc"

# define inital year and end year
iniyear = 2
endyear = 50

# define the contour plot region
latbounds1 = [5, 15]
lonbounds1 = [50, 80]

latbounds2 = [20, 30]
lonbounds2 = [60, 90]

latbounds3 = [5, 15]
lonbounds3 = [40, 80]


# month series
month = np.arange(1, 13, 1)
mname = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

################################################################################################
# S0-Define functions
################################################################################################


def getstats(var1, var2):
    n1 = var1.shape[0]
    n2 = var2.shape[0]

    var1mean = np.mean(var1, axis=0)
    var2mean = np.mean(var2, axis=0)
    var1std = np.std(var1, axis=0)
    var2std = np.std(var2, axis=0)

    vardiff = var1mean - var2mean
    varttest = vardiff/np.sqrt(var1std**2/n1+var2std**2/n2)

    return vardiff, abs(varttest)


################################################################################################
# S1-open climatological data
################################################################################################
# open data and read grids
fname1 = var_res+"_uwind_"+case1+".cam.h1.0001-01-01-00000_vertical_interp.nc"
fdata1 = Dataset(expdir1+fname1)

# read lat/lon grids
lats = fdata1.variables['lat'][:]
lons = fdata1.variables['lon'][:]
levs = fdata1.variables['lev'][:]

# latitude/longitude  lower and upper contour index

# withdrawal date from Syroka and Toumi 2004
latli1 = np.abs(lats - latbounds1[0]).argmin()
latui1 = np.abs(lats - latbounds1[1]).argmin()

lonli1 = np.abs(lons - lonbounds1[0]).argmin()
lonui1 = np.abs(lons - lonbounds1[1]).argmin()

latli2 = np.abs(lats - latbounds2[0]).argmin()
latui2 = np.abs(lats - latbounds2[1]).argmin()

lonli2 = np.abs(lons - lonbounds2[0]).argmin()
lonui2 = np.abs(lons - lonbounds2[1]).argmin()

latli3 = np.abs(lats - latbounds3[0]).argmin()
latui3 = np.abs(lats - latbounds3[1]).argmin()

lonli3 = np.abs(lons - lonbounds3[0]).argmin()
lonui3 = np.abs(lons - lonbounds3[1]).argmin()

levi = np.abs(levs - prelev).argmin()

var1 = np.zeros((endyear-iniyear+1)*365)
var2 = np.zeros((endyear-iniyear+1)*365)
var3 = np.zeros((endyear-iniyear+1)*365)
var4 = np.zeros((endyear-iniyear+1)*365)
var5 = np.zeros((endyear-iniyear+1)*365)

varw1 = np.zeros((endyear-iniyear+1)*365)
varw2 = np.zeros((endyear-iniyear+1)*365)
varw3 = np.zeros((endyear-iniyear+1)*365)
varw4 = np.zeros((endyear-iniyear+1)*365)
varw5 = np.zeros((endyear-iniyear+1)*365)

print('reading the data...')
for iyear in np.arange(iniyear, endyear+1, 1):
    if (iyear < 10):
        yearno = '000'+str(iyear)
    else:
        yearno = '00'+str(iyear)
    print('Current year is: '+yearno)

    fname1 = var_res+'_uwind_'+case1+'.cam.h1.'+yearno+'-01-01-00000_vertical_interp.nc'
    fname2 = var_res+'_uwind_'+case2+'.cam.h1.'+yearno+'-01-01-00000_vertical_interp.nc'
    fname3 = var_res+'_uwind_'+case3+'.cam.h1.'+yearno+'-01-01-00000_vertical_interp.nc'
    fname4 = var_res+'_uwind_'+case4+'.cam.h1.'+yearno+'-01-01-00000_vertical_interp.nc'
    fname5 = var_res+'_uwind_'+case5+'.cam.h1.'+yearno+'-01-01-00000_vertical_interp.nc'

    fdata1 = Dataset(expdir1+fname1)
    fdata2 = Dataset(expdir2+fname2)
    fdata3 = Dataset(expdir3+fname3)
    fdata4 = Dataset(expdir4+fname4)
    fdata5 = Dataset(expdir5+fname5)

    templ1 = fdata1.variables[varreader][:, levi, latli1:latui1+1, lonli1:lonui1+1]
    templ2 = fdata2.variables[varreader][:, levi, latli1:latui1+1, lonli1:lonui1+1]
    templ3 = fdata3.variables[varreader][:, levi, latli1:latui1+1, lonli1:lonui1+1]
    templ4 = fdata4.variables[varreader][:, levi, latli1:latui1+1, lonli1:lonui1+1]
    templ5 = fdata5.variables[varreader][:, levi, latli1:latui1+1, lonli1:lonui1+1]

    tempu1 = fdata1.variables[varreader][:, levi, latli2:latui2+1, lonli2:lonui2+1]
    tempu2 = fdata2.variables[varreader][:, levi, latli2:latui2+1, lonli2:lonui2+1]
    tempu3 = fdata3.variables[varreader][:, levi, latli2:latui2+1, lonli2:lonui2+1]
    tempu4 = fdata4.variables[varreader][:, levi, latli2:latui2+1, lonli2:lonui2+1]
    tempu5 = fdata5.variables[varreader][:, levi, latli2:latui2+1, lonli2:lonui2+1]

    tempw1 = fdata1.variables[varreader][:, levi, latli3:latui3+1, lonli3:lonui3+1]
    tempw2 = fdata2.variables[varreader][:, levi, latli3:latui3+1, lonli3:lonui3+1]
    tempw3 = fdata3.variables[varreader][:, levi, latli3:latui3+1, lonli3:lonui3+1]
    tempw4 = fdata4.variables[varreader][:, levi, latli3:latui3+1, lonli3:lonui3+1]
    tempw5 = fdata5.variables[varreader][:, levi, latli3:latui3+1, lonli3:lonui3+1]

    var1[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(templ1, axis=2), axis=1) - np.mean(np.mean(tempu1, axis=2), axis=1)
    var2[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(templ2, axis=2), axis=1) - np.mean(np.mean(tempu2, axis=2), axis=1)
    var3[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(templ3, axis=2), axis=1) - np.mean(np.mean(tempu3, axis=2), axis=1)
    var4[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(templ4, axis=2), axis=1) - np.mean(np.mean(tempu4, axis=2), axis=1)
    var5[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(templ5, axis=2), axis=1) - np.mean(np.mean(tempu5, axis=2), axis=1)

    varw1[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(tempw1, axis=2), axis=1)
    varw2[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(tempw2, axis=2), axis=1)
    varw3[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(tempw3, axis=2), axis=1)
    varw4[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(tempw4, axis=2), axis=1)
    varw5[(iyear-iniyear)*365:(iyear-iniyear+1)*365] = np.mean(np.mean(tempw5, axis=2), axis=1)

################################################################################################
# S1-Calculate mean wind index
################################################################################################

var1 = np.convolve(var1, np.ones((7,))/7, mode='valid')
var2 = np.convolve(var2, np.ones((7,))/7, mode='valid')
var3 = np.convolve(var3, np.ones((7,))/7, mode='valid')
var4 = np.convolve(var4, np.ones((7,))/7, mode='valid')
var5 = np.convolve(var5, np.ones((7,))/7, mode='valid')
print(len(var1))

# Syroka and Toumi 2004
var1_mean = np.zeros(365)
var2_mean = np.zeros(365)
var3_mean = np.zeros(365)
var4_mean = np.zeros(365)
var5_mean = np.zeros(365)

for iday in range(365):
    var1_mean[iday] = np.mean(var1[iday::365])
    var2_mean[iday] = np.mean(var2[iday::365])
    var3_mean[iday] = np.mean(var3[iday::365])
    var4_mean[iday] = np.mean(var4[iday::365])
    var5_mean[iday] = np.mean(var5[iday::365])

plt.clf()
fig = plt.figure()
xdays = np.arange(365)+1

plt.plot(xdays, var1_mean, c='k', linewidth=1., label=r'$S_{CTRL}$')
plt.plot(xdays, var2_mean, c='r', linewidth=1., label=r'$S_{2000}A_{2000}$')
plt.plot(xdays, var3_mean, c='b', linewidth=1., label=r'$S_{PERT}A_{2000}$')
plt.plot(xdays, var4_mean, c='g', linewidth=1., label=r'$S_{2000}A_{1950}$')
plt.plot(xdays, var5_mean, c='m', linewidth=1., label=r'$S_{PERT}A_{1950}$')

plt.legend(loc='upper left', fontsize=5)

xticks = []
for imon in range(12):
    if imon == 0:
        xticks.append(int(mdays[imon]/2))
    else:
        xticks.append(int(np.sum(mdays[0:imon])+mdays[imon]/2))

plt.axhline(y=0, color='k', linewidth=1.)
plt.xticks(xticks, mname, fontsize=7)
plt.yticks(fontsize=7)
plt.ylabel(varstr+' '+var_unit, fontsize=7)
plt.xlabel('Julian Days', fontsize=7)

plt.savefig(outdir+"vrseasia_aerosol_amip_monsoon_onset_ST_index.png", dpi=1200, bbox_inches='tight')
plt.title("Indian Monsoon circulation index", fontsize=10, y=-1.05)
plt.savefig(outdir+"vrseasia_aerosol_amip_monsoon_onset_ST_index.pdf", bbox_inches='tight')
plt.close(fig)

# Wang et al 2008
var1_mean = np.zeros(365)
var2_mean = np.zeros(365)
var3_mean = np.zeros(365)
var4_mean = np.zeros(365)
var5_mean = np.zeros(365)

for iday in range(365):
    var1_mean[iday] = np.mean(varw1[iday::365])
    var2_mean[iday] = np.mean(varw2[iday::365])
    var3_mean[iday] = np.mean(varw3[iday::365])
    var4_mean[iday] = np.mean(varw4[iday::365])
    var5_mean[iday] = np.mean(varw5[iday::365])

plt.clf()
fig = plt.figure()
xdays = np.arange(365)+1

plt.plot(xdays, var1_mean, c='k', linewidth=1., label=r'$S_{CTRL}$')
plt.plot(xdays, var2_mean, c='r', linewidth=1., label=r'$S_{2000}A_{2000}$')
plt.plot(xdays, var3_mean, c='b', linewidth=1., label=r'$S_{PERT}A_{2000}$')
plt.plot(xdays, var4_mean, c='g', linewidth=1., label=r'$S_{2000}A_{1950}$')
plt.plot(xdays, var5_mean, c='m', linewidth=1., label=r'$S_{PERT}A_{1950}$')

plt.legend(loc='upper left', fontsize=9)

xticks = []
for imon in range(12):
    if imon == 0:
        xticks.append(int(mdays[imon]/2))
    else:
        xticks.append(int(np.sum(mdays[0:imon])+mdays[imon]/2))

plt.axhline(y=6.2, color='k', linewidth=1.)
plt.xticks(xticks, mname, fontsize=7)
plt.yticks(fontsize=7)
plt.ylabel(varstr+' '+var_unit, fontsize=8)
plt.xlabel('Months', fontsize=8)

plt.savefig(outdir+"vrseasia_aerosol_amip_monsoon_onset_Wang_index.png", dpi=1200, bbox_inches='tight')
plt.title("Indian Monsoon circulation index", fontsize=10, y=-1.05)
plt.savefig(outdir+"vrseasia_aerosol_amip_monsoon_onset_Wang_index.pdf", bbox_inches='tight')
plt.close(fig)
