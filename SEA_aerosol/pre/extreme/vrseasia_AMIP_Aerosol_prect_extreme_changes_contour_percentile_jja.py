# this script is used to compare vrcesm against observations
# here extremes is presented
# by Harry Li


# import libraries
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
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

# set up land mask directory
rdir = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"

# set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/pre/extreme/prect_extreme_changes/JJA/"

# set up variable names and file name
varname = 'PRECT'

varstr = "Total Precip"
var_res = "fv02"
varfname = 'prect'
var_unit = 'mm/day'

# define inital year and end year
iniyear = 2
endyear = 50

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

# month series
month = np.arange(1, 13, 1)
mname = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# define the season period
inimonth = 6
endmonth = 8

iniday = np.sum(mdays[0:inimonth-1])
endday = np.sum(mdays[0:endmonth])
print(iniday)
print(endday)
ndays = endday-iniday

# set up percentile
percentile = 99
percentile_ranges = [50, 70, 80, 90, 95, 97, 99, 99.5, 99.9]

# set up nbins
nbins = 50

outdir_stats = outdir
outdir = outdir + str(percentile)+'th/'

################################################################################################
# S0-Define functions
################################################################################################


def plotextremes(lons, lats, var, relevel, titlestr, fname):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6, linewidth=0.1)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-125., 125.1, 25.)
    cs = map.contourf(x, y, var, clevs, cmap=cm.BrBG, alpha=0.9, extend="both")

    # add colorbar.
    cbar = map.colorbar(cs, location='bottom', pad="5%")
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+' ['+var_unit+']')

    # add title
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_"+str(percentile)+"_th_extremes_SEA_contour_"+str(inimonth)+"to"+str(endmonth)+"_"+relevel+"_"+fname+".png", dpi=600)
    plt.title(titlestr+" JJA "+str(percentile)+"th percentile precip "+relevel+"-year return level", fontsize=10, y=1.08)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_"+str(percentile)+"_th_extremes_SEA_contour_"+str(inimonth)+"to"+str(endmonth)+"_"+relevel+"_"+fname+".pdf")

    plt.close(fig)


# plot for response
def plotdiff(lons, lats, res, forcingstr, forcingfname):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6, linewidth=0.1)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-20., 22., 2.)
    norm = mpl.colors.BoundaryNorm(boundaries=clevs, ncolors=256)
    cs = map.pcolormesh(x, y, res, cmap=cm.BrBG, alpha=0.9, norm=norm, vmax=75., vmin=-75.)

    # add colorbar.
    cbar = map.colorbar(cs, location='bottom', pad="5%")
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label(varstr+' ['+var_unit+']')

    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_"+str(percentile)+"_th_extremes_SEA_contour_response_"+str(inimonth)+"to"+str(endmonth)+"_"+forcingfname+".png", dpi=600)
    plt.title(forcingstr+" "+varstr+" changes", fontsize=7, y=1.08)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_"+str(percentile)+"_th_extremes_SEA_contour_response_"+str(inimonth)+"to"+str(endmonth)+"_"+forcingfname+".pdf")


# plot for all responses together
def plotalldiff(lons, lats, res1, res2, res3):
    fig = plt.figure(figsize=(8, 12))
    # total response
    ax1 = fig.add_subplot(311)
    # ax1.set_title(r'$\Delta_{total} Rp$'+str(percentile), fontsize=10, pad=3)
    ax1.set_title('Total response', fontsize=10, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6, linewidth=0.1)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-40, 40.1, 10.)
    norm = mpl.colors.BoundaryNorm(boundaries=clevs, ncolors=256)
    cs = map.pcolormesh(x, y, res1, cmap=cm.BrBG, alpha=0.9, vmax=75., vmin=-75., norm=norm)

    # fast response
    ax2 = fig.add_subplot(312)
    # ax2.set_title(r'$\Delta_{fast} Rp$'+str(percentile), fontsize=10, pad=3)
    ax2.set_title('Atmospheric-forced', fontsize=10, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6, linewidth=0.1)
    cs = map.pcolormesh(x, y, res2, cmap=cm.BrBG, alpha=0.9, vmax=75., vmin=-75., norm=norm)

    # slow response
    ax3 = fig.add_subplot(313)
    # ax3.set_title(r'$\Delta_{slow} Rp$'+str(percentile), fontsize=10, pad=3)
    ax3.set_title('Ocean-mediated', fontsize=10, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6, linewidth=0.1)
    cs = map.pcolormesh(x, y, res3, cmap=cm.BrBG, alpha=0.9, vmax=75., vmin=-75., norm=norm)

    # add colorbar.
    fig.subplots_adjust(right=0.83, hspace=0.18)
    cbar_ax = fig.add_axes([0.80, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical', ticks=clevs)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Relative changes in precipitation extremes [%]', fontsize=9, labelpad=0.7)

    # add title
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_"+str(percentile) +
                "_th_extremes_SEA_contour_response_"+str(inimonth)+"to"+str(endmonth)+"_aerosolsinone.png", dpi=600, bbox_inches='tight')
    plt.suptitle("Aerosol Responses "+varstr+" Rp"+str(percentile)+" changes", fontsize=8, y=0.95)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_"+str(percentile) +
                "_th_extremes_SEA_contour_response_"+str(inimonth)+"to"+str(endmonth)+"_aerosolsinone.pdf", bbox_inches='tight')


################################################################################################
# S1-open daily data
################################################################################################

# open data and read grids
fname1 = var_res+"_PREC_"+case1+".cam.h0.0001-0050.nc"
fdata1 = Dataset(expdir1+fname1)

# read lat/lon grids
lats = fdata1.variables['lat'][:]
lons = fdata1.variables['lon'][:]

# latitude/longitude  lower and upper contour index
latli = np.abs(lats - latbounds[0]).argmin()
latui = np.abs(lats - latbounds[1]).argmin()

lonli = np.abs(lons - lonbounds[0]).argmin()
lonui = np.abs(lons - lonbounds[1]).argmin()

lats = lats[latli:latui+1]
lons = lons[lonli:lonui+1]

nlats = latui - latli + 1
nlons = lonui - lonli + 1

print(nlats)
print(nlons)

reg_latli = np.abs(lats - reg_lats[0]).argmin()
reg_latui = np.abs(lats - reg_lats[1]).argmin()

reg_lonli = np.abs(lons - reg_lons[0]).argmin()
reg_lonui = np.abs(lons - reg_lons[1]).argmin()


obsrate = (100.-percentile)/100.

var1 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
var2 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
var3 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
var4 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))
var5 = np.zeros(((endyear-iniyear+1)*ndays, nlats, nlons))

print('reading the data...')
for iyear in np.arange(iniyear, endyear+1, 1):
    if (iyear < 10):
        yearno = '000'+str(iyear)
    else:
        yearno = '00'+str(iyear)
    print('Current year is: '+yearno)

    fname1 = var_res+'_prect_'+case1+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname2 = var_res+'_prect_'+case2+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname3 = var_res+'_prect_'+case3+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname4 = var_res+'_prect_'+case4+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname5 = var_res+'_prect_'+case5+'.cam.h1.'+yearno+'-01-01-00000.nc'

    fdata1 = Dataset(expdir1+fname1)
    fdata2 = Dataset(expdir2+fname2)
    fdata3 = Dataset(expdir3+fname3)
    fdata4 = Dataset(expdir4+fname4)
    fdata5 = Dataset(expdir5+fname5)

    temp1 = fdata1.variables[varname][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000
    temp2 = fdata2.variables[varname][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000
    temp3 = fdata3.variables[varname][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000
    temp4 = fdata4.variables[varname][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000
    temp5 = fdata5.variables[varname][iniday: endday, latli:latui+1, lonli:lonui+1] * 86400 * 1000

    var1[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp1.copy()
    var2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp2.copy()
    var3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp3.copy()
    var4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp4.copy()
    var5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp5.copy()

################################################################################################
# S2-calculate for extremes
################################################################################################

var1_extreme = np.percentile(var1, percentile, axis=0)
var2_extreme = np.percentile(var2, percentile, axis=0)
var3_extreme = np.percentile(var3, percentile, axis=0)
var4_extreme = np.percentile(var4, percentile, axis=0)
var5_extreme = np.percentile(var5, percentile, axis=0)


################################################################################################
# S3-plot for response
################################################################################################

print('plotting responses...')

# all aerosol foring
forcingstr = "All aerosol forcings"
forcingfname = "allaerosols"
res = var2_extreme - var5_extreme

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol fast response1
forcingstr = "Aerosol fast response"
forcingfname = "fastaerosol1"
res = var2_extreme - var4_extreme

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol slow response1
forcingstr = "Aerosol slow response"
forcingfname = "slowaerosol1"
res = var4_extreme - var5_extreme

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol fast response2
forcingstr = "Aerosol fast response"
forcingfname = "fastaerosol2"
res = var3_extreme - var5_extreme

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol slow response2
forcingstr = "Aerosol slow response"
forcingfname = "slowaerosol2"
res = var2_extreme - var3_extreme

plotdiff(lons, lats, res, forcingstr, forcingfname)


# GHG and natural forcing
forcingstr = "GHG and natural forcings"
forcingfname = "GHGforcings"
res = var5_extreme - var1_extreme

plotdiff(lons, lats, res, forcingstr, forcingfname)


# All forcings
forcingstr = "All forcings"
forcingfname = "allforcings"
res = var2_extreme - var1_extreme

plotdiff(lons, lats, res, forcingstr, forcingfname)


#################################################################################################
# plot all aerosol respoenses in one figure


res1 = (var2_extreme - var5_extreme)/var5_extreme*100
res2 = ((var2_extreme + var3_extreme - var4_extreme - var5_extreme)/2)/(var4_extreme+var5_extreme)*2*100
res3 = (var4_extreme + var2_extreme - var5_extreme - var3_extreme)/2/(var5_extreme+var3_extreme)*2*100

plotalldiff(lons, lats, res1, res2, res3)
