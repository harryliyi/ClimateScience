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
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/pre/extreme/prect_extreme_changes/JJA/heavy_counts/"

# set up variable names and file name
varname = 'PRECT'

varstr = "Total Precip"
var_res = "fv02"
varfname = 'prect'
var_unit = 'mm/day'

# define inital year and end year
iniyear = 2
endyear = 50
nyears = endyear - iniyear + 1

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

# set up moderate thresholds
mod_thres = [5, 50]

# set up nbins
nbins = 50


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
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_extremes_SEA_contour_"+str(inimonth)+"to"+str(endmonth)+"_"+relevel+"_"+fname+".png", dpi=1200)
    plt.title(titlestr+" JJA precip "+relevel+"-year return level", fontsize=10, y=1.08)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_extremes_SEA_contour_"+str(inimonth)+"to"+str(endmonth)+"_"+relevel+"_"+fname+".pdf")

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

    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_extremes_SEA_contour_response_"+str(inimonth)+"to"+str(endmonth)+"_"+forcingfname+".png", dpi=1200)
    plt.title(forcingstr+" "+varstr+" changes", fontsize=7, y=1.08)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+varfname+"_extremes_SEA_contour_response_"+str(inimonth)+"to"+str(endmonth)+"_"+forcingfname+".pdf")
    plt.close(fig)


# plot for all responses together
def plotalldiff(lons, lats, res1, res2, res3, levname):
    fig = plt.figure()
    # total response
    ax1 = fig.add_subplot(311)
    ax1.set_title(r'$\Delta_{total} Counts$', fontsize=5, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    clevs = np.arange(-60, 60.1, 15.)
    norm = mpl.colors.BoundaryNorm(boundaries=clevs, ncolors=256)
    cs = map.pcolormesh(x, y, res1, cmap=cm.BrBG, alpha=0.9, vmax=75., vmin=-75., norm=norm)

    # fast response
    ax2 = fig.add_subplot(312)
    ax2.set_title(r'$\Delta_{fast} Counts$', fontsize=5, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.pcolormesh(x, y, res2, cmap=cm.BrBG, alpha=0.9, vmax=75., vmin=-75., norm=norm)

    # slow response
    ax3 = fig.add_subplot(313)
    ax3.set_title(r'$\Delta_{slow} Counts$', fontsize=5, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1], llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    cs = map.pcolormesh(x, y, res3, cmap=cm.BrBG, alpha=0.9, vmax=75., vmin=-75., norm=norm)

    # add colorbar.
#    fig.subplots_adjust(right=0.7,hspace = 0.15)
    cbar_ax = fig.add_axes([0.69, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical', ticks=clevs)
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('Counts [days]', fontsize=4, labelpad=0.7)

    # add title
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_extremes_SEA_contour_response_" +
                str(inimonth)+"to"+str(endmonth)+"_"+levname+"_aerosolsinone.png", dpi=1200, bbox_inches='tight')
    plt.suptitle("Aerosol Responses "+levname+" "+varstr+" changes", fontsize=8, y=0.95)
    plt.savefig(outdir+"vrseasia_aerosol_amip_jja_"+var_res+"_"+varfname+"_extremes_SEA_contour_response_"+str(inimonth)+"to"+str(endmonth)+"_"+levname+"_aerosolsinone.pdf", bbox_inches='tight')
    plt.close(fig)


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
# S2-calculate for moderate and extreme precipitation counts
################################################################################################

var1_mod = (var1 >= mod_thres[0]) & (var1 < mod_thres[1])
var1_mod_cnt = np.sum(var1_mod, axis=0)/nyears
var1_ext = (var1 >= mod_thres[1])
var1_ext_cnt = np.sum(var1_ext, axis=0)/nyears

var2_mod = (var2 >= mod_thres[0]) & (var2 < mod_thres[1])
var2_mod_cnt = np.sum(var2_mod, axis=0)/nyears
var2_ext = (var2 >= mod_thres[1])
var2_ext_cnt = np.sum(var2_ext, axis=0)/nyears

var3_mod = (var3 >= mod_thres[0]) & (var3 < mod_thres[1])
var3_mod_cnt = np.sum(var3_mod, axis=0)/nyears
var3_ext = (var3 >= mod_thres[1])
var3_ext_cnt = np.sum(var3_ext, axis=0)/nyears

var4_mod = (var4 >= mod_thres[0]) & (var4 < mod_thres[1])
var4_mod_cnt = np.sum(var4_mod, axis=0)/nyears
var4_ext = (var4 >= mod_thres[1])
var4_ext_cnt = np.sum(var4_ext, axis=0)/nyears

var5_mod = (var5 >= mod_thres[0]) & (var5 < mod_thres[1])
var5_mod_cnt = np.sum(var5_mod, axis=0)/nyears
var5_ext = (var5 >= mod_thres[1])
var5_ext_cnt = np.sum(var5_ext, axis=0)/nyears


################################################################################################
# S3-plot for response moderate precip
################################################################################################

print('plotting responses...')

# all aerosol foring
forcingstr = "moderate All aerosol forcings"
forcingfname = "moderate_allaerosols"
res = var2_mod_cnt - var5_mod_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol fast response1
forcingstr = "moderate Aerosol fast response"
forcingfname = "moderate_fastaerosol1"
res = var2_mod_cnt - var4_mod_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol slow response1
forcingstr = "moderate Aerosol slow response"
forcingfname = "moderate_slowaerosol1"
res = var4_mod_cnt - var5_mod_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol fast response2
forcingstr = "moderate Aerosol fast response"
forcingfname = "moderate_fastaerosol2"
res = var3_mod_cnt - var5_mod_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol slow response2
forcingstr = "moderate Aerosol slow response"
forcingfname = "moderate_slowaerosol2"
res = var2_mod_cnt - var3_mod_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# GHG and natural forcing
forcingstr = "moderate GHG and natural forcings"
forcingfname = "moderate_GHGforcings"
res = var5_mod_cnt - var1_mod_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# All forcings
forcingstr = "moderate All forcings"
forcingfname = "moderate_allforcings"
res = var2_mod_cnt - var1_mod_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


#################################################################################################
# plot all aerosol respoenses in one figure


res1 = (var2_mod_cnt - var5_mod_cnt)/var5_mod_cnt*100
res2 = ((var2_mod_cnt + var3_mod_cnt - var4_mod_cnt - var5_mod_cnt)/2)/(var4_mod_cnt+var5_mod_cnt)*2*100
res3 = (var4_mod_cnt + var2_mod_cnt - var5_mod_cnt - var3_mod_cnt)/2/(var5_mod_cnt+var3_mod_cnt)*2*100

plotalldiff(lons, lats, res1, res2, res3, 'moderate')


################################################################################################
# S3-plot for response heavy precip
################################################################################################

print('plotting responses...')

# all aerosol foring
forcingstr = "heavy All aerosol forcings"
forcingfname = "heavy_allaerosols"
res = var2_ext_cnt - var5_ext_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol fast response1
forcingstr = "heavy Aerosol fast response"
forcingfname = "heavy_fastaerosol1"
res = var2_ext_cnt - var4_ext_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol slow response1
forcingstr = "heavy Aerosol slow response"
forcingfname = "heavy_slowaerosol1"
res = var4_ext_cnt - var5_ext_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol fast response2
forcingstr = "heavy Aerosol fast response"
forcingfname = "heavy_fastaerosol2"
res = var3_ext_cnt - var5_ext_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# aerosol slow response2
forcingstr = "heavy Aerosol slow response"
forcingfname = "heavy_slowaerosol2"
res = var2_ext_cnt - var3_ext_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# GHG and natural forcing
forcingstr = "heavy GHG and natural forcings"
forcingfname = "heavy_GHGforcings"
res = var5_ext_cnt - var1_ext_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


# All forcings
forcingstr = "heavy All forcings"
forcingfname = "heavy_allforcings"
res = var2_ext_cnt - var1_ext_cnt

plotdiff(lons, lats, res, forcingstr, forcingfname)


#################################################################################################
# plot all aerosol respoenses in one figure


res1 = (var2_ext_cnt - var5_ext_cnt)/var5_ext_cnt*100
res2 = ((var2_ext_cnt + var3_ext_cnt - var4_ext_cnt - var5_ext_cnt)/2)/(var4_ext_cnt+var5_ext_cnt)*2*100
res3 = (var4_ext_cnt + var2_ext_cnt - var5_ext_cnt - var3_ext_cnt)/2/(var5_ext_cnt+var3_ext_cnt)*2*100

plotalldiff(lons, lats, res1, res2, res3, 'heavy')
