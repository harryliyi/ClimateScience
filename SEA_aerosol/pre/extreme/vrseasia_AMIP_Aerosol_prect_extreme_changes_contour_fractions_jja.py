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
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/pre/extreme/prect_extreme_changes/JJA/fractions/"

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

# define regions
reg_names = ['mainland SEA', 'Central India', 'South India', 'North India']
reg_str = ['mainSEA', 'ctInd', 'stInd', 'nrInd']
reg_lats = [[10, 20], [16.5, 26.5], [8, 20], [20, 28]]
reg_lons = [[100, 110], [74.5, 86.5], [70, 90], [65, 90]]

# set up moderate thresholds
# thres_bots = [0, 1, 5, 10, 20, 40, 60, 10, 30]
# thres_tops = [1, 5, 10, 20, 40, 60, 50000, 30, 50000]

thres_bots = [60, 30]
thres_tops = [50000, 50000]

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
# percentile_bots = [0, 50, 70, 70, 80, 90, 95, 95, 99]
# percentile_tops = [50, 70, 80, 90, 90, 95, 100, 99, 100]

percentile_bots = [95, 95, 99]
percentile_tops = [100, 99, 100]

# set up nbins
nbins = 50

outdir_stats = outdir


################################################################################################
# S0-Define functions
################################################################################################

def plot_box(plot_data, labels, yticks, ylabel, title, fname):

    box_data = []
    for idata in range(len(plot_data)):
        temp_data = plot_data[idata]
        temp_data = temp_data.flatten()
        temp_data = temp_data[~np.isnan(temp_data)]
        box_data.append(temp_data)

    # print(len(box_data[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.boxplot(box_data, sym='')
    # ax.boxplot(box_data)
    ax.axhline(y=0., linewidth=2, c='black', linestyle='solid')
    xticks = np.arange(len(box_data))+1
    xticknames = labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticknames, fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=8)
    ax.tick_params(axis='y', which='major', labelsize=6)

    ax.set_ylabel(ylabel, fontsize=8)

    # add title
    plt.savefig(fname+'.png', dpi=600, bbox_inches='tight')
    plt.suptitle(title, fontsize=8, y=0.95)
    plt.savefig(fname+'.pdf', bbox_inches='tight')
    plt.close(fig)


# plot for differences
def plotcontour(lons, lats, var, clevs, var_unit, title, fname):
    fig = plt.figure()
    # Res1
    ax1 = fig.add_subplot(111)
    # ax1.set_title(label[0], fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    # clevs = np.arange(-18, 18.1, 3)
    cs = map.contourf(x, y, var, clevs, cmap=cm.Spectral_r, alpha=0.9, extend="both")

    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('['+var_unit+']', fontsize=6, labelpad=-0.3)

    # add title
    plt.savefig(fname+".png", dpi=600, bbox_inches='tight')
    plt.suptitle(title, fontsize=7, y=1.08)
    plt.savefig(fname+".pdf", bbox_inches='tight')

    plt.close(fig)


# plot for differences
def plotaerodiff(lons, lats, res1, res2, res3, label, clevs, var_unit, forcingstr, fname):
    fig = plt.figure()
    # Res1
    ax1 = fig.add_subplot(311)
    ax1.set_title(label[0], fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    # clevs = np.arange(-18, 18.1, 3)
    cs = map.contourf(x, y, res1, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")

    # Res2
    ax2 = fig.add_subplot(312)
    ax2.set_title(label[1], fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    cs = map.contourf(x, y, res2, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")

    # Res3
    ax3 = fig.add_subplot(313)
    ax3.set_title(label[2], fontsize=7, pad=3)
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4)
    cs = map.contourf(x, y, res3, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")

    fig.subplots_adjust(hspace=0.2)
    cbar_ax = fig.add_axes([0.69, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('['+var_unit+']', fontsize=6, labelpad=-0.3)

    # add title
    plt.savefig(fname+".png", dpi=600, bbox_inches='tight')
    plt.suptitle(forcingstr+" "+varstr+" changes", fontsize=7, y=1.08)
    plt.savefig(fname+".pdf", bbox_inches='tight')

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

# read land mask
dir_lndfrc = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
flndfrc = 'USGS-gtopo30_0.23x0.31_remap_c061107.nc'
dataset_lndfrc = Dataset(dir_lndfrc+flndfrc)
lndfrc = dataset_lndfrc.variables['LANDFRAC'][latli:latui+1, lonli:lonui+1]


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

# for idx in range((endyear-iniyear+1)*ndays):
#     var1[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var1[idx, :, :])
#     var2[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var2[idx, :, :])
#     var3[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var3[idx, :, :])
#     var4[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var4[idx, :, :])
#     var5[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var5[idx, :, :])

################################################################################################
# S2-calculate the % of total precipitation
################################################################################################

for idx, percent_bot in enumerate(percentile_bots):
    percent_top = percentile_tops[idx]
    outdir_percent = outdir + str(percent_bot) + 'th/'

    print('Calculate the % of total precipitation for between '+str(percent_bot)+'th and '+str(percent_top)+'th...')

    var1_bot = np.percentile(var1, percent_bot, axis=0)
    var2_bot = np.percentile(var2, percent_bot, axis=0)
    var3_bot = np.percentile(var3, percent_bot, axis=0)
    var4_bot = np.percentile(var4, percent_bot, axis=0)
    var5_bot = np.percentile(var5, percent_bot, axis=0)

    if (percent_bot == 0):
        var1_bot[:, :] = 0.
        var2_bot[:, :] = 0.
        var3_bot[:, :] = 0.
        var4_bot[:, :] = 0.
        var5_bot[:, :] = 0.

    var1_top = np.percentile(var1, percent_top, axis=0)
    var2_top = np.percentile(var2, percent_top, axis=0)
    var3_top = np.percentile(var3, percent_top, axis=0)
    var4_top = np.percentile(var4, percent_top, axis=0)
    var5_top = np.percentile(var5, percent_top, axis=0)

    if (percent_top == 0):
        var1_top[:, :] = 50000.
        var2_top[:, :] = 50000.
        var3_top[:, :] = 50000.
        var4_top[:, :] = 50000.
        var5_top[:, :] = 50000.

    var_frc1 = np.zeros((nlats, nlons))
    var_frc2 = np.zeros((nlats, nlons))
    var_frc3 = np.zeros((nlats, nlons))
    var_frc4 = np.zeros((nlats, nlons))
    var_frc5 = np.zeros((nlats, nlons))

    for ilat in range(nlats):
        for ilon in range(nlons):
            temp = var1[:, ilat, ilon]
            var_frc1[ilat, ilon] = np.sum(temp[(temp >= var1_bot[ilat, ilon]) & (temp < var1_top[ilat, ilon])])/np.sum(temp)
            temp = var2[:, ilat, ilon]
            var_frc2[ilat, ilon] = np.sum(temp[(temp >= var2_bot[ilat, ilon]) & (temp < var2_top[ilat, ilon])])/np.sum(temp)
            temp = var3[:, ilat, ilon]
            var_frc3[ilat, ilon] = np.sum(temp[(temp >= var3_bot[ilat, ilon]) & (temp < var3_top[ilat, ilon])])/np.sum(temp)
            temp = var4[:, ilat, ilon]
            var_frc4[ilat, ilon] = np.sum(temp[(temp >= var4_bot[ilat, ilon]) & (temp < var4_top[ilat, ilon])])/np.sum(temp)
            temp = var5[:, ilat, ilon]
            var_frc5[ilat, ilon] = np.sum(temp[(temp >= var5_bot[ilat, ilon]) & (temp < var5_top[ilat, ilon])])/np.sum(temp)

    var_frc1 = var_frc1 * 100
    var_frc2 = var_frc2 * 100
    var_frc3 = var_frc3 * 100
    var_frc4 = var_frc4 * 100
    var_frc5 = var_frc5 * 100

    # plot for five cases
    clevs = np.arange(0, 26, 2)
    var_unit = '%'

    var = var_frc1
    title = r'$S_{CTRL}$'+' '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_case1'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_percent+fname)

    var = var_frc2
    title = r'$S_{2000}A_{2000}$'+' '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_case2'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_percent+fname)

    var = var_frc3
    title = r'$S_{PERT}A_{2000}$'+' '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_case3'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_percent+fname)

    var = var_frc4
    title = r'$S_{2000}A_{1950}$'+' '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_case4'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_percent+fname)

    var = var_frc5
    title = r'$S_{PERT}A_{1950}$'+' '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_case5'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_percent+fname)

    # plot for respenses
    clevs = np.arange(-6, 7, 2)
    var_unit = '%'

    ########################################
    # all aerosol response
    res = var_frc2 - var_frc5
    title = 'All aerosol response '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_changes_allaerosols'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_percent+fname)

    ########################################
    # aerosol fast response
    res = (var_frc2 + var_frc3 - var_frc4 - var_frc5)/2
    title = 'Atmospheric forced response '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_changes_fastaerosol'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_percent+fname)

    ########################################
    # aerosol slow response
    res = (var_frc4 + var_frc2 - var_frc5 - var_frc3)/2
    title = 'Ocean Mediated response '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_changes_slowaerosol'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_percent+fname)

    ########################################
    # GHG and natural forcing
    res = var_frc5 - var_frc1
    title = 'GHG and natural forcings response '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_changes_GHGforcings'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_percent+fname)

    ########################################
    # All forcings
    res = var_frc2 - var_frc1
    title = 'All forcings response '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_changes_allforcings'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_percent+fname)

    ########################################
    # All aerosol together
    res1 = var_frc2 - var_frc5
    res2 = (var_frc2 + var_frc3 - var_frc4 - var_frc5)/2
    res3 = (var_frc4 + var_frc2 - var_frc5 - var_frc3)/2
    label = ['All aerosol', 'Atmospheric-forced', 'Ocean-mediated']
    title = 'Aerosol response '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_changes_aerosolsinone'
    plotaerodiff(lons, lats, res1, res2, res3, label, clevs, var_unit, title, outdir_percent+fname)

    for ireg in range(len(reg_names)):
        print('Plotting for region: '+reg_names[ireg])
        reg_latli = np.abs(lats - reg_lats[ireg][0]).argmin()
        reg_latui = np.abs(lats - reg_lats[ireg][1]).argmin()

        reg_lonli = np.abs(lons - reg_lons[ireg][0]).argmin()
        reg_lonui = np.abs(lons - reg_lons[ireg][1]).argmin()

        # plot for heavy precipitation
        var1_reg = var_frc1[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var2_reg = var_frc2[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var3_reg = var_frc3[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var4_reg = var_frc4[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var5_reg = var_frc5[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]

        # plot for each cases
        plot_data = [var1_reg, var2_reg, var3_reg, var4_reg, var5_reg]

        labels = [r'$S_{CTRL}$', r'$S_{2000}A_{2000}$', r'$S_{PERT}A_{2000}$', r'$S_{2000}A_{1950}$', r'$S_{PERT}A_{1950}$']
        ylabel = 'As % of total precipitation [%]'
        yticks = np.arange(0, 25, 5)

        fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_'+reg_names[ireg]+'_box_exps'
        title = str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip over '+reg_names[ireg]
        plot_box(plot_data, labels, yticks, ylabel, title, outdir_percent+fname)

        # plot for absolute response
        labels = ['All', 'GHG', 'All Aerosol', 'ATM Aerosol', 'OCN Aerosol']

        res1 = var2_reg - var1_reg
        res2 = var5_reg - var1_reg
        res3 = var2_reg - var5_reg
        res4 = (var2_reg + var3_reg - var4_reg - var5_reg)/2
        res5 = (var4_reg + var2_reg - var5_reg - var3_reg)/2

        plot_data = [res1, res2, res3, res4, res5]

        ylabel = 'Changes in % of total precipitation [%]'
        yticks = np.arange(-6, 7, 2)

        fname = 'vrseasia_aerosol_amip_jja_'+str(percent_bot)+'-'+str(percent_top)+'th_prect_contour_percentile_fractions_changes_' + \
            reg_names[ireg]+'_box_absolute_response_'+str(inimonth)+"to"+str(endmonth)
        title = 'Changes in '+str(percent_bot)+'-'+str(percent_top)+'th precip as % of total precip over ' + reg_names[ireg]
        plot_box(plot_data, labels, yticks, ylabel, title, outdir_percent+fname)

var_mean1 = np.mean(var1, axis=0)
var_mean2 = np.mean(var2, axis=0)
var_mean3 = np.mean(var3, axis=0)
var_mean4 = np.mean(var4, axis=0)
var_mean5 = np.mean(var5, axis=0)

# plot for respenses
clevs = np.arange(-1.6, 1.7, 0.2)
var_unit = 'mm/day'

########################################
# all aerosol response
res = var_mean2 - var_mean5
title = 'All aerosol response total precip'
fname = 'vrseasia_aerosol_amip_jja_prect_contour_changes_allaerosols'
plotcontour(lons, lats, res, clevs, var_unit, title, outdir+fname)

########################################
# aerosol fast response
res = (var_mean2 + var_mean3 - var_mean4 - var_mean5)/2
title = 'Atmospheric forced response total precip'
fname = 'vrseasia_aerosol_amip_jja_prect_contour_changes_fastaerosol'
plotcontour(lons, lats, res, clevs, var_unit, title, outdir+fname)

########################################
# aerosol slow response
res = (var_mean4 + var_mean2 - var_mean5 - var_mean3)/2
title = 'Ocean Mediated response total precip'
fname = 'vrseasia_aerosol_amip_jja_prect_contour_changes_slowaerosol'
plotcontour(lons, lats, res, clevs, var_unit, title, outdir+fname)

########################################
# GHG and natural forcing
res = var_mean5 - var_mean1
title = 'GHG and natural forcings response total precip'
fname = 'vrseasia_aerosol_amip_jja_prect_contour_changes_GHGforcings'
plotcontour(lons, lats, res, clevs, var_unit, title, outdir+fname)

########################################
# All forcings
res = var_mean2 - var_mean1
title = 'All forcings response total precip'
fname = 'vrseasia_aerosol_amip_jja_prect_contour_changes_allforcings'
plotcontour(lons, lats, res, clevs, var_unit, title, outdir+fname)

########################################
# All aerosol together
res1 = var_mean2 - var_mean5
res2 = (var_mean2 + var_mean3 - var_mean4 - var_mean5)/2
res3 = (var_mean4 + var_mean2 - var_mean5 - var_mean3)/2
label = ['All aerosol', 'Atmospheric-forced', 'Ocean-mediated']
title = 'Aerosol response total precip'
fname = 'vrseasia_aerosol_amip_jja_prect_contour_changes_aerosolsinone'
plotaerodiff(lons, lats, res1, res2, res3, label, clevs, var_unit, title, outdir+fname)

for ireg in range(len(reg_names)):
    print('Plotting for region: '+reg_names[ireg])
    reg_latli = np.abs(lats - reg_lats[ireg][0]).argmin()
    reg_latui = np.abs(lats - reg_lats[ireg][1]).argmin()

    reg_lonli = np.abs(lons - reg_lons[ireg][0]).argmin()
    reg_lonui = np.abs(lons - reg_lons[ireg][1]).argmin()

    # plot for heavy precipitation
    var1_reg = var_mean1[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
    var2_reg = var_mean2[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
    var3_reg = var_mean3[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
    var4_reg = var_mean4[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
    var5_reg = var_mean5[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]

    # plot for each cases
    plot_data = [var1_reg, var2_reg, var3_reg, var4_reg, var5_reg]

    labels = [r'$S_{CTRL}$', r'$S_{2000}A_{2000}$', r'$S_{PERT}A_{2000}$', r'$S_{2000}A_{1950}$', r'$S_{PERT}A_{1950}$']
    ylabel = 'Total precipitation [mm/day]'
    yticks = np.arange(0, 30, 5)

    fname = 'vrseasia_aerosol_amip_jja_prect_contour_'+reg_names[ireg]+'_box_exps'
    title = 'total precip over '+reg_names[ireg]
    plot_box(plot_data, labels, yticks, ylabel, title, outdir+fname)

    # plot for absolute response
    labels = ['All', 'GHG', 'All Aerosol', 'ATM Aerosol', 'OCN Aerosol']

    res1 = var2_reg - var1_reg
    res2 = var5_reg - var1_reg
    res3 = var2_reg - var5_reg
    res4 = (var2_reg + var3_reg - var4_reg - var5_reg)/2
    res5 = (var4_reg + var2_reg - var5_reg - var3_reg)/2

    plot_data = [res1, res2, res3, res4, res5]

    ylabel = 'Changes in total precipitation [mm/day]'
    yticks = np.arange(-2, 2.1, 0.4)

    fname = 'vrseasia_aerosol_amip_jja_prect_contour_changes_' + \
        reg_names[ireg]+'_box_absolute_response_'+str(inimonth)+"to"+str(endmonth)
    title = 'Changes in total precip as % of total precip over ' + reg_names[ireg]
    plot_box(plot_data, labels, yticks, ylabel, title, outdir+fname)


################################################################################################
# S3-calculate the % of total precipitation for mm
################################################################################################

for idx, thres_bot in enumerate(thres_bots):
    thres_top = thres_tops[idx]
    outdir_thres = outdir + str(thres_bot) + 'mm/'

    print('Calculate the % of total precipitation for between '+str(thres_bot)+'mm and '+str(thres_top)+'mm...')

    var_frc1 = np.zeros((nlats, nlons))
    var_frc2 = np.zeros((nlats, nlons))
    var_frc3 = np.zeros((nlats, nlons))
    var_frc4 = np.zeros((nlats, nlons))
    var_frc5 = np.zeros((nlats, nlons))

    for ilat in range(nlats):
        for ilon in range(nlons):
            temp = var1[:, ilat, ilon]
            var_frc1[ilat, ilon] = np.sum(temp[(temp >= thres_bot) & (temp < thres_top)])/np.sum(temp)
            temp = var2[:, ilat, ilon]
            var_frc2[ilat, ilon] = np.sum(temp[(temp >= thres_bot) & (temp < thres_top)])/np.sum(temp)
            temp = var3[:, ilat, ilon]
            var_frc3[ilat, ilon] = np.sum(temp[(temp >= thres_bot) & (temp < thres_top)])/np.sum(temp)
            temp = var4[:, ilat, ilon]
            var_frc4[ilat, ilon] = np.sum(temp[(temp >= thres_bot) & (temp < thres_top)])/np.sum(temp)
            temp = var5[:, ilat, ilon]
            var_frc5[ilat, ilon] = np.sum(temp[(temp >= thres_bot) & (temp < thres_top)])/np.sum(temp)

    var_frc1 = var_frc1 * 100
    var_frc2 = var_frc2 * 100
    var_frc3 = var_frc3 * 100
    var_frc4 = var_frc4 * 100
    var_frc5 = var_frc5 * 100

    # plot for five cases
    clevs = np.arange(0, 26, 2)
    var_unit = '%'

    var = var_frc1
    title = r'$S_{CTRL}$'+' '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_case1'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_thres+fname)

    var = var_frc2
    title = r'$S_{2000}A_{2000}$'+' '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_case2'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_thres+fname)

    var = var_frc3
    title = r'$S_{PERT}A_{2000}$'+' '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_case3'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_thres+fname)

    var = var_frc4
    title = r'$S_{2000}A_{1950}$'+' '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_case4'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_thres+fname)

    var = var_frc5
    title = r'$S_{PERT}A_{1950}$'+' '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_case5'
    plotcontour(lons, lats, var, clevs, var_unit, title, outdir_thres+fname)

    # plot for respenses
    clevs = np.arange(-6, 7, 2)
    var_unit = '%'

    ########################################
    # all aerosol response
    res = var_frc2 - var_frc5
    title = 'All aerosol response '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_changes_allaerosols'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_thres+fname)

    ########################################
    # aerosol fast response
    res = (var_frc2 + var_frc3 - var_frc4 - var_frc5)/2
    title = 'Atmospheric forced response '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_changes_fastaerosol'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_thres+fname)

    ########################################
    # aerosol slow response
    res = (var_frc4 + var_frc2 - var_frc5 - var_frc3)/2
    title = 'Ocean Mediated response '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_changes_slowaerosol'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_thres+fname)

    ########################################
    # GHG and natural forcing
    res = var_frc5 - var_frc1
    title = 'GHG and natural forcings response '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_changes_GHGforcings'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_thres+fname)

    ########################################
    # All forcings
    res = var_frc2 - var_frc1
    title = 'All forcings response '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_changes_allforcings'
    plotcontour(lons, lats, res, clevs, var_unit, title, outdir_thres+fname)

    ########################################
    # All aerosol together
    res1 = var_frc2 - var_frc5
    res2 = (var_frc2 + var_frc3 - var_frc4 - var_frc5)/2
    res3 = (var_frc4 + var_frc2 - var_frc5 - var_frc3)/2
    label = ['All aerosol', 'Atmospheric forced', 'Ocean mediated']
    title = 'Aerosol response '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip'
    fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_changes_aerosolsinone'
    plotaerodiff(lons, lats, res1, res2, res3, label, clevs, var_unit, title, outdir_thres+fname)

    for ireg in range(len(reg_names)):
        print('Plotting for region: '+reg_names[ireg])
        reg_latli = np.abs(lats - reg_lats[ireg][0]).argmin()
        reg_latui = np.abs(lats - reg_lats[ireg][1]).argmin()

        reg_lonli = np.abs(lons - reg_lons[ireg][0]).argmin()
        reg_lonui = np.abs(lons - reg_lons[ireg][1]).argmin()

        # plot for heavy precipitation
        var1_reg = var_frc1[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var2_reg = var_frc2[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var3_reg = var_frc3[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var4_reg = var_frc4[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var5_reg = var_frc5[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]

        # plot for each cases
        plot_data = [var1_reg, var2_reg, var3_reg, var4_reg, var5_reg]

        labels = [r'$S_{CTRL}$', r'$S_{2000}A_{2000}$', r'$S_{PERT}A_{2000}$', r'$S_{2000}A_{1950}$', r'$S_{PERT}A_{1950}$']
        ylabel = 'As % of total precipitation [%]'
        yticks = np.arange(0, 36, 5)

        fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_'+reg_names[ireg]+'_box_exps'
        title = str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip over '+reg_names[ireg]
        plot_box(plot_data, labels, yticks, ylabel, title, outdir_thres+fname)

        # plot for absolute response
        labels = ['All', 'GHG', 'All Aerosol', 'ATM Aerosol', 'OCN Aerosol']

        res1 = var2_reg - var1_reg
        res2 = var5_reg - var1_reg
        res3 = var2_reg - var5_reg
        res4 = (var2_reg + var3_reg - var4_reg - var5_reg)/2
        res5 = (var4_reg + var2_reg - var5_reg - var3_reg)/2

        plot_data = [res1, res2, res3, res4, res5]

        ylabel = 'Changes in % of total precipitation [%]'
        yticks = np.arange(-6, 7, 2)

        fname = 'vrseasia_aerosol_amip_jja_'+str(thres_bot)+'-'+str(thres_top)+'mm_prect_contour_percentile_fractions_changes_' + \
            reg_names[ireg]+'_box_absolute_response_'+str(inimonth)+"to"+str(endmonth)
        title = 'Changes in '+str(thres_bot)+'mm to '+str(thres_top)+'mm precip as % of total precip over ' + reg_names[ireg]
        plot_box(plot_data, labels, yticks, ylabel, title, outdir_thres+fname)
