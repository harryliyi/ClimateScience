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
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/pre/extreme/prect_extreme_changes/JJA/15yr_test/"

# set up variable names and file name
varname = 'PRECT'

varstr = "Total Precip"
var_res = "fv09"
varfname = 'prect'
var_unit = 'mm/day'

# define inital year and end year
iniyear = 2
endyear = 15

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# define regions
reg_names = ['mainland SEA', 'Central Indian']
reg_str = ['mainSEA', 'ctInd']
reg_lats = [[10, 20], [16.5, 26.5]]
reg_lons = [[100, 110], [74.5, 86.5]]


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
# percentile_ranges = [50, 70, 80, 90, 95, 97, 99, 99.5, 99.9]
percentile_ranges = [99]

# set up nbins
nbins = 50

outdir_stats = outdir


################################################################################################
# S0-Define functions
################################################################################################

def plot_box(plot_data, labels, ylabel, title, fname):

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
    ax.axhline(y=0., linewidth=2, c='black', linestyle='solid')
    xticks = np.arange(len(box_data))+1
    xticknames = labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticknames, fontsize=8)
    ax.tick_params(axis='y', which='major', labelsize=6)

    ax.set_ylabel(ylabel, fontsize=8)

    # add title
    plt.savefig(fname+'.png', dpi=600, bbox_inches='tight')
    plt.suptitle(title, fontsize=8, y=0.95)
    plt.savefig(fname+'.pdf', bbox_inches='tight')
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

for idx in range((endyear-iniyear+1)*ndays):
    var1[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var1[idx, :, :])
    var2[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var2[idx, :, :])
    var3[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var3[idx, :, :])
    var4[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var4[idx, :, :])
    var5[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var5[idx, :, :])

################################################################################################
# S2-calculate for extremes
################################################################################################

for ipercent in range(len(percentile_ranges)):
    percentile = percentile_ranges[ipercent]
    outdir_percent = outdir + str(percentile)+'th/'
    print('Plotting for '+str(percentile)+'th...')

    var1_extreme = np.percentile(var1, percentile, axis=0)
    var2_extreme = np.percentile(var2, percentile, axis=0)
    var3_extreme = np.percentile(var3, percentile, axis=0)
    var4_extreme = np.percentile(var4, percentile, axis=0)
    var5_extreme = np.percentile(var5, percentile, axis=0)

    for ireg in range(len(reg_names)):
        print('Plotting for region: '+reg_names[ireg])
        reg_latli = np.abs(lats - reg_lats[ireg][0]).argmin()
        reg_latui = np.abs(lats - reg_lats[ireg][1]).argmin()

        reg_lonli = np.abs(lons - reg_lons[ireg][0]).argmin()
        reg_lonui = np.abs(lons - reg_lons[ireg][1]).argmin()

        # print(lats[reg_latli])
        # print(lats[reg_latui])
        # print(lons[reg_lonli])
        # print(lons[reg_lonui])

        var1_reg = var1_extreme[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var2_reg = var2_extreme[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var3_reg = var3_extreme[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var4_reg = var4_extreme[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]
        var5_reg = var5_extreme[reg_latli: reg_latui+1, reg_lonli: reg_lonui+1]

        # plot for each cases
        plot_data = [var1_reg, var2_reg, var3_reg, var4_reg, var5_reg]

        labels = [r'$S_{CTRL}$', r'$S_{2000}A_{2000}$', r'$S_{PERT}A_{2000}$', r'$S_{2000}A_{1950}$', r'$S_{PERT}A_{1950}$']
        ylabel = 'Distribution of '+str(percentile)+'th Precipitation [mm/day]'

        fname = var_res + '_vrseasia_aerosol_amip_jja_'+varfname+'_'+str(percentile)+'_th_extremes_'+reg_names[ireg]+'_box_exps_'+str(inimonth)+"to"+str(endmonth)
        title = var_res + ' Aerosol experiments '+varstr+' Rp'+str(percentile)+' over '+reg_names[ireg]
        plot_box(plot_data, labels, ylabel, title, outdir_percent+fname)

        # plot for absolute response
        labels = ['All', 'GHG and Natural', 'All Aerosol', 'Aerosol Fast', 'Aerosol Slow']

        res1 = var2_reg - var1_reg
        res2 = var5_reg - var1_reg
        res3 = var2_reg - var5_reg
        res4 = (var2_reg + var3_reg - var4_reg - var5_reg)/2
        res5 = (var4_reg + var2_reg - var5_reg - var3_reg)/2

        plot_data = [res1, res2, res3, res4, res5]

        ylabel = 'Changes in Distribution of '+str(percentile)+'th Precipitation [mm/day]'

        fname = var_res + '_vrseasia_aerosol_amip_jja_'+varfname+'_'+str(percentile)+'_th_extremes_'+reg_names[ireg]+'_box_absolute_response_'+str(inimonth)+"to"+str(endmonth)
        title = var_res + ' Changes in '+varstr+' Rp'+str(percentile)+' over '+reg_names[ireg]
        plot_box(plot_data, labels, ylabel, title, outdir_percent+fname)

        # plot for absolute response
        res1 = (var2_reg - var1_reg)/var1_reg*100
        res2 = (var5_reg - var1_reg)/var1_reg*100
        res3 = (var2_reg - var5_reg)/var5_reg*100
        res4 = (var2_reg + var3_reg - var4_reg - var5_reg)/(var4_reg + var5_reg)*100
        res5 = (var4_reg + var2_reg - var5_reg - var3_reg)/(var5_reg + var3_reg)*100

        plot_data = [res1, res2, res3, res4, res5]

        ylabel = 'Relative Changes in Distribution of '+str(percentile)+'th Precipitation [%]'

        fname = var_res + '_vrseasia_aerosol_amip_jja_'+varfname+'_'+str(percentile)+'_th_extremes_'+reg_names[ireg]+'_box_relative_response_'+str(inimonth)+"to"+str(endmonth)
        title = var_res + ' Relative Changes in '+varstr+' Rp'+str(percentile)+' over '+reg_names[ireg]
        plot_box(plot_data, labels, ylabel, title, outdir_percent+fname)
