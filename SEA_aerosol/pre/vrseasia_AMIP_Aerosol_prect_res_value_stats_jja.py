# This script is used to read daily data from Aerosol AMIP runs and investigate the add-on value of high resolution

# S1-read daily data from model and obs
# S2-plot contours
#
# by Harry Li

# import libraries
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import netCDF4 as nc
from netCDF4 import Dataset
import datetime as datetime
import calendar
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')


# set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/pre/res_comparison/"

# set up variable names and file name
varname = 'PRECT'
varstr = "Total Precip"
var_res = "fv02"


# define inital year and end year
iniyear = 2
endyear = 50
obsiniyear = 1950
obsendyear = 1959

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# define regions
reg_names = ['mainland SEA', 'Central India', 'South Asia', 'Western Ghats', 'South India', 'North India']
reg_str = ['mainSEA', 'ctInd', 'SA', 'WG', 'stInd', 'nrInd']
reg_lats = [[10, 20], [16.5, 26.5], [10, 35], [10, 19], [8, 20], [20, 28]]
reg_lons = [[100, 110], [74.5, 86.5], [70, 90], [72, 76], [70, 90], [65, 90]]

# month series
month = np.arange(1, 13, 1)
mname = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

################################################################################################
# S0-Define functions
################################################################################################


def get_stats(dataset, var, time, lats, lons, reg_lats, reg_lons):

    datasets = ['vrseasia', 'fv1.9x2.5', 'APHRODITE', 'CRU', 'GPCC']

    if dataset not in datasets:
        return ValueError('Dataset can not found!')

    latli = np.abs(lats - reg_lats[0]).argmin()
    latui = np.abs(lats - reg_lats[1]).argmin()

    lonli = np.abs(lons - reg_lons[0]).argmin()
    lonui = np.abs(lons - reg_lons[1]).argmin()

    var = np.nanmean(np.nanmean(var[:, latli: latui + 1, lonli: lonui + 1], axis=2), axis=1)

    if (dataset == 'vrseasia') | (dataset == 'fv1.9x2.5'):
        var_ts = []
        for iyear in range(endyear-iniyear+1):
            var_ts.append(np.nanmean(var[iyear*92: iyear*92+30]))
            var_ts.append(np.nanmean(var[iyear*92+30: iyear*92+61]))
            var_ts.append(np.nanmean(var[iyear*92+61: iyear*92+92]))
        var_ts = np.array(var_ts)
    elif (dataset == 'APHRODITE'):
        var_ts = []
        for iyear in range(int(len(var)/92)):
            var_ts.append(np.nanmean(var[iyear*92: iyear*92+30]))
            var_ts.append(np.nanmean(var[iyear*92+30: iyear*92+61]))
            var_ts.append(np.nanmean(var[iyear*92+61: iyear*92+92]))
        var_ts = np.array(var_ts)
    else:
        var_ts = var

    res_mean = np.mean(var_ts)
    res_std = np.std(var_ts)

    return res_mean, res_std


def get_ext(dataset, var, time, lats, lons, reg_lats, reg_lons):

    datasets = ['vrseasia', 'fv1.9x2.5', 'APHRODITE']

    if dataset not in datasets:
        return ValueError('Dataset can not found!')

    latli = np.abs(lats - reg_lats[0]).argmin()
    latui = np.abs(lats - reg_lats[1]).argmin()

    lonli = np.abs(lons - reg_lons[0]).argmin()
    lonui = np.abs(lons - reg_lons[1]).argmin()

    var = np.percentile(var, 99, axis=0)

    res_mean = np.nanmean(var[latli: latui + 1, lonli: lonui + 1])
    res_std = np.nanstd(var[latli: latui + 1, lonli: lonui + 1])

    return res_mean, res_std


def readcesm(dataset, latbounds, lonbounds):

    select_months = [6, 7, 8]

    datasets = ['vrseasia', 'fv1.9x2.5']
    indirs = {'vrseasia': '/project/p/pjk/harryli/cesm1/vrcesm/archive/vrseasia_19501959_OBS/atm/',
              'fv1.9x2.5': '/project/p/pjk/harukih/aerosol_AGCM/archive/DATA_FROM_OTHER_MACHINES/19501959_OBS_clone1/atm/hist/',
              }
    reffnames = {'vrseasia': 'fv02_prect_vrseasia_19501959_OBS.cam.h1.0001-01-01-00000.nc',
                 'fv1.9x2.5': 'nccompress.19501959_OBS_clone1.cam.h1.0001-01-01-00000.nc',
                 }
    fnames = {'vrseasia': 'fv02_prect_vrseasia_19501959_OBS.cam.h1.00*',
              'fv1.9x2.5': 'nccompress.19501959_OBS_clone1.cam.h1.00*',
              }

    # setup for land fraction
    dir_lndfrc = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
    flndfrc = {'vrseasia': 'USGS-gtopo30_0.23x0.31_remap_c061107.nc',
               'fv1.9x2.5': 'USGS-gtopo30_1.9x2.5_remap_c050602.nc',
               }

    if dataset in datasets:
        indir = indirs[dataset]
        reffname = reffnames[dataset]
        fname = fnames[dataset]
    else:
        return ValueError('Dataset can not found!')

    print('Start to reading dataset: '+dataset+'...')

    refdata = Dataset(indir+reffname)

    # read lat/lon grids
    lats = refdata.variables['lat'][:]
    lons = refdata.variables['lon'][:]

    latli = np.abs(lats - latbounds[0]).argmin()
    latui = np.abs(lats - latbounds[1]).argmin()

    lonli = np.abs(lons - lonbounds[0]).argmin()
    lonui = np.abs(lons - lonbounds[1]).argmin()

    lats = lats[latli: latui + 1]
    lons = lons[lonli: lonui + 1]

    ncdata = nc.MFDataset(indir+fname)
    time_var = ncdata.variables['time']
    cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    # print(cftime)
    # print(type(cftime))
    # dtime = [i.strftime("%Y-%m-%d %H:%M:%S") for i in cftime]
    dtime = []
    select_dtime = []
    for istr in cftime:
        temp = istr.strftime('%Y-%m-%d %H:%M:%S')
        if np.in1d(istr.month, select_months) & (~((istr.month == 2) & (istr.day == 29))) & (istr.year >= iniyear) & (istr.year <= endyear):
            select_dtime.append(True)
            temp = temp.lstrip(' ')
            if len(temp) == 16:
                temp = '000' + temp
            else:
                temp = '00' + temp
            dtime.append(temp)
        else:
            select_dtime.append(False)
    select_dtime = np.array(select_dtime, dtype=bool)
    # print(select_dtime)
    # print(dtime)
    # print(dtime)
    # dtime = pd.to_datetime(dtime, format='%Y-%m-%d %H:%M:%S')

    # select_dtime = (np.in1d(dtime.month, select_months)) & (~((dtime.month == 2) & (dtime.day == 29)))
    # dtime = dtime[select_dtime]

    if dataset == 'vrseasia':
        var = ncdata.variables['PRECT'][select_dtime, latli: latui + 1, lonli: lonui + 1]
    if dataset == 'fv1.9x2.5':
        var = ncdata.variables['PRECL'][select_dtime, latli: latui + 1, lonli: lonui + 1] + ncdata.variables['PRECC'][select_dtime, latli: latui + 1, lonli: lonui + 1]

    var = var * 86400 * 1000

    dataset_lndfrc = nc.Dataset(dir_lndfrc+flndfrc[dataset])
    lndfrc = dataset_lndfrc.variables['LANDFRAC'][latli: latui + 1, lonli: lonui + 1]
    var = np.ma.array(var)
    for idx in range(len(dtime)):
        var[idx, :, :] = np.ma.masked_where(lndfrc < 0.5, var[idx, :, :])

    # print(var.mask)

    var[var.mask] = np.nan

    return var, dtime, lats, lons


def readobs(dataset, latbounds, lonbounds):

    print('Start to reading dataset: '+dataset+'...')

    datasets = ['APHRODITE', 'CRU', 'GPCC']
    select_months = [6, 7, 8]

    indirs = {'APHRODITE': '/scratch/d/dylan/harryli/obsdataset/APHRODITE/MA/',
              'CRU': '/scratch/d/dylan/harryli/obsdataset/CRU/pre/',
              'GPCC': '/scratch/d/dylan/harryli/obsdataset/GPCC/',
              }
    fnames = {'APHRODITE': 'APHRO_MA_025deg_V1101.1951-2007.nc',
              'CRU': 'cru_ts4.03.1901.2018.pre.dat.nc',
              'GPCC': 'precip.mon.total.v7.nc',
              }
    varnames = {'APHRODITE': 'precip',
                'CRU': 'pre',
                'GPCC': 'precip',
                }
    latnames = {'APHRODITE': 'latitude',
                'CRU': 'lat',
                'GPCC': 'lat',
                }
    lonnames = {'APHRODITE': 'longitude',
                'CRU': 'lon',
                'GPCC': 'lon',
                }

    if dataset in datasets:
        indir = indirs[dataset]
        fname = fnames[dataset]
        varname = varnames[dataset]
    else:
        return ValueError('Dataset can not found!')

    ncdata = Dataset(indir+fname)

    # read lat/lon grids
    lats = ncdata.variables[latnames[dataset]][:]
    lons = ncdata.variables[lonnames[dataset]][:]

    latli = np.abs(lats - latbounds[0]).argmin()
    latui = np.abs(lats - latbounds[1]).argmin()

    lonli = np.abs(lons - lonbounds[0]).argmin()
    lonui = np.abs(lons - lonbounds[1]).argmin()

    lat_swap = False
    lon_swap = False

    if latli > latui:
        lat_swap = True
        temp = latli
        latli = latui
        latui = temp
    if lonli > lonui:
        lon_swap = True
        temp = lonli
        lonli = lonui
        lonui = temp

    lats = lats[latli: latui + 1]
    lons = lons[lonli: lonui + 1]

    time_var = ncdata.variables['time']
    if dataset == 'GPCC':
        cftime = nc.num2date(time_var[:], units=time_var.units)
    else:
        cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    dtime = [i.strftime("%Y-%m-%d %H:%M:%S") for i in cftime]
    dtime = pd.to_datetime(dtime, format='%Y-%m-%d %H:%M:%S')
    select_dtime = (np.in1d(dtime.month, select_months)) & (~((dtime.month == 2) & (dtime.day == 29))) & (dtime.year >= obsiniyear) & (dtime.year <= obsendyear)
    dtime = dtime[select_dtime]

    var = ncdata.variables[varname][select_dtime, latli: latui + 1, lonli: lonui + 1]

    if (dataset == 'CRU') | (dataset == 'GPCC'):
        for idx in np.arange(len(dtime)):
            var[idx, :, :] = var[idx, :, :] / calendar.monthrange(dtime[idx].year, dtime[idx].month)[1]

    if lat_swap:
        var = var[:, ::-1, :]
        lats = lats[::-1]

    if lon_swap:
        var = var[:, :, ::-1]
        lons = lons[::-1]

    var[var < 0] = np.nan

    return var, dtime, lats, lons


def plotcontour(dataset, var, lats, lons, clevs, colormap, latbounds, lonbounds, fname):

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111)
    # ax.set_title(dataset, fontsize=5, pad=5)

    # create basemap
    map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                  llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()

    # draw lat/lon lines
    parallels = np.arange(latbounds[0], latbounds[1], 20)
    meridians = np.arange(lonbounds[0], lonbounds[1], 20)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)

    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)

    # plot the contour
    cs = map.contourf(x, y, var, clevs, cmap=colormap, alpha=0.9, extend="both")

    # set x/y tick label size
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)

    fig.subplots_adjust(bottom=0.23, wspace=0.2, hspace=0.2)
    cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('[mm/day]', fontsize=5, labelpad=-0.5)

    # add title
    plt.savefig(fname+'.png', bbox_inches='tight', dpi=600)
    plt.suptitle(dataset, fontsize=7, y=0.95)

    # save figure
    plt.savefig(fname+'.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)

    return

################################################################################################
# S1- read daily data from model and obs
################################################################################################


# read vrcesm
vrdata, vrtime, vrlats, vrlons = readcesm('vrseasia', latbounds, lonbounds)
# print(vrtime)
# print(vrlats)
# print(vrlons)
# print(vrdata.shape)

fvdata, fvtime, fvlats, fvlons = readcesm('fv1.9x2.5', latbounds, lonbounds)
# print(fvtime)
# print(fvdata.shape)

aphdata, aphtime, aphlats, aphlons = readobs('APHRODITE', latbounds, lonbounds)
# print(aphtime)
# print(aphdata.shape)

crudata, crutime, crulats, crulons = readobs('CRU', latbounds, lonbounds)
# print(crutime)
# print(crudata.shape)

gpcdata, gpctime, gpclats, gpclons = readobs('GPCC', latbounds, lonbounds)
# print(gpctime)
# print(gpclats)
# print(gpcdata.shape)

################################################################################################
# S2-Calculate Regional Mean
################################################################################################


df = {'type': np.array(['vr-mean', 'vr-std', 'fv-mean', 'fv-std', 'aph-mean', 'aph-std', 'cru-mean', 'cru-std', 'gpcc-mean', 'gpcc-std'])}


for ireg in range(len(reg_names)):
    reg_name = reg_names[ireg]

    temp = []
    mean, std = get_stats('vrseasia', vrdata, vrtime, vrlats, vrlons, reg_lats[ireg], reg_lons[ireg])
    temp.append(mean)
    temp.append(std)

    mean, std = get_stats('fv1.9x2.5', fvdata, fvtime, fvlats, fvlons, reg_lats[ireg], reg_lons[ireg])
    temp.append(mean)
    temp.append(std)

    mean, std = get_stats('APHRODITE', aphdata, aphtime, aphlats, aphlons, reg_lats[ireg], reg_lons[ireg])
    temp.append(mean)
    temp.append(std)

    mean, std = get_stats('CRU', crudata, crutime, crulats, crulons, reg_lats[ireg], reg_lons[ireg])
    temp.append(mean)
    temp.append(std)

    mean, std = get_stats('GPCC', gpcdata, gpctime, gpclats, gpclons, reg_lats[ireg], reg_lons[ireg])
    temp.append(mean)
    temp.append(std)

    df[reg_name] = np.array(temp)

print(df)
df = pd.DataFrame(df)
df.set_index('type', inplace=True)
df.to_csv(outdir+'vrseasia_aerosol_amip_jja_regional_monthly_mean.csv', sep=',')


################################################################################################
# S3-Plot contour map
################################################################################################
clevs = np.arange(0, 20, 2)
colormap = cm.Spectral_r

fname = outdir+'vrseasia_aerosol_amip_jja_prect_MA_contour_mean_vrseasia'
plotcontour('vrseasia', np.nanmean(vrdata, axis=0), vrlats, vrlons, clevs, colormap, latbounds, lonbounds, fname)

fname = outdir+'vrseasia_aerosol_amip_jja_prect_MA_contour_mean_fv19x25'
plotcontour('fv1.9x2.5', np.nanmean(fvdata, axis=0), fvlats, fvlons, clevs, colormap, latbounds, lonbounds, fname)

fname = outdir+'vrseasia_aerosol_amip_jja_prect_MA_contour_mean_aphrodite'
plotcontour('APHRODITE', np.nanmean(aphdata, axis=0), aphlats, aphlons, clevs, colormap, latbounds, lonbounds, fname)

fname = outdir+'vrseasia_aerosol_amip_jja_prect_MA_contour_mean_cru'
plotcontour('CRU', np.nanmean(crudata, axis=0), crulats, crulons, clevs, colormap, latbounds, lonbounds, fname)

fname = outdir+'vrseasia_aerosol_amip_jja_prect_MA_contour_mean_gpcc'
plotcontour('GPCC', np.nanmean(gpcdata, axis=0), gpclats, gpclons, clevs, colormap, latbounds, lonbounds, fname)


################################################################################################
# S4--Calculate Regional Mean for 99th precip
################################################################################################

df = {'type': np.array(['vr-mean', 'vr-std', 'fv-mean', 'fv-std', 'aph-mean', 'aph-std'])}


for ireg in range(len(reg_names)):
    reg_name = reg_names[ireg]

    temp = []
    mean, std = get_ext('vrseasia', vrdata, vrtime, vrlats, vrlons, reg_lats[ireg], reg_lons[ireg])
    temp.append(mean)
    temp.append(std)

    mean, std = get_ext('fv1.9x2.5', fvdata, fvtime, fvlats, fvlons, reg_lats[ireg], reg_lons[ireg])
    temp.append(mean)
    temp.append(std)

    mean, std = get_ext('APHRODITE', aphdata, aphtime, aphlats, aphlons, reg_lats[ireg], reg_lons[ireg])
    temp.append(mean)
    temp.append(std)

    df[reg_name] = np.array(temp)

print(df)
df = pd.DataFrame(df)
df.set_index('type', inplace=True)
df.to_csv(outdir+'vrseasia_aerosol_amip_jja_regional_monthly_99th_mean.csv', sep=',')


################################################################################################
# S3-Plot contour map
################################################################################################
clevs = np.arange(10, 60, 5)
colormap = cm.Spectral_r

fname = outdir+'vrseasia_aerosol_amip_jja_prect_MA_contour_99th_mean_vrseasia'
plotcontour('vrseasia', np.percentile(vrdata, 99, axis=0), vrlats, vrlons, clevs, colormap, latbounds, lonbounds, fname)

fname = outdir+'vrseasia_aerosol_amip_jja_prect_MA_contour_99th_mean_fv19x25'
plotcontour('fv1.9x2.5', np.percentile(fvdata, 99, axis=0), fvlats, fvlons, clevs, colormap, latbounds, lonbounds, fname)

fname = outdir+'vrseasia_aerosol_amip_jja_prect_MA_contour_99th_mean_aphrodite'
plotcontour('APHRODITE', np.percentile(aphdata, 99, axis=0), aphlats, aphlons, clevs, colormap, latbounds, lonbounds, fname)
