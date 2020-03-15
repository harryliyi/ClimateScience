# this script is used to compare vrcesm against observations
# here extremes is presented
# by Harry Li


# import libraries
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# set up land mask directory
rdir = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"

# set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/pre/res_comparison/"

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
# reg_names = ['mainland SEA', 'Central India']
# reg_str = ['mainSEA', 'ctInd']
# reg_lats = [[10, 20], [16.5, 26.5]]
# reg_lons = [[100, 110], [74.5, 86.5]]

reg_names = ['mainland SEA', 'Central India', 'South Asia', 'Western Ghats', 'South India', 'North India']
reg_str = ['mainSEA', 'ctInd', 'SA', 'WG', 'stInd', 'nrInd']
reg_lats = [[10, 20], [16.5, 26.5], [10, 35], [10, 19], [8, 20], [20, 28]]
reg_lons = [[100, 110], [74.5, 86.5], [70, 90], [72, 76], [70, 90], [65, 90]]

# set up moderate thresholds
mod_thres = [[5, 30], [5, 60], [5, 50], [5, 50], [5, 40], [5, 60]]


# month series
month = np.arange(1, 13, 1)
mname = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# define the season period
inimonth = 6
endmonth = 8

select_months = [6, 7, 8]

iniday = np.sum(mdays[0:inimonth-1])
endday = np.sum(mdays[0:endmonth])
# print(iniday)
# print(endday)
ndays = endday-iniday

# set up percentile
percentile_ranges = [50, 70, 80, 90, 95, 97, 99, 99.5, 99.9]
# percentile_ranges = [99]


# set up nbins
nbins = 50

outdir_stats = outdir


################################################################################################
# S0-Define functions
################################################################################################

def plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, fname):

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
    if (len(labels) == 3):
        ax.axhline(y=0., linewidth=2, c='black', linestyle='solid')
    xticks = np.arange(len(box_data))+1
    xticknames = xticklabels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticknames, fontsize=8)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticks, fontsize=8)
    ax.tick_params(axis='y', which='major', labelsize=6)

    ax.set_ylabel(ylabel, fontsize=8)

    fig.subplots_adjust(bottom=0.08, wspace=0.2)

    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.1))
    ax2.spines["bottom"].set_visible(True)
    ax2.set_frame_on(False)
    ax2.patch.set_visible(False)
    if (len(labels) == 3):
        ax2.set_xticks([.18, .5, .82])
    if (len(labels) == 4):
        ax2.set_xticks([.11, .38, .62, .89])
    ax2.set_xticklabels(labels, fontsize=8)

    # add title
    plt.savefig(fname+'.png', dpi=600, bbox_inches='tight')
    plt.suptitle(title, fontsize=8, y=0.95)
    plt.savefig(fname+'.pdf', bbox_inches='tight')

    plt.close(fig)


def plot_bars(plot_data, labels, xticklabels, title, fname):

    mean_data = []
    std_data = []
    for idata in range(len(plot_data)):
        temp_data = plot_data[idata]
        temp_data = temp_data.flatten()
        temp_data = temp_data[~np.isnan(temp_data)]
        mean_data.append(np.mean(temp_data))
        std_data.append(np.std(temp_data))

    # print(len(box_data[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xticks = np.arange(len(mean_data))+1
    ax.bar(xticks, mean_data, yerr=std_data)

    xticknames = xticklabels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticknames, fontsize=8)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticks, fontsize=8)
    ax.tick_params(axis='y', which='major', labelsize=6)

    # ax.set_ylabel(fontsize=8)

    fig.subplots_adjust(bottom=0.08, wspace=0.2)

    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.1))
    ax2.spines["bottom"].set_visible(True)
    ax2.set_frame_on(False)
    ax2.patch.set_visible(False)
    if (len(labels) == 3):
        ax2.set_xticks([.18, .5, .82])
    if (len(labels) == 4):
        ax2.set_xticks([.11, .38, .62, .89])
    ax2.set_xticklabels(labels, fontsize=8)

    # add title
    plt.savefig(fname+'.png', dpi=600, bbox_inches='tight')
    plt.suptitle(title, fontsize=8, y=0.95)
    plt.savefig(fname+'.pdf', bbox_inches='tight')

    plt.close(fig)


def plot_hist(plot_data, labels, colors, linetypes, title, fname):

    # print(len(box_data[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for idata in range(len(plot_data)):
        temp_data = plot_data[idata]
        temp_data = temp_data.flatten()
        temp_data = temp_data[~np.isnan(temp_data)]
        hist, bin_edges = np.histogram(temp_data, bins=nbins, density=True)
        bin_mids = []
        for ii in range(nbins):
            bin_mids.append((bin_edges[1]-bin_edges[0])/2+bin_edges[ii])
        ax.plot(bin_mids, hist, color=colors[idata], linestyle=linetypes[idata], label=labels[idata])

    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticks, fontsize=8)
    ax.tick_params(axis='y', which='major', labelsize=6)
    ax.set_yscale('log')

    plt.legend(handlelength=4, fontsize=5)

    # add title
    plt.savefig(fname+'.png', dpi=600, bbox_inches='tight')
    plt.suptitle(title, fontsize=8, y=0.95)
    plt.savefig(fname+'.pdf', bbox_inches='tight')

    plt.close(fig)


################################################################################################
# S1-open daily data from vrcesm
################################################################################################

# set up data directories and filenames
case2 = "vrseasia_20002010_OBS"
case3 = "vrseasia_20002009_OBS_SUBAERSST_CESM1CAM5_SST"
case4 = "vrseasia_20002009_OBS_AEREMIS1950"
case5 = "vrseasia_20002009_OBS_AEREMIS1950_SUBAERSST_CESM1CAM5_SST"

expdir2 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case2+"/atm/"
expdir3 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case3+"/atm/"
expdir4 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case4+"/atm/"
expdir5 = "/project/p/pjk/harryli/cesm1/vrcesm/archive/"+case5+"/atm/"

# open data and read grids
fname1 = var_res+"_PREC_"+case2+".cam.h0.0001-0050.nc"
fdata1 = Dataset(expdir2+fname1)

# read lat/lon grids
vrlats = fdata1.variables['lat'][:]
vrlons = fdata1.variables['lon'][:]

# latitude/longitude  lower and upper contour index
vrlatli = np.abs(vrlats - latbounds[0]).argmin()
vrlatui = np.abs(vrlats - latbounds[1]).argmin()

vrlonli = np.abs(vrlons - lonbounds[0]).argmin()
vrlonui = np.abs(vrlons - lonbounds[1]).argmin()

vrlats = vrlats[vrlatli:vrlatui+1]
vrlons = vrlons[vrlonli:vrlonui+1]

nvrlats = vrlatui - vrlatli + 1
nvrlons = vrlonui - vrlonli + 1

print(nvrlats)
print(nvrlons)

# read land mask
dir_lndfrc = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
flndfrc = 'USGS-gtopo30_0.23x0.31_remap_c061107.nc'
dataset_lndfrc = Dataset(dir_lndfrc+flndfrc)
vrlndfrc = dataset_lndfrc.variables['LANDFRAC'][vrlatli:vrlatui+1, vrlonli:vrlonui+1]


vrdata2 = np.zeros(((endyear-iniyear+1)*ndays, nvrlats, nvrlons))
vrdata3 = np.zeros(((endyear-iniyear+1)*ndays, nvrlats, nvrlons))
vrdata4 = np.zeros(((endyear-iniyear+1)*ndays, nvrlats, nvrlons))
vrdata5 = np.zeros(((endyear-iniyear+1)*ndays, nvrlats, nvrlons))

print('reading the vrseasia data...')
for iyear in np.arange(iniyear, endyear+1, 1):
    if (iyear < 10):
        yearno = '000'+str(iyear)
    else:
        yearno = '00'+str(iyear)
    print('Current year is: '+yearno)

    fname2 = var_res+'_prect_'+case2+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname3 = var_res+'_prect_'+case3+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname4 = var_res+'_prect_'+case4+'.cam.h1.'+yearno+'-01-01-00000.nc'
    fname5 = var_res+'_prect_'+case5+'.cam.h1.'+yearno+'-01-01-00000.nc'

    fdata2 = Dataset(expdir2+fname2)
    fdata3 = Dataset(expdir3+fname3)
    fdata4 = Dataset(expdir4+fname4)
    fdata5 = Dataset(expdir5+fname5)

    temp2 = fdata2.variables[varname][iniday: endday, vrlatli:vrlatui+1, vrlonli:vrlonui+1] * 86400 * 1000
    temp3 = fdata3.variables[varname][iniday: endday, vrlatli:vrlatui+1, vrlonli:vrlonui+1] * 86400 * 1000
    temp4 = fdata4.variables[varname][iniday: endday, vrlatli:vrlatui+1, vrlonli:vrlonui+1] * 86400 * 1000
    temp5 = fdata5.variables[varname][iniday: endday, vrlatli:vrlatui+1, vrlonli:vrlonui+1] * 86400 * 1000

    vrdata2[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp2.copy()
    vrdata3[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp3.copy()
    vrdata4[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp4.copy()
    vrdata5[(iyear-iniyear)*ndays:(iyear-iniyear+1)*ndays, :, :] = temp5.copy()

# vrdata2 = np.ma.array(vrdata2)
# vrdata3 = np.ma.array(vrdata3)
# vrdata4 = np.ma.array(vrdata4)
# vrdata5 = np.ma.array(vrdata5)
#
# for idx in range((endyear-iniyear+1)*ndays):
#     vrdata2[idx, :, :] = np.ma.masked_where(vrlndfrc < 0.5, vrdata2[idx, :, :])
#     vrdata3[idx, :, :] = np.ma.masked_where(vrlndfrc < 0.5, vrdata3[idx, :, :])
#     vrdata4[idx, :, :] = np.ma.masked_where(vrlndfrc < 0.5, vrdata4[idx, :, :])
#     vrdata5[idx, :, :] = np.ma.masked_where(vrlndfrc < 0.5, vrdata5[idx, :, :])
#
# vrdata2[vrdata2.mask] = np.nan
# vrdata3[vrdata2.mask] = np.nan
# vrdata4[vrdata2.mask] = np.nan
# vrdata5[vrdata2.mask] = np.nan

################################################################################################
# S2-open daily data from fv0.9x1.25
################################################################################################


print('reading the fv0.9x1.25 data...')

###########################################################################
# read fv S2000A2000
print('reading S2000A2000...')

expdir = "/project/p/pjk/harukih/aerosol_AGCM/archive/20002010_OBS_niagara_clone1/atm/"
fname1 = '20002010_OBS_niagara_clone1.cam.h1.y1-32.PRECC.nc'
fname2 = '20002010_OBS_niagara_clone1.cam.h1.y1-32.PRECL.nc'

fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)

# read lat/lon grids
fvlats = fdata1.variables['lat'][:]
fvlons = fdata1.variables['lon'][:]

# latitude/longitude  lower and upper contour index
fvlatli = np.abs(fvlats - latbounds[0]).argmin()
fvlatui = np.abs(fvlats - latbounds[1]).argmin()

fvlonli = np.abs(fvlons - lonbounds[0]).argmin()
fvlonui = np.abs(fvlons - lonbounds[1]).argmin()

fvlats = fvlats[fvlatli:fvlatui+1]
fvlons = fvlons[fvlonli:fvlonui+1]

nfvlats = fvlatui - fvlatli + 1
nfvlons = fvlonui - fvlonli + 1

# read land mask
dir_lndfrc = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/topo/"
flndfrc = 'USGS-gtopo30_0.9x1.25_remap_c051027.nc'
dataset_lndfrc = Dataset(dir_lndfrc+flndfrc)
fvlndfrc = dataset_lndfrc.variables['LANDFRAC'][fvlatli:fvlatui+1, fvlonli:fvlonui+1]


expdir = "/project/p/pjk/harukih/aerosol_AGCM/archive/20002010_OBS_niagara_clone1/atm/"
fname1 = '20002010_OBS_niagara_clone1.cam.h1.y1-32.PRECC.nc'
fname2 = '20002010_OBS_niagara_clone1.cam.h1.y1-32.PRECL.nc'
fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)
fvdata2 = fdata1.variables['PRECC'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1] + \
    fdata2.variables['PRECL'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1]
time_var = fdata1.variables['time']
cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
# print(fvdata2.shape)
# print(len(cftime))

expdir = "/project/p/pjk/harukih/aerosol_AGCM/archive/20002010_OBS_niagara_clone2/atm/"
fname1 = 'daily.20002010_OBS_niagara_clone2.cam.h1.y1-4.PRECC.nc'
fname2 = 'daily.20002010_OBS_niagara_clone2.cam.h1.y1-4.PRECL.nc'
fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)
tempdata = fdata1.variables['PRECC'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1] + \
    fdata2.variables['PRECL'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1]
time_var = fdata1.variables['time']
temptime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

fvdata2 = np.concatenate((fvdata2, tempdata), axis=0)
cftime = np.append(cftime, temptime)

fname1 = '20002010_OBS_niagara_clone2.cam.h1.y6-24.PRECC.nc'
fname2 = '20002010_OBS_niagara_clone2.cam.h1.y6-24.PRECL.nc'
fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)
tempdata = fdata1.variables['PRECC'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1] + \
    fdata2.variables['PRECL'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1]
time_var = fdata1.variables['time']
temptime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

fvdata2 = np.concatenate((fvdata2, tempdata), axis=0)
cftime = np.append(cftime, temptime)

# print(cftime)
# print(len(cftime))
# print(fvdata2.shape)

select_dtime = []
for istr in cftime:
    temp = istr.strftime('%Y-%m-%d %H:%M:%S')
    if np.in1d(istr.month, select_months) & (~((istr.month == 2) & (istr.day == 29))):
        select_dtime.append(True)
    else:
        select_dtime.append(False)
select_dtime = np.array(select_dtime, dtype=bool)
fvtime2 = cftime[select_dtime]
fvdata2 = fvdata2[select_dtime, :, :]
fvdata2 = fvdata2 * 86400 * 1000
fvnyears2 = int(len(fvtime2)/92)

# fvdata2 = np.ma.array(fvdata2)
#
# for idx in range(len(fvtime2)):
#     fvdata2[idx, :, :] = np.ma.masked_where(fvlndfrc < 0.5, fvdata2[idx, :, :])

# print(fvdata2[0, :, :])
# print(fvnyears2)
print(fvdata2.shape)
print(fvtime2)

###########################################################################
# read fv SpertA2000
print('reading SpertA2000...')

expdir = "/project/p/pjk/harukih/aerosol_AGCM/archive/20002009_OBS_SUBAERSST_CESM1CAM5_SST/atm/"
fname1 = '20002009_OBS_SUBAERSST_CESM1CAM5_SST.cam.h1.y1-32.PRECC.nc'
fname2 = '20002009_OBS_SUBAERSST_CESM1CAM5_SST.cam.h1.y1-32.PRECL.nc'
fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)
fvdata3 = fdata1.variables['PRECC'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1] + \
    fdata2.variables['PRECL'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1]
time_var = fdata1.variables['time']
cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

fname1 = '20002009_OBS_SUBAERSST_CESM1CAM5_SST.cam.h1.y33-50.PRECC.nc'
fname2 = '20002009_OBS_SUBAERSST_CESM1CAM5_SST.cam.h1.y33-50.PRECL.nc'
fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)
tempdata = fdata1.variables['PRECC'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1] + \
    fdata2.variables['PRECL'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1]
time_var = fdata1.variables['time']
temptime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

# print(temptime[0:25])
temptime = temptime[:-2:24]
daydata = np.zeros((18*92, nfvlats, nfvlons))
count = 0
daytime = []
for idx in range(len(temptime)):
    if np.in1d(temptime[idx].month, select_months):
        daydata[count, :, :] = np.mean(tempdata[idx*24:idx*24+24, :, :], axis=0)
        count = count + 1
        daytime.append(temptime[idx])

# print(count)

fvdata3 = np.concatenate((fvdata3, daydata), axis=0)
cftime = np.append(cftime, daytime)

select_dtime = []
for istr in cftime:
    temp = istr.strftime('%Y-%m-%d %H:%M:%S')
    if np.in1d(istr.month, select_months) & (~((istr.month == 2) & (istr.day == 29))):
        select_dtime.append(True)
    else:
        select_dtime.append(False)
select_dtime = np.array(select_dtime, dtype=bool)
fvtime3 = cftime[select_dtime]
fvdata3 = fvdata3[select_dtime, :, :]
fvdata3 = fvdata3 * 86400 * 1000
fvnyears3 = int(len(fvtime3)/92)

# fvdata3 = np.ma.array(fvdata3)
#
# for idx in range(len(fvtime3)):
#     fvdata3[idx, :, :] = np.ma.masked_where(fvlndfrc < 0.5, fvdata3[idx, :, :])

# print(fvtime3)
# print(fvdata3[0, :, :])
print(fvdata3.shape)
print(fvtime2)

###########################################################################
# read fv S2000A1950
print('reading S2000A1950...')

expdir = "/project/p/pjk/harukih/aerosol_AGCM/archive/20002009_OBS_AEREMIS1950_niagara_clone1/atm/"
fname1 = '20002009_OBS_AEREMIS1950_niagara_clone1.cam.h1.y1-32.PRECC.nc'
fname2 = '20002009_OBS_AEREMIS1950_niagara_clone1.cam.h1.y1-32.PRECL.nc'
fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)

fvdata4 = fdata1.variables['PRECC'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1] + \
    fdata2.variables['PRECL'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1]
time_var = fdata1.variables['time']
cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

fname1 = 'daily.20002009_OBS_AEREMIS1950_niagara_clone1.cam.h1.y33-52.PRECC.nc'
fname2 = 'daily.20002009_OBS_AEREMIS1950_niagara_clone1.cam.h1.y33-52.PRECL.nc'
fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)
tempdata = fdata1.variables['PRECC'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1] + \
    fdata2.variables['PRECL'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1]
time_var = fdata1.variables['time']
temptime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

fvdata4 = np.concatenate((fvdata4, tempdata), axis=0)
cftime = np.append(cftime, temptime)

fname1 = 'daily.20002009_OBS_AEREMIS1950_niagara_clone1.cam.h1.y53-72.PRECC.nc'
fname2 = 'daily.20002009_OBS_AEREMIS1950_niagara_clone1.cam.h1.y53-72.PRECL.nc'
fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)
tempdata = fdata1.variables['PRECC'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1] + \
    fdata2.variables['PRECL'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1]
time_var = fdata1.variables['time']
temptime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

fvdata4 = np.concatenate((fvdata4, tempdata), axis=0)
cftime = np.append(cftime, temptime)
# print(cftime)

select_dtime = []
for istr in cftime:
    temp = istr.strftime('%Y-%m-%d %H:%M:%S')
    if np.in1d(istr.month, select_months) & (~((istr.month == 2) & (istr.day == 29))):
        select_dtime.append(True)
    else:
        select_dtime.append(False)
select_dtime = np.array(select_dtime, dtype=bool)
fvtime4 = cftime[select_dtime]
fvdata4 = fvdata4[select_dtime, :, :]
fvdata4 = fvdata4 * 86400 * 1000
fvnyears4 = int(len(fvtime4)/92)

# fvdata4 = np.ma.array(fvdata4)
# for idx in range(len(fvtime4)):
#     fvdata4[idx, :, :] = np.ma.masked_where(fvlndfrc < 0.5, fvdata4[idx, :, :])

# print(fvdata4[0, :, :])
print(fvdata4.shape)
print(fvtime4)

###########################################################################
# read fv SpertA1950
print('reading SpertA1950...')

expdir = "/project/p/pjk/harukih/aerosol_AGCM/archive/20002009_OBS_AEREMIS1950_SUBAERSST_CESMCAM5_SST_clone1/atm/"
fname1 = '20002009_OBS_AEREMIS1950_SUBAERSST_CESMCAM5_SST_clone1.cam.h1.y1-54.PRECC.nc'
fname2 = '20002009_OBS_AEREMIS1950_SUBAERSST_CESMCAM5_SST_clone1.cam.h1.y1-54.PRECL.nc'
fdata1 = Dataset(expdir+fname1)
fdata2 = Dataset(expdir+fname2)
fvdata5 = fdata1.variables['PRECC'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1] + \
    fdata2.variables['PRECL'][:, fvlatli:fvlatui+1, fvlonli:fvlonui+1]
time_var = fdata1.variables['time']
cftime = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

select_dtime = []
for istr in cftime:
    temp = istr.strftime('%Y-%m-%d %H:%M:%S')
    if np.in1d(istr.month, select_months) & (~((istr.month == 2) & (istr.day == 29))):
        select_dtime.append(True)
    else:
        select_dtime.append(False)
select_dtime = np.array(select_dtime, dtype=bool)
fvtime5 = cftime[select_dtime]
fvdata5 = fvdata5[select_dtime, :, :]
fvdata5 = fvdata5 * 86400 * 1000
fvnyears5 = int(len(fvtime5)/92)

# fvdata5 = np.ma.array(fvdata5)

# for idx in range(len(fvtime5)):
#     fvdata5[idx, :, :] = np.ma.masked_where(fvlndfrc < 0.5, fvdata5[idx, :, :])

# print(fvdata5[0, :, :])
print(fvdata5.shape)
print(fvtime5)

# fvdata2[fvdata2.mask] = np.nan
# fvdata3[fvdata3.mask] = np.nan
# fvdata4[fvdata4.mask] = np.nan
# fvdata5[fvdata5.mask] = np.nan

################################################################################################
# S3-calculate for counts of extremes
################################################################################################
print('Calculating and plotting for occurence..')

for ireg in range(len(reg_names)):
    print('Plotting for region: '+reg_names[ireg])
    vrreg_latli = np.abs(vrlats - reg_lats[ireg][0]).argmin()
    vrreg_latui = np.abs(vrlats - reg_lats[ireg][1]).argmin()

    vrreg_lonli = np.abs(vrlons - reg_lons[ireg][0]).argmin()
    vrreg_lonui = np.abs(vrlons - reg_lons[ireg][1]).argmin()

    fvreg_latli = np.abs(fvlats - reg_lats[ireg][0]).argmin()
    fvreg_latui = np.abs(fvlats - reg_lats[ireg][1]).argmin()

    fvreg_lonli = np.abs(fvlons - reg_lons[ireg][0]).argmin()
    fvreg_lonui = np.abs(fvlons - reg_lons[ireg][1]).argmin()

    plot_data = [vrdata2[:, vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1], fvdata2[:, fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1],
                 vrdata3[:, vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui +
                         1], fvdata3[:, fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1],
                 vrdata4[:, vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui +
                         1], fvdata4[:, fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1],
                 vrdata5[:, vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1], fvdata5[:, fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]]

    labels = [r'$0.25-S_{2000}A_{2000}$', r'$1-S_{2000}A_{2000}$',
              r'$0.25-S_{PERT}A_{2000}$', r'$1-S_{PERT}A_{2000}$',
              r'$0.25-S_{2000}A_{1950}$', r'$1-S_{2000}A_{1950}$',
              r'$0.25-S_{PERT}A_{1950}$', r'$1-S_{PERT}A_{1950}$']
    colors = ['red', 'red', 'blue', 'blue', 'green', 'green', 'orange', 'orange']
    linetypes = ['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']

    fname = 'vrseasia_aerosol_amip_jja_'+varfname+'_extremes_' + \
        reg_names[ireg]+'_hist_exps_'+str(inimonth)+"to"+str(endmonth)
    title = 'Aerosol experiments hist '+varstr+' over '+reg_names[ireg]
    plot_hist(plot_data, labels, colors, linetypes, title, outdir+fname)

    xticklabels = [r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$',
                   r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$']
    labels = [r'$S_{2000}A_{2000}$', r'$S_{PERT}A_{2000}$', r'$S_{2000}A_{1950}$', r'$S_{PERT}A_{1950}$']

    fname = 'vrseasia_aerosol_amip_jja_'+varfname+'_extremes_' + \
        reg_names[ireg]+'_bar_exps_'+str(inimonth)+"to"+str(endmonth)
    title = 'Aerosol experiments mean '+varstr+' over '+reg_names[ireg]
    plot_bars(plot_data, labels, xticklabels, title, outdir+fname)

    # print(lats[reg_latli])
    # print(lats[reg_latui])
    # print(lons[reg_lonli])
    # print(lons[reg_lonui])

    ##############################################################
    # calculate moderate precip and heavy precip counts for region

    vrvar2_mod = (vrdata2 >= mod_thres[ireg][0]) & (vrdata2 < mod_thres[ireg][1])
    vrvar2_mod_cnt = np.sum(vrvar2_mod, axis=0)/nyears
    vrvar2_ext = (vrdata2 >= mod_thres[ireg][1])
    vrvar2_ext_cnt = np.sum(vrvar2_ext, axis=0)/nyears

    vrvar3_mod = (vrdata3 >= mod_thres[ireg][0]) & (vrdata3 < mod_thres[ireg][1])
    vrvar3_mod_cnt = np.sum(vrvar3_mod, axis=0)/nyears
    vrvar3_ext = (vrdata3 >= mod_thres[ireg][1])
    vrvar3_ext_cnt = np.sum(vrvar3_ext, axis=0)/nyears

    vrvar4_mod = (vrdata4 >= mod_thres[ireg][0]) & (vrdata4 < mod_thres[ireg][1])
    vrvar4_mod_cnt = np.sum(vrvar4_mod, axis=0)/nyears
    vrvar4_ext = (vrdata4 >= mod_thres[ireg][1])
    vrvar4_ext_cnt = np.sum(vrvar4_ext, axis=0)/nyears

    vrvar5_mod = (vrdata5 >= mod_thres[ireg][0]) & (vrdata5 < mod_thres[ireg][1])
    vrvar5_mod_cnt = np.sum(vrvar5_mod, axis=0)/nyears
    vrvar5_ext = (vrdata5 >= mod_thres[ireg][1])
    vrvar5_ext_cnt = np.sum(vrvar5_ext, axis=0)/nyears

    fvvar2_mod = (fvdata2 >= mod_thres[ireg][0]) & (fvdata2 < mod_thres[ireg][1])
    fvvar2_mod_cnt = np.sum(fvvar2_mod, axis=0)/fvnyears2
    fvvar2_ext = (fvdata2 >= mod_thres[ireg][1])
    fvvar2_ext_cnt = np.sum(fvvar2_ext, axis=0)/fvnyears2

    fvvar3_mod = (fvdata3 >= mod_thres[ireg][0]) & (fvdata3 < mod_thres[ireg][1])
    fvvar3_mod_cnt = np.sum(fvvar3_mod, axis=0)/fvnyears3
    fvvar3_ext = (fvdata3 >= mod_thres[ireg][1])
    fvvar3_ext_cnt = np.sum(fvvar3_ext, axis=0)/fvnyears3

    fvvar4_mod = (fvdata4 >= mod_thres[ireg][0]) & (fvdata4 < mod_thres[ireg][1])
    fvvar4_mod_cnt = np.sum(fvvar4_mod, axis=0)/fvnyears4
    fvvar4_ext = (fvdata4 >= mod_thres[ireg][1])
    fvvar4_ext_cnt = np.sum(fvvar4_ext, axis=0)/fvnyears4

    fvvar5_mod = (fvdata5 >= mod_thres[ireg][0]) & (fvdata5 < mod_thres[ireg][1])
    fvvar5_mod_cnt = np.sum(fvvar5_mod, axis=0)/fvnyears5
    fvvar5_ext = (fvdata5 >= mod_thres[ireg][1])
    fvvar5_ext_cnt = np.sum(fvvar5_ext, axis=0)/fvnyears5

    ##############################################################
    # plot for moderate precipitation
    vrvar2_reg = vrvar2_mod_cnt[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
    vrvar3_reg = vrvar3_mod_cnt[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
    vrvar4_reg = vrvar4_mod_cnt[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
    vrvar5_reg = vrvar5_mod_cnt[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]

    fvvar2_reg = fvvar2_mod_cnt[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
    fvvar3_reg = fvvar3_mod_cnt[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
    fvvar4_reg = fvvar4_mod_cnt[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
    fvvar5_reg = fvvar5_mod_cnt[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]

    print(fvvar2_reg)

    # plot for each cases
    plot_data = [vrvar2_reg, fvvar2_reg,  vrvar3_reg, fvvar3_reg,
                 vrvar4_reg, fvvar4_reg, vrvar5_reg, fvvar5_reg]

    xticklabels = [r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$',
                   r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$']
    labels = [r'$S_{2000}A_{2000}$', r'$S_{PERT}A_{2000}$', r'$S_{2000}A_{1950}$', r'$S_{PERT}A_{1950}$']
    ylabel = 'Distribution of moderate Precipitation Counts [Days]'
    yticks = np.arange(0, 80, 10)

    fname = 'vrseasia_aerosol_amip_jja_moderate_'+varfname+'_extremes_' + \
        reg_names[ireg]+'_box_exps_'+str(inimonth)+"to"+str(endmonth)
    title = 'Aerosol experiments moderate '+varstr+' over '+reg_names[ireg]
    plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir+fname)

    # plot for absolute response
    xticklabels = [r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$',
                   r'$0.25^{\circ}$', r'$1^{\circ}$']
    labels = ['All Aerosol', 'ATM Aerosol', 'OCN Aerosol']

    vrres3 = vrvar2_reg - vrvar5_reg
    vrres4 = (vrvar2_reg + vrvar3_reg - vrvar4_reg - vrvar5_reg)/2
    vrres5 = (vrvar4_reg + vrvar2_reg - vrvar5_reg - vrvar3_reg)/2

    fvres3 = fvvar2_reg - fvvar5_reg
    fvres4 = (fvvar2_reg + fvvar3_reg - fvvar4_reg - fvvar5_reg)/2
    fvres5 = (fvvar4_reg + fvvar2_reg - fvvar5_reg - fvvar3_reg)/2

    plot_data = [vrres3, fvres3, vrres4, fvres4, vrres5, fvres5]

    ylabel = 'Changes in Distribution of moderate Precipitation Counts [days]'
    yticks = np.arange(-12.5, 15, 12.5)

    fname = 'vrseasia_aerosol_amip_jja_moderate_'+varfname+'_extremes_' + \
        reg_names[ireg]+'_box_absolute_response_'+str(inimonth)+"to"+str(endmonth)
    title = 'Changes in moderate '+varstr+' over '+reg_names[ireg]
    plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir+fname)

    # plot for relative response
    vrres3 = (vrvar2_reg - vrvar5_reg)/vrvar5_reg*100
    vrres4 = (vrvar2_reg + vrvar3_reg - vrvar4_reg - vrvar5_reg)/(vrvar4_reg + vrvar5_reg)*100
    vrres5 = (vrvar4_reg + vrvar2_reg - vrvar5_reg - vrvar3_reg)/(vrvar5_reg + vrvar3_reg)*100

    fvres3 = (fvvar2_reg - fvvar5_reg)/fvvar5_reg*100
    fvres4 = (fvvar2_reg + fvvar3_reg - fvvar4_reg - fvvar5_reg)/(fvvar4_reg + fvvar5_reg)*100
    fvres5 = (fvvar4_reg + fvvar2_reg - fvvar5_reg - fvvar3_reg)/(fvvar5_reg + fvvar3_reg)*100

    plot_data = [vrres3, fvres3, vrres4, fvres4, vrres5, fvres5]

    ylabel = 'Relative Changes in Distribution of moderate Precipitation Counts [%]'
    yticks = np.arange(-25, 30, 5)

    fname = 'vrseasia_aerosol_amip_jja_moderate_'+varfname+'_extremes_' + \
        reg_names[ireg]+'_box_relative_response_'+str(inimonth)+"to"+str(endmonth)
    title = 'Relative Changes in moderate '+varstr+' over '+reg_names[ireg]
    plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir+fname)

    ##############################################################
    # plot for heavy precipitation
    vrvar2_reg = vrvar2_ext_cnt[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
    vrvar3_reg = vrvar3_ext_cnt[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
    vrvar4_reg = vrvar4_ext_cnt[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
    vrvar5_reg = vrvar5_ext_cnt[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]

    fvvar2_reg = fvvar2_ext_cnt[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
    fvvar3_reg = fvvar3_ext_cnt[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
    fvvar4_reg = fvvar4_ext_cnt[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
    fvvar5_reg = fvvar5_ext_cnt[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]

    print(fvvar2_reg)

    # plot for each cases
    plot_data = [vrvar2_reg, fvvar2_reg,  vrvar3_reg, fvvar3_reg,
                 vrvar4_reg, fvvar4_reg, vrvar5_reg, fvvar5_reg]

    xticklabels = [r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$',
                   r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$']
    labels = [r'$S_{2000}A_{2000}$', r'$S_{PERT}A_{2000}$', r'$S_{2000}A_{1950}$', r'$S_{PERT}A_{1950}$']
    ylabel = 'Distribution of heavy Precipitation Counts [Days]'
    yticks = np.arange(0, 7, 1)

    fname = 'vrseasia_aerosol_amip_jja_heavy_'+varfname+'_extremes_' + \
        reg_names[ireg]+'_box_exps_'+str(inimonth)+"to"+str(endmonth)
    title = 'Aerosol experiments heavy '+varstr+' over '+reg_names[ireg]
    plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir+fname)

    # plot for absolute response
    xticklabels = [r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$',
                   r'$0.25^{\circ}$', r'$1^{\circ}$']
    labels = ['All Aerosol', 'ATM Aerosol', 'OCN Aerosol']

    vrres3 = vrvar2_reg - vrvar5_reg
    vrres4 = (vrvar2_reg + vrvar3_reg - vrvar4_reg - vrvar5_reg)/2
    vrres5 = (vrvar4_reg + vrvar2_reg - vrvar5_reg - vrvar3_reg)/2

    fvres3 = fvvar2_reg - fvvar5_reg
    fvres4 = (fvvar2_reg + fvvar3_reg - fvvar4_reg - fvvar5_reg)/2
    fvres5 = (fvvar4_reg + fvvar2_reg - fvvar5_reg - fvvar3_reg)/2

    plot_data = [vrres3, fvres3, vrres4, fvres4, vrres5, fvres5]

    ylabel = 'Changes in Distribution of heavy Precipitation Counts [days]'
    yticks = np.arange(-2., 2.1, 0.5)

    fname = 'vrseasia_aerosol_amip_jja_heavy_'+varfname+'_extremes_' + \
        reg_names[ireg]+'_box_absolute_response_'+str(inimonth)+"to"+str(endmonth)
    title = 'Changes in heavy '+varstr+' over '+reg_names[ireg]
    plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir+fname)

    # plot for relative response
    vrres3 = (vrvar2_reg - vrvar5_reg)/vrvar5_reg*100
    vrres4 = (vrvar2_reg + vrvar3_reg - vrvar4_reg - vrvar5_reg)/(vrvar4_reg + vrvar5_reg)*100
    vrres5 = (vrvar4_reg + vrvar2_reg - vrvar5_reg - vrvar3_reg)/(vrvar5_reg + vrvar3_reg)*100

    fvres3 = (fvvar2_reg - fvvar5_reg)/fvvar5_reg*100
    fvres4 = (fvvar2_reg + fvvar3_reg - fvvar4_reg - fvvar5_reg)/(fvvar4_reg + fvvar5_reg)*100
    fvres5 = (fvvar4_reg + fvvar2_reg - fvvar5_reg - fvvar3_reg)/(fvvar5_reg + fvvar3_reg)*100

    plot_data = [vrres3, fvres3, vrres4, fvres4, vrres5, fvres5]

    ylabel = 'Relative Changes in Distribution of heavy Precipitation Counts [%]'
    yticks = np.arange(-100., 255, 50.)

    fname = 'vrseasia_aerosol_amip_jja_heavy_'+varfname+'_extremes_' + \
        reg_names[ireg]+'_box_relative_response_'+str(inimonth)+"to"+str(endmonth)
    title = 'Relative Changes in heavy '+varstr+' over '+reg_names[ireg]
    plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir+fname)

################################################################################################
# S4-calculate for 99th extremes
################################################################################################
print('Calculating and plotting for extremes..')

for ipercent in range(len(percentile_ranges)):
    percentile = percentile_ranges[ipercent]
    outdir_percent = outdir + str(percentile)+'th/'
    print('Plotting for '+str(percentile)+'th...')

    vrvar2_extreme = np.percentile(vrdata2, percentile, axis=0)
    vrvar3_extreme = np.percentile(vrdata3, percentile, axis=0)
    vrvar4_extreme = np.percentile(vrdata4, percentile, axis=0)
    vrvar5_extreme = np.percentile(vrdata5, percentile, axis=0)

    fvvar2_extreme = np.percentile(fvdata2, percentile, axis=0)
    fvvar3_extreme = np.percentile(fvdata3, percentile, axis=0)
    fvvar4_extreme = np.percentile(fvdata4, percentile, axis=0)
    fvvar5_extreme = np.percentile(fvdata5, percentile, axis=0)

    for ireg in range(len(reg_names)):
        print('Plotting for region: '+reg_names[ireg])
        vrreg_latli = np.abs(vrlats - reg_lats[ireg][0]).argmin()
        vrreg_latui = np.abs(vrlats - reg_lats[ireg][1]).argmin()

        vrreg_lonli = np.abs(vrlons - reg_lons[ireg][0]).argmin()
        vrreg_lonui = np.abs(vrlons - reg_lons[ireg][1]).argmin()

        fvreg_latli = np.abs(fvlats - reg_lats[ireg][0]).argmin()
        fvreg_latui = np.abs(fvlats - reg_lats[ireg][1]).argmin()

        fvreg_lonli = np.abs(fvlons - reg_lons[ireg][0]).argmin()
        fvreg_lonui = np.abs(fvlons - reg_lons[ireg][1]).argmin()

        vrvar2_reg = vrvar2_extreme[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
        vrvar3_reg = vrvar3_extreme[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
        vrvar4_reg = vrvar4_extreme[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
        vrvar5_reg = vrvar5_extreme[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]

        fvvar2_reg = fvvar2_extreme[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
        fvvar3_reg = fvvar3_extreme[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
        fvvar4_reg = fvvar4_extreme[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
        fvvar5_reg = fvvar5_extreme[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]

        # plot for each cases
        plot_data = [vrvar2_reg, fvvar2_reg,  vrvar3_reg, fvvar3_reg,
                     vrvar4_reg, fvvar4_reg, vrvar5_reg, fvvar5_reg]

        xticklabels = [r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$',
                       r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$']
        labels = [r'$S_{2000}A_{2000}$', r'$S_{PERT}A_{2000}$', r'$S_{2000}A_{1950}$', r'$S_{PERT}A_{1950}$']
        yticks = []
        ylabel = 'Distribution of '+str(percentile)+'th Precipitation [mm/day]'

        fname = 'vrseasia_aerosol_amip_jja_'+varfname+'_' + \
            str(percentile)+'_th_extremes_'+reg_names[ireg]+'_box_exps_'+str(inimonth)+"to"+str(endmonth)
        title = 'Aerosol experiments '+varstr+' Rp'+str(percentile)+' over '+reg_names[ireg]
        plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir_percent+fname)

        # plot for absolute response
        xticklabels = [r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$',
                       r'$0.25^{\circ}$', r'$1^{\circ}$']
        labels = ['All Aerosol', 'ATM Aerosol', 'OCN Aerosol']

        vrres3 = vrvar2_reg - vrvar5_reg
        vrres4 = (vrvar2_reg + vrvar3_reg - vrvar4_reg - vrvar5_reg)/2
        vrres5 = (vrvar4_reg + vrvar2_reg - vrvar5_reg - vrvar3_reg)/2

        fvres3 = fvvar2_reg - fvvar5_reg
        fvres4 = (fvvar2_reg + fvvar3_reg - fvvar4_reg - fvvar5_reg)/2
        fvres5 = (fvvar4_reg + fvvar2_reg - fvvar5_reg - fvvar3_reg)/2

        plot_data = [vrres3, fvres3, vrres4, fvres4, vrres5, fvres5]

        yticks = []
        ylabel = 'Changes in Distribution of '+str(percentile)+'th Precipitation [mm/day]'

        fname = 'vrseasia_aerosol_amip_jja_'+varfname+'_' + \
            str(percentile)+'_th_extremes_'+reg_names[ireg]+'_box_absolute_response_'+str(inimonth)+"to"+str(endmonth)
        title = 'Changes in '+varstr+' Rp'+str(percentile)+' over '+reg_names[ireg]

        plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir_percent+fname)

        # plot for absolute response
        vrres3 = (vrvar2_reg - vrvar5_reg)/vrvar5_reg*100
        vrres4 = (vrvar2_reg + vrvar3_reg - vrvar4_reg - vrvar5_reg)/(vrvar4_reg + vrvar5_reg)*100
        vrres5 = (vrvar4_reg + vrvar2_reg - vrvar5_reg - vrvar3_reg)/(vrvar5_reg + vrvar3_reg)*100

        fvres3 = (fvvar2_reg - fvvar5_reg)/fvvar5_reg*100
        fvres4 = (fvvar2_reg + fvvar3_reg - fvvar4_reg - fvvar5_reg)/(fvvar4_reg + fvvar5_reg)*100
        fvres5 = (fvvar4_reg + fvvar2_reg - fvvar5_reg - fvvar3_reg)/(fvvar5_reg + fvvar3_reg)*100

        plot_data = [vrres3, fvres3, vrres4, fvres4, vrres5, fvres5]

        yticks = []
        ylabel = 'Relative Changes in Distribution of '+str(percentile)+'th Precipitation [%]'

        fname = 'vrseasia_aerosol_amip_jja_'+varfname+'_' + \
            str(percentile)+'_th_extremes_'+reg_names[ireg]+'_box_relative_response_'+str(inimonth)+"to"+str(endmonth)
        title = 'Relative Changes in '+varstr+' Rp'+str(percentile)+' over '+reg_names[ireg]
        plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir_percent+fname)


################################################################################################
# S5-calculate for climatological response
################################################################################################
print('Calculating and plotting for climatological responses..')

vrvar2_mean = np.mean(vrdata2, axis=0)
vrvar3_mean = np.mean(vrdata3, axis=0)
vrvar4_mean = np.mean(vrdata4, axis=0)
vrvar5_mean = np.mean(vrdata5, axis=0)

fvvar2_mean = np.mean(fvdata2, axis=0)
fvvar3_mean = np.mean(fvdata3, axis=0)
fvvar4_mean = np.mean(fvdata4, axis=0)
fvvar5_mean = np.mean(fvdata5, axis=0)

for ireg in range(len(reg_names)):
    print('Plotting for region: '+reg_names[ireg])
    vrreg_latli = np.abs(vrlats - reg_lats[ireg][0]).argmin()
    vrreg_latui = np.abs(vrlats - reg_lats[ireg][1]).argmin()

    vrreg_lonli = np.abs(vrlons - reg_lons[ireg][0]).argmin()
    vrreg_lonui = np.abs(vrlons - reg_lons[ireg][1]).argmin()

    fvreg_latli = np.abs(fvlats - reg_lats[ireg][0]).argmin()
    fvreg_latui = np.abs(fvlats - reg_lats[ireg][1]).argmin()

    fvreg_lonli = np.abs(fvlons - reg_lons[ireg][0]).argmin()
    fvreg_lonui = np.abs(fvlons - reg_lons[ireg][1]).argmin()

    vrvar2_reg = vrvar2_mean[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
    vrvar3_reg = vrvar3_mean[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
    vrvar4_reg = vrvar4_mean[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]
    vrvar5_reg = vrvar5_mean[vrreg_latli: vrreg_latui+1, vrreg_lonli: vrreg_lonui+1]

    fvvar2_reg = fvvar2_mean[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
    fvvar3_reg = fvvar3_mean[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
    fvvar4_reg = fvvar4_mean[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]
    fvvar5_reg = fvvar5_mean[fvreg_latli: fvreg_latui+1, fvreg_lonli: fvreg_lonui+1]

    # plot for each cases
    plot_data = [vrvar2_reg, fvvar2_reg,  vrvar3_reg, fvvar3_reg,
                 vrvar4_reg, fvvar4_reg, vrvar5_reg, fvvar5_reg]

    xticklabels = [r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$',
                   r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$']
    labels = [r'$S_{2000}A_{2000}$', r'$S_{PERT}A_{2000}$', r'$S_{2000}A_{1950}$', r'$S_{PERT}A_{1950}$']
    yticks = []
    ylabel = 'Seasonal mean precipitation [mm/day]'

    fname = 'vrseasia_aerosol_amip_jja_'+varfname+'_mean_'+reg_names[ireg]+'_box_exps_'+str(inimonth)+"to"+str(endmonth)
    title = 'Aerosol experiments climatological mean prect over '+reg_names[ireg]
    plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir+fname)

    # plot for absolute response
    xticklabels = [r'$0.25^{\circ}$', r'$1^{\circ}$', r'$0.25^{\circ}$', r'$1^{\circ}$',
                   r'$0.25^{\circ}$', r'$1^{\circ}$']
    labels = ['All Aerosol', 'ATM Aerosol', 'OCN Aerosol']

    vrres3 = vrvar2_reg - vrvar5_reg
    vrres4 = (vrvar2_reg + vrvar3_reg - vrvar4_reg - vrvar5_reg)/2
    vrres5 = (vrvar4_reg + vrvar2_reg - vrvar5_reg - vrvar3_reg)/2

    fvres3 = fvvar2_reg - fvvar5_reg
    fvres4 = (fvvar2_reg + fvvar3_reg - fvvar4_reg - fvvar5_reg)/2
    fvres5 = (fvvar4_reg + fvvar2_reg - fvvar5_reg - fvvar3_reg)/2

    plot_data = [vrres3, fvres3, vrres4, fvres4, vrres5, fvres5]

    yticks = []
    ylabel = 'Changes in seasonal mean precipitation [mm/day]'

    fname = 'vrseasia_aerosol_amip_jja_'+varfname+'_mean_'+reg_names[ireg]+'_box_absolute_response_'+str(inimonth)+"to"+str(endmonth)
    title = 'Changes in climatological mean prect over '+reg_names[ireg]

    plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir+fname)

    # plot for absolute response
    vrres3 = (vrvar2_reg - vrvar5_reg)/vrvar5_reg*100
    vrres4 = (vrvar2_reg + vrvar3_reg - vrvar4_reg - vrvar5_reg)/(vrvar4_reg + vrvar5_reg)*100
    vrres5 = (vrvar4_reg + vrvar2_reg - vrvar5_reg - vrvar3_reg)/(vrvar5_reg + vrvar3_reg)*100

    fvres3 = (fvvar2_reg - fvvar5_reg)/fvvar5_reg*100
    fvres4 = (fvvar2_reg + fvvar3_reg - fvvar4_reg - fvvar5_reg)/(fvvar4_reg + fvvar5_reg)*100
    fvres5 = (fvvar4_reg + fvvar2_reg - fvvar5_reg - fvvar3_reg)/(fvvar5_reg + fvvar3_reg)*100

    plot_data = [vrres3, fvres3, vrres4, fvres4, vrres5, fvres5]

    yticks = []
    ylabel = 'Relative Changes in seasonal mean precipitation [%]'

    fname = 'vrseasia_aerosol_amip_jja_'+varfname+'_mean_'+reg_names[ireg]+'_box_relative_response_'+str(inimonth)+"to"+str(endmonth)
    title = 'Relative Changes in climatological mean prect over '+reg_names[ireg]
    plot_box(plot_data, labels, xticklabels, yticks, ylabel, title, outdir+fname)
