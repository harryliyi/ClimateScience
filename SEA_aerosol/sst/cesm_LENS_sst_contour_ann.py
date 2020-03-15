# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import math as math
import pandas as pd
import matplotlib.cm as cm
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, shiftgrid
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# set up data directories and filenames
refdir1 = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/sst/"
refdir2 = "/scratch/d/dylan/harryli/cesm1/inputdata/atm/cam/sst/"

# set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_Aerosol/sst/LENS/"

# set up variable names and file name
varname = 'SST'
varfname = 'sst'
var_longname = "Sea Surface Temperature"
var_unit = r'$^{\circ}C$'
var_res = "fv09"

# define inital year and end year
iniyear = 1950
endyear = 2019
nyears = endyear - iniyear + 1

# define the contour plot region
latbounds = [-20, 50]
lonbounds = [40, 160]

# define Southeast region
reg_lats = [10, 25]
reg_lons = [100, 110]

# ensemble numbers
allens = 30
aerens = 20
ennum = 30

# month series
month = np.arange(1, 13, 1)
mname = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


################################################################################################
# S0-Define functions
################################################################################################
def getstats_diff(var1, var2):
    n1 = var1.shape[0]
    n2 = var2.shape[0]

    var1mean = np.mean(var1, axis=0)
    var2mean = np.mean(var2, axis=0)
    var1std = np.std(var1, axis=0)
    var2std = np.std(var2, axis=0)

    vardiff = var1mean - var2mean
    varttest = vardiff/np.sqrt(var1std**2/n1+var2std**2/n2)

    return vardiff, abs(varttest)


########################################################################################
# plot for all responses together


def plotalldiff(lons, lats, res, tt, opt):

    fig = plt.figure()

    # coulped response
    ax1 = fig.add_subplot(111)
    # ax1.set_title(r'$\Delta_{coupled}$ '+varname+', regional mean='+str(round(np.mean(res1),4))+var_unit,fontsize=5,pad=3)
    # ax1.set_title(r'$\Delta_{coupled}$ '+varname, fontsize=6, pad=3)

    map = Basemap(projection='cyl', llcrnrlat=-90., urcrnrlat=90.,
                  llcrnrlon=-180., urcrnrlon=180., resolution='c')
    map.drawcoastlines(linewidth=0.3)
    map.drawcountries()
    parallels = np.arange(-90., 91., 30.)
    meridians = np.arange(-180., 181., 60.)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)
    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)
    # print(mlons)
    # island = map.is_land(mlons.flatten(), mlats.flatten())
    # island = island.reshape((len(lons), len(lats)))
    # res[island] = np.nan
    clevs = np.arange(-1., 1.1, 0.1)
    cs = plt.contourf(x, y, res, clevs, cmap=cm.RdBu_r, alpha=0.9, extend="both")

    if (opt == 1):
        levels = [0., 2.01, tt.max()]
        csm = plt.contourf(x, y, tt, levels=levels, colors='none', hatches=["", "....."], alpha=0)

    map.fillcontinents()

    # colorbar
    fig.subplots_adjust(bottom=0.15, wspace=0.1, hspace=0.05)
    cbar_ax = fig.add_axes([0.15, 0.2, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
#    cbar = fig.colorbar(cs,orientation='horizontal',fraction=0.15, aspect= 25,shrink = 0.8)
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label(var_longname+' ['+var_unit+']', fontsize=5, labelpad=-0.5)

    # add title
    if (opt == 1):
        plt.savefig(outdir+"cesm_lens_ann_"+varfname +
                    "_regrid_GLOBE_contour_aero_response_with_siglev.png", dpi=600, bbox_inches='tight')
    else:
        plt.savefig(outdir+"cesm_lens_ann_"+varfname +
                    "_regrid_GLOBE_contour_aero_response.png", dpi=600, bbox_inches='tight')

    plt.suptitle("Aerosol Responses "+var_longname+" changes", fontsize=10, y=0.95)
    if (opt == 1):
        plt.savefig(outdir+"cesm_lens_ann_"+varfname +
                    "_regrid_GLOBE_contour_aero_response_with_siglev.pdf", bbox_inches='tight')
    else:
        plt.savefig(outdir+"cesm_lens_ann_"+varfname+"_regrid_GLOBE_contour_aero_response.pdf", bbox_inches='tight')
    plt.close(fig)


################################################################################################
# S1-open climatological data
################################################################################################


# read reference data grids
fname1 = 'sst_HadOIBl_bc_1x1_20002010_mean.nc'
refdata1 = Dataset(refdir1+fname1)

fname2 = 'CESMCAM5_SST_NO_AER.nc'
refdata2 = Dataset(refdir2+fname2)

rlats = refdata1.variables['lat'][:]
rlons = refdata1.variables['lon'][:]
# rlons[rlons > 180] = rlons[rlons > 180] - 360.
print(rlons[:])

nrlats = len(rlats)
nrlons = len(rlats)

rvar_all = np.zeros((12, nrlats, nrlons))
rvar_xaer = np.zeros((12, nrlats, nrlons))

rvar_all = refdata1.variables['SST_cpl'][:]
rvar_xaer = refdata2.variables['SST_cpl'][:]


print(rvar_all)


# reg_boolean = (rlats >= reg_lats[0]) & (rlats <= reg_lats[1]) & (rlons >= reg_lons[0]) & (rlons <= reg_lons[1])
#
# print('total grids of selected region: '+str(np.sum(reg_boolean)))
#
# nrlats = rlats.shape[0]
# nrlons = rlats.shape[1]
#
# print('total latitude grids: '+str(nrlats))
# print('total longitude grids: '+str(nrlons))

# test the selected region
# testsst = refdata1.variables['SST'][:, 0, :, :]
# print(testsst.shape)
# for i in range(testsst.shape[0]):
#     temp = testsst[i, :, :]
#     print(np.mean(temp[reg_boolean]))


#################################################################################################
# contour plot for aerosol responses
#################################################################################################
rvar_all20s = np.mean(rvar_all, axis=0)
rvar_xaer20s = np.mean(rvar_xaer, axis=0)

rvar_all20s_new, rlons_new = shiftgrid(180, rvar_all20s, rlons, start=False, cyclic=360.0)
rvar_xaer20s_new, rlons_new = shiftgrid(180, rvar_xaer20s, rlons, start=False, cyclic=360.0)

print(rvar_all20s)

# rvar_all19s = rvar_all[0: (1959-iniyear+1)*12, :, :, :]
# rvar_all20s = rvar_all[(2000-iniyear)*12: (2009-iniyear+1)*12, :, :, :]
# rvar_xaer19s = rvar_xaer[0: (1959-iniyear+1)*12, :, :, :]
# rvar_xaer20s = rvar_xaer[(2000-iniyear)*12: (2009-iniyear+1)*12, :, :, :]
#
#
# rvar_all19s = rvar_all19s.reshape(10*12*30, nrlats, nrlons)
# rvar_all20s = rvar_all20s.reshape(10*12*30, nrlats, nrlons)
# rvar_xaer19s = rvar_xaer19s.reshape(10*12*20, nrlats, nrlons)
# rvar_xaer20s = rvar_xaer20s.reshape(10*12*20, nrlats, nrlons)

# res, tt = getstats_diff((rvar_all20s-rvar_all19s), (rvar_xaer20s-rvar_xaer19s))

res = rvar_all20s_new - rvar_xaer20s_new
tt = 0

print('Plot for contours')
# res_shift, rlonsout = shiftgrid(-180, res, rlons, start=True, cyclic=360.0)
plotalldiff(rlons_new, rlats, res, tt, 0)
# plotalldiff(rlons, rlats, res, tt, 1)

del rvar_all20s, rvar_xaer20s
