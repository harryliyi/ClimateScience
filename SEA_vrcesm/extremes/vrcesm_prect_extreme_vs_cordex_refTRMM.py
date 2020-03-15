# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-plot basic analysis
# S3-calculate and plot extreme
#
# Written by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.stats.mod_stats_extremes import rmse, pdffunc, cdffunc, pdffunc_noloc, cdffunc_noloc, kstest, cdfcheck, pdfcheck, gpdfit_moment
from modules.datareader.mod_dataread_obs_TRMM import readobs_pre_TRMM_day
from modules.plot.mod_plt_lines import plot_lines
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex
from scipy.stats import genpareto as gpd
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import basemap
import pickle
plt.switch_backend('agg')

############################################################################
# setup directory
############################################################################
ncluster = 4
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/TRMM/extremes/mainSEA/overland/' + \
    str(ncluster)+' clusters/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# time bounds
iniyear = 1998
endyear = 2005
yearts = np.arange(iniyear, endyear+1)

# define regions
latbounds = [5, 24]
lonbounds = [97, 110]

# mainland Southeast Asia
reg_lats = [5, 24]
reg_lons = [97, 110]

# set data frequency
frequency = 'day'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# percentile
percentile = 97
premin = 0.

# set up ignored years
ignore_years = []

# set up nbins
nbins = 50

############################################################################
# functions
############################################################################


def plot_fitcheck(plot_data, plot_fit, plot_ks, legends, groupidx):

    for idx in range(len(plot_data)):
        temp_data = plot_data[idx]
        temp_fit = plot_fit[idx]
        tmep_ks = plot_ks[idx]

        plt.clf()
        fig = plt.figure(1)

        # plot pdf
        ax1 = plt.subplot(3, 1, 1)
        x, xbins, ypdf = pdfcheck(temp_data, nbins, temp_fit)
        ax1.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
        ax1.hist(temp_data, bins=xbins, alpha=0.5, density=True, label='Empirical')
        ax1.set_title('pdf check, ks p-value ='+str(tmep_ks.pvalue), fontsize=7, pad=2)
        ax1.legend(loc='upper right', fontsize=8)
        # set x/y tick label size
        ax1.xaxis.set_tick_params(labelsize=5)
        ax1.yaxis.set_tick_params(labelsize=5)
        ax1.set_xlabel('Frequency', fontsize=5)
        ax1.set_ylabel('Precip (mm/day)', fontsize=5)

        # plot qq-plot
        ax2 = fig.add_subplot(3, 1, 2)
        temp_res = ss.probplot(temp_data, dist=ss.genpareto, sparams=(temp_fit[0], 0., temp_fit[1]))
        ax2.scatter(temp_res[0][0], temp_res[0][1], c='b', s=3)
        ax2.plot(temp_res[0][0], temp_res[0][0], linestyle='solid', linewidth=1.5, color='r')
        ax2.set_title('qq-plot', fontsize=7, pad=2)
        # ax2.legend(loc='upper right', fontsize=8)
        # set x/y tick label size
        ax2.xaxis.set_tick_params(labelsize=5)
        ax2.yaxis.set_tick_params(labelsize=5)
        ax2.set_xlabel('Theratical quantiles', fontsize=5)
        ax2.set_ylabel('Ordered values', fontsize=5)

        ax3 = fig.add_subplot(3, 1, 3)
        x, xbins, ycdf = cdfcheck(temp_data, nbins, temp_fit)
        ax3.plot(x, ycdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
        ax3.hist(temp_data, bins=xbins, alpha=0.5, density=True, histtype='step', cumulative=True, label='Empirical')
        anderson = ss.anderson_ksamp([temp_data, gpd.rvs(temp_fit[0], 0., temp_fit[1], size=len(temp_data))])
        ax3.set_title('cdf check, AD sig level='+str(anderson.significance_level), fontsize=7, pad=2)
        ax3.legend(loc='lower right', fontsize=8)
        # set x/y tick label size
        ax3.xaxis.set_tick_params(labelsize=5)
        ax3.yaxis.set_tick_params(labelsize=5)
        ax3.set_xlabel('Probability', fontsize=5)
        ax3.set_ylabel('Precip (mm/day)', fontsize=5)

        title = 'Gourp: '+str(groupidx)+' '+str(iniyear)+' to '+str(endyear)+' ' + \
            legends[idx]+' '+str(percentile)+'th percentile precip GPD fit'
        plt.suptitle(title, fontsize=10, y=0.95)

        fig.subplots_adjust(hspace=0.2)
        fname = str(percentile)+'th_extremes_GPD_fit_group'+str(groupidx)+'_'+str(idx+1)+'.png'
        plt.savefig(outdir+fname, bbox_inches='tight', dpi=1000)
        plt.close(fig)


# function to plot label maps
def plot_label_map(plot_data, plot_lats, plot_lons, legends):
    print('plotting label maps...')
    for idx in range(len(plot_data)):
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # create basemap
        map = Basemap(projection='cyl', llcrnrlat=latbounds[0], urcrnrlat=latbounds[1],
                      llcrnrlon=lonbounds[0], urcrnrlon=lonbounds[1], resolution='l')
        map.drawcoastlines(linewidth=0.3)
        map.drawcountries()

        # draw lat/lon lines
        parallels = np.arange(latbounds[0], latbounds[1], 5)
        meridians = np.arange(lonbounds[0], lonbounds[1], 5)
        map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=4, linewidth=0.1)
        map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=4, linewidth=0.1)

        mlons, mlats = np.meshgrid(plot_lons[idx], plot_lats[idx])
        x, y = map(mlons, mlats)

        # plot the contour
        cs = map.pcolormesh(x, y, plot_data[idx],
                            cmap=plt.cm.get_cmap('viridis', ncluster), alpha=0.9)

        # add colorbar
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.3, 0.15, 0.4, 0.02])
        cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal', ticks=range(ncluster))
        cbar.ax.tick_params(labelsize=4)
        cbar.set_label('Cluster labels', fontsize=5)
        plt.clim(-0.5, ncluster-0.5)

        title = 'K-means '+str(ncluster)+'-cluster result of TRMM projected on '+legends[idx]
        fname = 'TRMM_precip_kmeans_'+str(ncluster)+'cluster_result'
        plt.suptitle(title, fontsize=5, y=0.95)
        plt.savefig(outdir+fname+'_on_'+str(idx+1)+'.pdf', bbox_inches='tight', dpi=3000)
        plt.close(fig)

############################################################################
# read data
############################################################################


print('Reading CORDEX-SEA data...')

# read cordex
project = 'SEA-22'
varname = 'pr'
cordex_models = ['ICHEC-EC-EARTH', 'IPSL-IPSL-CM5A-LR', 'MPI-M-MPI-ESM-MR', 'MOHC-HadGEM2-ES']

modelname = 'ICHEC-EC-EARTH'
cordex_var1, cordex_time1, cordex_lats1, cordex_lons1 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=1, ignore_years=ignore_years)
modelname = 'IPSL-IPSL-CM5A-LR'
cordex_var2, cordex_time2, cordex_lats2, cordex_lons2 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=1, ignore_years=ignore_years)
modelname = 'MPI-M-MPI-ESM-MR'
cordex_var3, cordex_time3, cordex_lats3, cordex_lons3 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=1, ignore_years=ignore_years)
modelname = 'MOHC-HadGEM2-ES'
cordex_var4, cordex_time4, cordex_lats4, cordex_lons4 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=1, ignore_years=ignore_years)

cordex_var1[cordex_var1.mask] = np.nan
cordex_var2[cordex_var2.mask] = np.nan
cordex_var3[cordex_var3.mask] = np.nan
cordex_var4[cordex_var4.mask] = np.nan

# print(cordex_var4.shape)
# print(cordex_time4)
# print(cordex_lats4)

# read vrcesm
print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'prec'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1, ignore_years=ignore_years)

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1, ignore_years=ignore_years)

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1, ignore_years=ignore_years)

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=1, ignore_years=ignore_years)

model_var1 = 86400 * 1000 * model_var1
model_var2 = 86400 * 1000 * model_var2
model_var3 = 86400 * 1000 * model_var3
model_var4 = 86400 * 1000 * model_var4

model_var1[model_var1.mask] = np.nan
model_var2[model_var2.mask] = np.nan
model_var3[model_var3.mask] = np.nan
model_var4[model_var4.mask] = np.nan

print(model_var2[0, :, :])
print(model_lats2)

# read Observations

print('Reading TRMM data...')

# read TRMM
obs_var, obs_time, obs_lats, obs_lons = readobs_pre_TRMM_day(
    'precipitation', iniyear, endyear, latbounds, lonbounds, oceanmask=1)

print(obs_time)
print(obs_lats)
print(obs_var.shape)
# print(obs_var.mask)

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES', 'SA-OBS']

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES', 'TRMM']
colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod', 'darkcyan', 'darkmagenta', 'black']
line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-.', '-.', '-.', '-.', '-']

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'TRMM']
cesm_colors = ['red', 'yellow', 'green', 'blue', 'black']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-']

############################################################################
# fit the extremes
############################################################################
# read kmeans result for TRMM data
res_load = pickle.load(open(
    '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/obs/TRMM/clustering/overland/TRMM_kmeans_result_overland_'+str(ncluster)+'cluster.p', "rb"))
kmeans_labels = res_load[str(ncluster)+' clusters']
kmeans_res_lats = res_load['kmeans_res_lats']
kmeans_res_lons = res_load['kmeans_res_lons']

# create the labels map for each dataset
cordex_map_labels1 = np.empty((len(cordex_lats1), len(cordex_lons1)))
cordex_map_labels2 = np.empty((len(cordex_lats2), len(cordex_lons2)))
cordex_map_labels3 = np.empty((len(cordex_lats3), len(cordex_lons3)))
cordex_map_labels4 = np.empty((len(cordex_lats4), len(cordex_lons4)))
model_map_labels1 = np.empty((len(model_lats1), len(model_lons1)))
model_map_labels2 = np.empty((len(model_lats2), len(model_lons2)))
model_map_labels3 = np.empty((len(model_lats3), len(model_lons3)))
model_map_labels4 = np.empty((len(model_lats4), len(model_lons4)))
obs_map_labels = np.empty((len(obs_lats), len(obs_lons)))

cordex_map_labels1[:] = np.nan
cordex_map_labels2[:] = np.nan
cordex_map_labels3[:] = np.nan
cordex_map_labels4[:] = np.nan
model_map_labels1[:] = np.nan
model_map_labels2[:] = np.nan
model_map_labels3[:] = np.nan
model_map_labels4[:] = np.nan
obs_map_labels[:] = np.nan

for idx in range(len(kmeans_labels)):
    # assgin the the label maps
    # # cordex
    # lat_res = np.argmin(np.abs(cordex_lats1 - kmeans_res_lats[idx]))
    # lon_res = np.argmin(np.abs(cordex_lons1 - kmeans_res_lons[idx]))
    # cordex_map_labels1[lat_res, lon_res] = kmeans_labels[idx]
    #
    # lat_res = np.argmin(np.abs(cordex_lats2 - kmeans_res_lats[idx]))
    # lon_res = np.argmin(np.abs(cordex_lons2 - kmeans_res_lons[idx]))
    # cordex_map_labels2[lat_res, lon_res] = kmeans_labels[idx]
    #
    # lat_res = np.argmin(np.abs(cordex_lats3 - kmeans_res_lats[idx]))
    # lon_res = np.argmin(np.abs(cordex_lons3 - kmeans_res_lons[idx]))
    # cordex_map_labels3[lat_res, lon_res] = kmeans_labels[idx]
    #
    # lat_res = np.argmin(np.abs(cordex_lats4 - kmeans_res_lats[idx]))
    # lon_res = np.argmin(np.abs(cordex_lons4 - kmeans_res_lons[idx]))
    # cordex_map_labels4[lat_res, lon_res] = kmeans_labels[idx]
    #
    # # cesm
    # lat_res = np.argmin(np.abs(model_lats1 - kmeans_res_lats[idx]))
    # lon_res = np.argmin(np.abs(model_lons1 - kmeans_res_lons[idx]))
    # model_map_labels1[lat_res, lon_res] = kmeans_labels[idx]

    lat_res = np.argmin(np.abs(model_lats2 - kmeans_res_lats[idx]))
    lon_res = np.argmin(np.abs(model_lons2 - kmeans_res_lons[idx]))
    model_map_labels2[lat_res, lon_res] = kmeans_labels[idx]

    lat_res = np.argmin(np.abs(model_lats3 - kmeans_res_lats[idx]))
    lon_res = np.argmin(np.abs(model_lons3 - kmeans_res_lons[idx]))
    model_map_labels3[lat_res, lon_res] = kmeans_labels[idx]

    lat_res = np.argmin(np.abs(model_lats4 - kmeans_res_lats[idx]))
    lon_res = np.argmin(np.abs(model_lons4 - kmeans_res_lons[idx]))
    model_map_labels4[lat_res, lon_res] = kmeans_labels[idx]

    # TRMM
    lat_res = np.argmin(np.abs(obs_lats - kmeans_res_lats[idx]))
    lon_res = np.argmin(np.abs(obs_lons - kmeans_res_lons[idx]))
    obs_map_labels[lat_res, lon_res] = kmeans_labels[idx]

lonout, latout = np.meshgrid(model_lons1, model_lats1)
model_map_labels1 = basemap.interp(obs_map_labels[:, :], obs_lons, obs_lats, lonout, latout, order=0)

# lonout, latout = np.meshgrid(model_lons2, model_lats2)
# model_map_labels2 = basemap.interp(obs_map_labels[:, :], obs_lons, obs_lats, lonout, latout, order=0)
#
# lonout, latout = np.meshgrid(model_lons3, model_lats3)
# model_map_labels3 = basemap.interp(obs_map_labels[:, :], obs_lons, obs_lats, lonout, latout, order=0)
#
# lonout, latout = np.meshgrid(model_lons4, model_lats4)
# model_map_labels4 = basemap.interp(obs_map_labels[:, :], obs_lons, obs_lats, lonout, latout, order=0)

lonout, latout = np.meshgrid(cordex_lons1, cordex_lats1)
cordex_map_labels1 = basemap.interp(obs_map_labels[:, :], obs_lons, obs_lats, lonout, latout, order=0)

lonout, latout = np.meshgrid(cordex_lons2, cordex_lats2)
cordex_map_labels2 = basemap.interp(obs_map_labels[:, :], obs_lons, obs_lats, lonout, latout, order=0)

lonout, latout = np.meshgrid(cordex_lons3, cordex_lats3)
cordex_map_labels3 = basemap.interp(obs_map_labels[:, :], obs_lons, obs_lats, lonout, latout, order=0)

lonout, latout = np.meshgrid(cordex_lons4, cordex_lats4)
cordex_map_labels4 = basemap.interp(obs_map_labels[:, :], obs_lons, obs_lats, lonout, latout, order=0)

print(model_map_labels2)

# plot label maps and confirm the projection
plot_data = [model_map_labels1, model_map_labels2, model_map_labels3, model_map_labels4,
             cordex_map_labels1, cordex_map_labels2, cordex_map_labels3, cordex_map_labels4, obs_map_labels]
plot_lats = [model_lats1, model_lats2, model_lats3, model_lats4,
             cordex_lats1, cordex_lats2, cordex_lats3, cordex_lats4, obs_lats]
plot_lons = [model_lons1, model_lons2, model_lons3, model_lons4,
             cordex_lons1, cordex_lons2, cordex_lons3, cordex_lons4, obs_lons]

plot_label_map(plot_data, plot_lats, plot_lons, legends)


for icluster in range(ncluster):
    # select data for different cluster
    # cesm
    maps_res_3d = np.broadcast_to(model_map_labels1 == icluster, model_var1.shape)
    model_ts1 = model_var1[maps_res_3d]

    maps_res_3d = np.broadcast_to(model_map_labels2 == icluster, model_var2.shape)
    model_ts2 = model_var2[maps_res_3d]

    maps_res_3d = np.broadcast_to(model_map_labels3 == icluster, model_var3.shape)
    model_ts3 = model_var3[maps_res_3d]

    maps_res_3d = np.broadcast_to(model_map_labels4 == icluster, model_var4.shape)
    model_ts4 = model_var4[maps_res_3d]

    # cordex
    maps_res_3d = np.broadcast_to(cordex_map_labels1 == icluster, cordex_var1.shape)
    cordex_ts1 = cordex_var1[maps_res_3d]

    maps_res_3d = np.broadcast_to(cordex_map_labels2 == icluster, cordex_var2.shape)
    cordex_ts2 = cordex_var2[maps_res_3d]

    maps_res_3d = np.broadcast_to(cordex_map_labels3 == icluster, cordex_var3.shape)
    cordex_ts3 = cordex_var3[maps_res_3d]

    maps_res_3d = np.broadcast_to(cordex_map_labels4 == icluster, cordex_var4.shape)
    cordex_ts4 = cordex_var4[maps_res_3d]

    # TRMM
    maps_res_3d = np.broadcast_to(obs_map_labels == icluster, obs_var.shape)
    obs_ts = obs_var[maps_res_3d]

    # clean out nan data at the coastal region
    model_ts1 = model_ts1[~np.isnan(model_ts1)]
    model_ts2 = model_ts2[~np.isnan(model_ts2)]
    model_ts3 = model_ts3[~np.isnan(model_ts3)]
    model_ts4 = model_ts4[~np.isnan(model_ts4)]

    cordex_ts1 = cordex_ts1[~np.isnan(cordex_ts1)]
    cordex_ts2 = cordex_ts2[~np.isnan(cordex_ts2)]
    cordex_ts3 = cordex_ts3[~np.isnan(cordex_ts3)]
    cordex_ts4 = cordex_ts4[~np.isnan(cordex_ts4)]

    obs_ts = obs_ts[~np.isnan(obs_ts)]

    # plot the histogram for precip over threshold
    binmax = np.amax(obs_ts[~np.isnan(obs_ts)])
    binarray = np.arange(premin, binmax, (binmax-premin)/50)

    print('Plot the histogram of precip over threshold for model and TRMM')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_data = [model_ts1, model_ts2, model_ts3, model_ts4, cordex_ts1, cordex_ts2, cordex_ts3, cordex_ts4, obs_ts]
    for ii in range(9):
        tempdata = plot_data[ii]
        y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarray, density=True)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        # print(bincenters)
        plt.plot(bincenters, y, c=colors[ii], linestyle=line_types[ii], linewidth=1.5, label=legends[ii])

    plt.yscale('log')
    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("Days")
    plt.xlabel("Precip(mm/day)")

    title = str(iniyear)+' to '+str(endyear)+' Group '+str(icluster) + \
        '  distribution of precip over '+str(premin)+'mm/day'
    fname = 'vrcesm_prect_over_'+str(premin)+'mm_hist_vs_cordex_refTRMM_group_'+str(icluster)+'.pdf'

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

    # test the mean residual life plot for SA-OBS
    obs_max = np.amax(obs_ts)
    test_thresholds = np.arange(0.01 * obs_max/2, 1. * obs_max/2, 2.)
    print(test_thresholds)

    # create array for mean excess and 95% confidence bounds
    test_me = np.zeros(len(test_thresholds))
    test_meup = np.zeros(len(test_thresholds))
    test_mebot = np.zeros(len(test_thresholds))
    test_menum = np.zeros(len(test_thresholds))
    test_c = np.zeros(len(test_thresholds))
    test_sigma = np.zeros(len(test_thresholds))

    for ii, ithres in enumerate(test_thresholds):
        # print('current threshold: '+str(ithres))
        obs_ts_sub = obs_ts[obs_ts > ithres] - ithres
        test_len = len(obs_ts_sub)
        test_me[ii] = np.mean(obs_ts_sub)
        test_std = np.std(obs_ts_sub)
        test_menum[ii] = len(obs_ts_sub)
        test_meup[ii] = test_me[ii] + 1.96*(test_std/np.sqrt(test_len))
        test_mebot[ii] = test_me[ii] - 1.96*(test_std/np.sqrt(test_len))

        # data1_fit = gpd.fit(obs_ts_sub)
        # data1_fit = gpd.fit(obs_ts_sub)
        #
        # print(data1_fit)

        obs_fit = gpdfit_moment(obs_ts_sub)
        # print(obs_fit)
        tempscore = ss.kstest(obs_ts_sub, 'genpareto', args=[obs_fit[0], 0., obs_fit[1]], alternative='two-sided')
        # print(tempscore)
        test_c[ii] = obs_fit[0]
        test_sigma[ii] = obs_fit[1]

    print(test_thresholds[(np.abs(test_menum - 500)).argmin()])
    print(ss.percentileofscore(obs_ts, test_thresholds[(np.abs(test_menum - 500)).argmin()]))
    obs_percent = np.percentile(obs_ts, percentile)

    # mean residual life
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(test_thresholds, test_me, linestyle='solid', linewidth=0.7, c='k')
    plt.plot(test_thresholds, test_meup, linestyle='dashed', linewidth=0.7, c='k')
    plt.plot(test_thresholds, test_mebot, linestyle='dashed', linewidth=0.7, c='k')

    plt.ylabel("Mean Excess")
    plt.xlabel("Thresholds")
    ax.set_title(str(percentile)+"th percentile: "+str(obs_percent)+",  500 obs needs threshold below: " +
                 str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=8)

    title = 'Mean residual life plot for TRMM in group: '+str(icluster)
    plt.suptitle(title, fontsize=10, y=0.95)

    fname = 'TRMM_prect_'+str(percentile)+'th_mean_residual_life_plot_group_'+str(icluster)+'.pdf'
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

    # c
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(test_thresholds, test_c, linestyle='solid', linewidth=1.5, c='k')

    plt.ylabel('Xi')
    plt.xlabel("Thresholds")
    ax.set_title(str(percentile)+"th percentile: "+str(obs_percent)+",  500 obs needs threshold below: " +
                 str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=8)

    title = 'Estimated parameter (Xi) for TRMM in group: '+str(icluster)
    plt.suptitle(title, fontsize=10, y=0.95)

    fname = 'TRMM_prect_'+str(percentile)+'th_xi_plot_group_'+str(icluster)+'.pdf'
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

    # sigma
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(test_thresholds, test_sigma, linestyle='solid', linewidth=1.5, c='k')

    plt.ylabel('Xi')
    plt.xlabel("Thresholds")
    ax.set_title(str(percentile)+"th percentile: "+str(obs_percent)+",  500 obs needs threshold below: " +
                 str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=8)

    title = 'Estimated parameter (sigma) for TRMM in group: '+str(icluster)
    plt.suptitle(title, fontsize=10, y=0.95)

    fname = 'TRMM_prect_'+str(percentile)+'th_sigma_plot_group_'+str(icluster)+'.pdf'
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

    # use gpd fit models
    model_percent1 = np.percentile(model_ts1, percentile)
    model_percent2 = np.percentile(model_ts2, percentile)
    model_percent3 = np.percentile(model_ts3, percentile)
    model_percent4 = np.percentile(model_ts4, percentile)

    cordex_percent1 = np.percentile(cordex_ts1, percentile)
    cordex_percent2 = np.percentile(cordex_ts2, percentile)
    cordex_percent3 = np.percentile(cordex_ts3, percentile)
    cordex_percent4 = np.percentile(cordex_ts4, percentile)

    obs_percent = np.percentile(obs_ts, percentile)

    print([model_percent1, model_percent2, model_percent3, model_percent4, cordex_percent1,
           cordex_percent2, cordex_percent3, cordex_percent4, obs_percent])

    model_ts1_sub = model_ts1[model_ts1 > model_percent1] - model_percent1
    model_ts2_sub = model_ts2[model_ts2 > model_percent2] - model_percent2
    model_ts3_sub = model_ts3[model_ts3 > model_percent3] - model_percent3
    model_ts4_sub = model_ts4[model_ts4 > model_percent4] - model_percent4

    cordex_ts1_sub = cordex_ts1[cordex_ts1 > cordex_percent1] - cordex_percent1
    cordex_ts2_sub = cordex_ts2[cordex_ts2 > cordex_percent2] - cordex_percent2
    cordex_ts3_sub = cordex_ts3[cordex_ts3 > cordex_percent3] - cordex_percent3
    cordex_ts4_sub = cordex_ts4[cordex_ts4 > cordex_percent4] - cordex_percent4

    obs_ts_sub = obs_ts[obs_ts > obs_percent] - obs_percent

    # fit data
    model_fit1 = gpdfit_moment(model_ts1_sub)
    model_fit2 = gpdfit_moment(model_ts2_sub)
    model_fit3 = gpdfit_moment(model_ts3_sub)
    model_fit4 = gpdfit_moment(model_ts4_sub)

    cordex_fit1 = gpdfit_moment(cordex_ts1_sub)
    cordex_fit2 = gpdfit_moment(cordex_ts2_sub)
    cordex_fit3 = gpdfit_moment(cordex_ts3_sub)
    cordex_fit4 = gpdfit_moment(cordex_ts4_sub)

    obs_fit = gpdfit_moment(obs_ts_sub)

    print('Group'+str(icluster)+' CESM GPD fitting results:')
    print(model_fit1)
    print(model_fit2)
    print(model_fit3)
    print(model_fit4)

    print('Group'+str(icluster)+' CORDEX GPD fitting results:')
    print(cordex_fit1)
    print(cordex_fit2)
    print(cordex_fit3)
    print(cordex_fit4)

    print('Group'+str(icluster)+' TRMM GPD fitting results:')
    print(obs_fit)

    # check the goodness of fitting
    model_ks1 = ss.kstest(model_ts1_sub, 'genpareto', args=[model_fit1[0], 0., model_fit1[1]], alternative='two-sided')
    model_ks2 = ss.kstest(model_ts2_sub, 'genpareto', args=[model_fit2[0], 0., model_fit2[1]], alternative='two-sided')
    model_ks3 = ss.kstest(model_ts3_sub, 'genpareto', args=[model_fit3[0], 0., model_fit3[1]], alternative='two-sided')
    model_ks4 = ss.kstest(model_ts4_sub, 'genpareto', args=[model_fit4[0], 0., model_fit4[1]], alternative='two-sided')

    cordex_ks1 = ss.kstest(cordex_ts1_sub, 'genpareto', args=[
                           cordex_fit1[0], 0., cordex_fit1[1]], alternative='two-sided')
    cordex_ks2 = ss.kstest(cordex_ts2_sub, 'genpareto', args=[
                           cordex_fit2[0], 0., cordex_fit2[1]], alternative='two-sided')
    cordex_ks3 = ss.kstest(cordex_ts3_sub, 'genpareto', args=[
                           cordex_fit3[0], 0., cordex_fit3[1]], alternative='two-sided')
    cordex_ks4 = ss.kstest(cordex_ts4_sub, 'genpareto', args=[
                           cordex_fit4[0], 0., cordex_fit4[1]], alternative='two-sided')

    obs_ks = ss.kstest(obs_ts_sub, 'genpareto', args=[obs_fit[0], 0., obs_fit[1]], alternative='two-sided')

    plot_data = [model_ts1_sub, model_ts2_sub, model_ts3_sub, model_ts4_sub,
                 cordex_ts1_sub, cordex_ts2_sub, cordex_ts3_sub, cordex_ts4_sub, obs_ts_sub]
    plot_fit = [model_fit1, model_fit2, model_fit3, model_fit4,
                cordex_fit1, cordex_fit2, cordex_fit3, cordex_fit4, obs_fit]
    plot_ks = [model_ks1, model_ks2, model_ks3, model_ks4, cordex_ks1, cordex_ks2, cordex_ks3, cordex_ks4, obs_ks]

    plot_fitcheck(plot_data, plot_fit, plot_ks, legends, icluster)

    # calculate N-year return level
    year_returns = np.arange(1., 151., 1.)

    model_return1 = np.zeros(len(year_returns))
    model_return2 = np.zeros(len(year_returns))
    model_return3 = np.zeros(len(year_returns))
    model_return4 = np.zeros(len(year_returns))

    cordex_return1 = np.zeros(len(year_returns))
    cordex_return2 = np.zeros(len(year_returns))
    cordex_return3 = np.zeros(len(year_returns))
    cordex_return4 = np.zeros(len(year_returns))

    obs_return = np.zeros(len(year_returns))

    for ii, iyear in enumerate(year_returns):
        m = iyear * 365.
        obsrate = (100.-percentile)/100.

        model_return1[ii] = model_percent1 + model_fit1[1] / model_fit1[0] * ((m*obsrate) ** model_fit1[0] - 1)
        model_return2[ii] = model_percent2 + model_fit2[1] / model_fit2[0] * ((m*obsrate) ** model_fit2[0] - 1)
        model_return3[ii] = model_percent3 + model_fit3[1] / model_fit3[0] * ((m*obsrate) ** model_fit3[0] - 1)
        model_return4[ii] = model_percent4 + model_fit4[1] / model_fit4[0] * ((m*obsrate) ** model_fit4[0] - 1)

        cordex_return1[ii] = cordex_percent1 + cordex_fit1[1] / cordex_fit1[0] * ((m*obsrate) ** cordex_fit1[0] - 1)
        cordex_return2[ii] = cordex_percent2 + cordex_fit2[1] / cordex_fit2[0] * ((m*obsrate) ** cordex_fit2[0] - 1)
        cordex_return3[ii] = cordex_percent3 + cordex_fit3[1] / cordex_fit3[0] * ((m*obsrate) ** cordex_fit3[0] - 1)
        cordex_return4[ii] = cordex_percent4 + cordex_fit4[1] / cordex_fit4[0] * ((m*obsrate) ** cordex_fit4[0] - 1)

        obs_return[ii] = obs_percent + obs_fit[1] / obs_fit[0] * ((m*obsrate) ** obs_fit[0] - 1)

    print('plotting return levels for the gourp'+str(icluster)+'...')
    plot_data = [model_return1, model_return2, model_return3, model_return4,
                 cordex_return1, cordex_return2, cordex_return3, cordex_return4, obs_return]

    xlabel = 'Return years'
    ylabel = 'Precip(mm/day)'
    title = 'Group '+str(icluster)+' Precip extreme return levels'
    fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels_group_'+str(icluster)+'.pdf'

    plot_lines(year_returns, plot_data, colors, line_types, legends, xlabel, ylabel, title, outdir+fname)

    xlabel = 'TRMM Return years'
    ylabel = 'Model Return years'
    title = 'Group '+str(icluster)+' Precip extreme return levels vs TRMM'
    fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels_refTRMM_group_'+str(icluster)+'.pdf'
    plot_lines(obs_return, plot_data, colors, line_types, legends, xlabel, ylabel, title, outdir+fname)
