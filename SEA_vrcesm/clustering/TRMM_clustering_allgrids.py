# This script is used to  separate the stations in SA-OBS into 3 clusters
# Several steps are inplemented:
# S1-read precip data from SA-OBS
# S2-clean data and PCA
# S3-cluster the data
#
# Written by Harry Li

# import libraries
import numpy as np
import pathmagic  # noqa: F401
from modules.stats.mod_stats_clustering import kmeans_cluster
from modules.datareader.mod_dataread_obs_TRMM import readobs_pre_TRMM_day
from modules.stats.mod_stats_clim import mon2clim
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, scale
from sklearn.cluster import KMeans
plt.switch_backend('agg')


############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/obs/TRMM/clustering/allgrids/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Total Precip'
varstr = 'precipitation'
var_unit = 'mm/day'

# time bounds
iniyear = 1998
endyear = 2005
nyears = endyear - iniyear + 1
yearts = np.arange(iniyear, endyear+1)
# yearts    = np.delete(yearts,9,None)
print(yearts)

# define regions
latbounds = [5, 30]
lonbounds = [90, 115]

# mainland Southeast Asia
reg_lats = [10, 24]
reg_lons = [98, 110]

# set data frequency
frequency = 'day'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# color for clusters
colors = ['red', 'blue', 'green', 'magenta']

# days in each month
mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mindx = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

############################################################################
# read data
############################################################################

print('Reading TRMM data...')

# read TRMM
var, time, lats, lons = readobs_pre_TRMM_day('precipitation', iniyear, endyear, latbounds, lonbounds, oceanmask=0)

############################################################################
# clustering
############################################################################
lat_1 = np.argmin(np.abs(lats - reg_lats[0]))
lat_2 = np.argmin(np.abs(lats - reg_lats[1]))
lon_1 = np.argmin(np.abs(lons - reg_lons[0]))
lon_2 = np.argmin(np.abs(lons - reg_lons[1]))

dataset_var = var[:, lat_1:lat_2+1, lon_1:lon_2+1]
temp_var = dataset_var.reshape((dataset_var.shape[0], -1))
print(temp_var.shape)
temp_var = np.transpose(temp_var, (1, 0))

fname = 'TRMM_precip_kmeans_nclusters'
ncluster = 3
# res_spectral = spectral_cluster(dataset_var, ncluster, stnlats, stnlons, stnnames, outdir+fname)
# res = kmeans_cluster(dataset_var, ncluster, stnlats, stnlons, stnnames, outdir+fname, map_plot=False)

nclusters_list = np.arange(2, 10)
Jcost = []
res = []

# use pca to compress data
# temp_var = scale(temp_var)
# pca = PCA().fit(temp_var)
# print(pca.explained_variance_ratio_)
# print(np.sum(pca.explained_variance_ratio_))
# sum = 0
# k = 0
# while sum < 0.99:
#     sum += pca.explained_variance_ratio_[k]
#     k += 1
# print(k)

for icluster in nclusters_list:
    print('run kMeans algorithm for '+str(icluster)+' clusters')

    # use pca as first guess
    pca = PCA(n_components=icluster).fit(temp_var)
    kmeans = KMeans(n_clusters=icluster, init=pca.components_, n_init=1, max_iter=300).fit(temp_var)

    # use pca to compress data
    # pca_data = PCA(n_components=k).fit_transform(temp_var)
    # print(pca_data.shape)
    # kmeans = KMeans(n_clusters=icluster, init='k-means++', n_init=100, max_iter=300).fit(pca_data)

    print(kmeans.labels_)
    Jcost.append(kmeans.inertia_)

    res.append(kmeans)

    # plot the map
    temp_label = kmeans.labels_
    temp_label = temp_label.reshape((lat_2-lat_1+1, lon_2-lon_1+1))

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

    maps_res = np.empty((len(lats), len(lons)))
    maps_res[:] = np.nan
    maps_res[lat_1:lat_2+1, lon_1:lon_2+1] = temp_label

    mlons, mlats = np.meshgrid(lons, lats)
    x, y = map(mlons, mlats)

    # plot the contour
    cs = map.contourf(x, y, maps_res, levels=np.arange(-0.5, icluster),
                      cmap=plt.cm.get_cmap('viridis', icluster), alpha=0.9)

    # add colorbar
    fig.subplots_adjust(bottom=0.22, wspace=0.2, hspace=0.2)
    cbar_ax = fig.add_axes([0.25, 0.17, 0.5, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal', ticks=range(icluster))
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('Cluster labels')
    plt.clim(-0.5, icluster-0.5)

    title = 'K-means clustering: '+str(icluster)+' groups'
    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+fname+'_'+str(icluster)+'.pdf', bbox_inches='tight', dpi=3000)
    plt.close(fig)

# plot to choose best cluster numbers
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(nclusters_list, Jcost, c='k', linewidth=1.5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')

title = 'K-means clustering'
plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname+'.pdf', bbox_inches='tight', dpi=3000)
plt.close(fig)

for idx in range(len(res)):
    temp = res[idx]
    print('number of clusterings: '+str(idx+2))
    print(temp.labels_)

############################################################################
# plot for the different groups
############################################################################

index = np.arange(9)
bar_width = 0.8
opacity = 0.8
shape_type = ['', '', '', '', '..', '..', '..', '..', '//']

cluster_labels = res[ncluster-2].labels_
cluster_labels = cluster_labels.reshape((lat_2-lat_1+1, lon_2-lon_1+1))
# cluster_labels = np.array([0,0,1,2,0,0,0,0,0,0,1,0,2,0,2,1,0,0,2,1,1,0])
print('Current number of clusterings is: '+str(ncluster))

dataset_monmean = np.ma.zeros((nyears*12, lat_2-lat_1+1, lon_2-lon_1+1))
for iyear in np.arange(nyears):
    for imon in range(12):
        dataset_monmean[iyear*12+imon, :,
                        :] = np.ma.mean(dataset_var[iyear*365 + mindx[imon]: iyear*365 + mindx[imon + 1], :, :], axis=0)

print(dataset_monmean)

# ############################################################################
# # plot for monthly mean TS
# ############################################################################
#
# monthts = np.arange((endyear-iniyear+1)*12) + 1
# xlabel = 'Month'
# ylabel = 'Precip (mm/day)'
# xticks = np.arange(6, (endyear-iniyear+1)*12, 12)
# xticknames = [str(iyear) for iyear in yearts]
#
# # plot monthly mean ts for each group
# for idx in range(ncluster):
#     groupname = 'group'+str(idx)
#
#     print('plot mothly mean ts for ' + groupname)
#
#     title = str(iniyear)+' to '+str(endyear)+' TRMM Monthly mean precip TS in the group: '+str(idx)
#     fname = 'TRMM_prect_monthly_mean_ts_group_'+str(idx)+'.pdf'
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for iname in tempnames[:-1]:
#         plt.plot(monthts, dataset_monmean[iname], color=colors[idx], marker='o',
#                  markersize=1, linestyle='-', linewidth=1.5, label=iname, alpha=0.4)
#     plt.plot(monthts, dataset_monmean[groupname], color=colors[idx], marker='o',
#              markersize=1, linestyle='-', linewidth=1.5, label=groupname)
#
#     plt.legend(handlelength=4, fontsize=5)
#     plt.xticks(xticks, xticknames, fontsize=6)
#     plt.yticks(fontsize=6)
#     plt.xlabel(xlabel, fontsize=8)
#     plt.ylabel(ylabel, fontsize=8)
#
#     plt.suptitle(title, fontsize=9, y=0.95)
#     plt.savefig(outdir+fname, bbox_inches='tight')
#     plt.close(fig)
#
# # plot monthly mean ts for all groups
# print('plot mothly mean ts for all groups')
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# title = str(iniyear)+' to '+str(endyear)+' SA-OBS Monthly mean precip TS'
# fname = 'SAOBS_prect_monthly_mean_ts_groupinone.pdf'
#
# for idx in range(len(stnnames)):
#     iname = stnnames[idx]
#     plt.plot(monthts, dataset_monmean[iname], color=colors[cluster_labels[idx]], marker='o',
#              markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)
# lines = []
# legendnames = []
# for idx in range(ncluster):
#     iname = 'group'+str(idx)
#     legendnames.append(iname)
#     lines += plt.plot(monthts, dataset_monmean[iname], color=colors[idx], marker='o',
#                       markersize=1, linestyle='-', linewidth=1.5, label=iname)
#
# plt.legend(lines, legendnames, handlelength=4, fontsize=5)
# plt.xticks(xticks, xticknames, fontsize=6)
# plt.yticks(fontsize=6)
# plt.xlabel(xlabel, fontsize=8)
# plt.ylabel(ylabel, fontsize=8)
#
# plt.suptitle(title, fontsize=9, y=0.95)
# plt.savefig(outdir+fname, bbox_inches='tight')
# plt.close(fig)
#
#
# ############################################################################
# # plot for annual mean TS
# ############################################################################
# xlabel = 'Year'
# ylabel = 'Precip (mm/day)'
#
# # plot annual mean ts for each group
# for idx in range(ncluster):
#     groupname = 'group'+str(idx)
#     tempnames = stnnames[cluster_labels == idx]
#
#     tempnames = np.append(tempnames, groupname)
#
#     # print(dataset_monmean[tempnames])
#
#     print('plot annual mean ts for ' + groupname)
#
#     title = str(iniyear)+' to '+str(endyear)+' SA-OBS Annual mean precip TS in the group: '+str(idx)
#     fname = 'SAOBS_prect_annual_mean_ts_group_'+str(idx)+'.pdf'
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for iname in tempnames[:-1]:
#         plt.plot(yearts, dataset_annmean[iname], color=colors[idx], marker='o',
#                  markersize=1, linestyle='-', linewidth=1.5, label=iname, alpha=0.4)
#     plt.plot(yearts, dataset_annmean[groupname], color=colors[idx], marker='o',
#              markersize=1, linestyle='-', linewidth=1.5, label=groupname)
#
#     plt.legend(handlelength=4, fontsize=5)
#     plt.xticks(fontsize=6)
#     plt.yticks(fontsize=6)
#     plt.xlabel(xlabel, fontsize=8)
#     plt.ylabel(ylabel, fontsize=8)
#
#     plt.suptitle(title, fontsize=9, y=0.95)
#     plt.savefig(outdir+fname, bbox_inches='tight')
#     plt.close(fig)
#
# # plot annual mean ts for all groups
# print('plot annual mean ts for all groups')
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# title = str(iniyear)+' to '+str(endyear)+' SA-OBS Annual mean precip TS'
# fname = 'SAOBS_prect_annual_mean_ts_groupinone.pdf'
#
# for idx in range(len(stnnames)):
#     iname = stnnames[idx]
#     plt.plot(yearts, dataset_annmean[iname], color=colors[cluster_labels[idx]], marker='o',
#              markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)
# lines = []
# legendnames = []
# for idx in range(ncluster):
#     iname = 'group'+str(idx)
#     legendnames.append(iname)
#     lines += plt.plot(yearts, dataset_annmean[iname], color=colors[idx], marker='o',
#                       markersize=1, linestyle='-', linewidth=1.5, label=iname)
#
# plt.legend(lines, legendnames, handlelength=4, fontsize=5)
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.xlabel(xlabel, fontsize=8)
# plt.ylabel(ylabel, fontsize=8)
#
# plt.suptitle(title, fontsize=9, y=0.95)
# plt.savefig(outdir+fname, bbox_inches='tight')
# plt.close(fig)
#
# ############################################################################
# # plot for histogram
# ############################################################################
# xlabel = 'Precip (mm/day)'
# ylabel = 'Frequency'
#
# # plot annual mean ts for each group
# binarrays = []
# for idx in range(ncluster):
#     groupname = 'group'+str(idx)
#     tempnames = stnnames[cluster_labels == idx]
#
#     # print(dataset_monmean[tempnames])
#
#     print('plot histogram for ' + groupname)
#
#     tempdata = dataset[tempnames].values
#     binmax = np.amax(tempdata[~np.isnan(tempdata)])*2./3.
#     binarray = np.arange(0, binmax, binmax/20)
#     binarrays.append(binarray)
#
#     title = str(iniyear)+' to '+str(endyear)+' SA-OBS Total precip distribution in the group: '+str(idx)
#     fname = 'SAOBS_prect_hist_group_'+str(idx)+'.pdf'
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for iname in tempnames:
#         tempdata = dataset[iname]
#         y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarray, density=True)
#         bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#         plt.plot(bincenters, y, c=colors[idx], linestyle='-', linewidth=1.5, label=iname, alpha=0.4)
#
#     tempdata = dataset[tempnames].values
#     y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarray, density=True)
#     bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#     plt.plot(bincenters, y, c=colors[idx], linestyle='-', linewidth=1.5, label=groupname)
#
#     plt.yscale('log')
#     plt.legend(handlelength=4, fontsize=5)
#     plt.xticks(fontsize=6)
#     plt.yticks(fontsize=6)
#     plt.xlabel(xlabel, fontsize=8)
#     plt.ylabel(ylabel, fontsize=8)
#
#     plt.suptitle(title, fontsize=9, y=0.95)
#     plt.savefig(outdir+fname, bbox_inches='tight')
#     plt.close(fig)
#
# # plot annual mean ts for all groups
# print('plot histogram for all groups')
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# title = str(iniyear)+' to '+str(endyear)+' SA-OBS Total precip distribution'
# fname = 'SAOBS_prect_hist_groupinone.pdf'
#
# for idx in range(len(stnnames)):
#     iname = stnnames[idx]
#     tempdata = dataset[iname]
#     y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarrays[cluster_labels[idx]], density=True)
#     bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#     plt.plot(bincenters, y, c=colors[cluster_labels[idx]], linestyle='-', linewidth=1.5, label=iname, alpha=0.4)
# lines = []
# legendnames = []
# for idx in range(ncluster):
#     iname = 'group'+str(idx)
#     legendnames.append(iname)
#     tempnames = stnnames[cluster_labels == idx]
#     tempdata = dataset[tempnames]
#     y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarrays[idx], density=True)
#     bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#     lines += plt.plot(bincenters, y, c=colors[idx], linestyle='-', linewidth=1.5, label=iname)
#
# plt.yscale('log')
# plt.legend(lines, legendnames, handlelength=4, fontsize=5)
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.xlabel(xlabel, fontsize=8)
# plt.ylabel(ylabel, fontsize=8)
#
# plt.suptitle(title, fontsize=9, y=0.95)
# plt.savefig(outdir+fname, bbox_inches='tight')
# plt.close(fig)
#
# ############################################################################
# # plot for long term mean climatology
# ############################################################################
# xlabel = 'Month'
# ylabel = 'Precip (mm/day)'
#
# # plot climatology for each group
# for idx in range(ncluster):
#     groupname = 'group'+str(idx)
#     tempnames = stnnames[cluster_labels == idx]
#     tempnames = np.append(tempnames, groupname)
#
#     # print(dataset_monmean[tempnames])
#
#     print('plot climatology for ' + groupname)
#
#     title = str(iniyear)+' to '+str(endyear)+' SA-OBS Seasonal cycle of precip in the group: '+str(idx)
#     fname = 'SAOBS_prect_clim_mean_line_group_'+str(idx)+'.pdf'
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for iname in tempnames[:-1]:
#         tempdata = dataset_monmean[iname]
#         temp_mean, temp_std = mon2clim(tempdata)
#         plt.plot(months, temp_mean, color=colors[idx], marker='o', markersize=1,
#                  linestyle='-', linewidth=1.5, label=iname, alpha=0.4)
#     tempdata = dataset_monmean[groupname]
#     temp_mean, temp_std = mon2clim(tempdata)
#     plt.plot(months, temp_mean, color=colors[idx], marker='o',
#              markersize=1, linestyle='-', linewidth=1.5, label=groupname)
#
#     plt.legend(handlelength=4, fontsize=5)
#     plt.xticks(fontsize=6)
#     plt.yticks(fontsize=6)
#     plt.xlabel(xlabel, fontsize=8)
#     plt.ylabel(ylabel, fontsize=8)
#
#     plt.suptitle(title, fontsize=9, y=0.95)
#     plt.savefig(outdir+fname, bbox_inches='tight')
#     plt.close(fig)
#
# # plot annual mean ts for all groups
# print('plot annual mean ts for all groups')
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# title = str(iniyear)+' to '+str(endyear)+' SA-OBS Seasonal cycle of precip'
# fname = 'SAOBS_prect_clim_mean_line_groupinone.pdf'
#
# for idx in range(len(stnnames)):
#     iname = stnnames[idx]
#     tempdata = dataset_monmean[iname]
#     temp_mean, temp_std = mon2clim(tempdata)
#     plt.plot(months, temp_mean, color=colors[cluster_labels[idx]], marker='o',
#              markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)
# lines = []
# legendnames = []
# for idx in range(ncluster):
#     iname = 'group'+str(idx)
#     legendnames.append(iname)
#     tempdata = dataset_monmean[iname]
#     temp_mean, temp_std = mon2clim(tempdata)
#     lines += plt.plot(months, temp_mean, color=colors[idx], marker='o',
#                       markersize=1, linestyle='-', linewidth=1.5, label=iname)
#
# plt.legend(lines, legendnames, handlelength=4, fontsize=5)
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=6)
# plt.xlabel(xlabel, fontsize=8)
# plt.ylabel(ylabel, fontsize=8)
#
# plt.suptitle(title, fontsize=9, y=0.95)
# plt.savefig(outdir+fname, bbox_inches='tight')
# plt.close(fig)
