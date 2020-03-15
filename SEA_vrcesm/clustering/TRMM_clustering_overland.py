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
import pickle
plt.switch_backend('agg')


############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/obs/TRMM/clustering/overland/'

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
reg_lats = [5, 24]
reg_lons = [97, 110]

# set data frequency
frequency = 'day'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# color for clusters
colors = ['red', 'blue', 'green', 'magenta', 'darkslategrey', 'orange']

# days in each month
mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mindx = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

############################################################################
# read data
############################################################################

print('Reading TRMM data...')

# read TRMM
var, time, lats, lons = readobs_pre_TRMM_day('precipitation', iniyear, endyear, latbounds, lonbounds, oceanmask=1)

# print(time)
print(var.shape)
# print(var.mask)
print(time[time.month == 2])

############################################################################
# clustering
############################################################################
lat_1 = np.argmin(np.abs(lats - reg_lats[0]))
lat_2 = np.argmin(np.abs(lats - reg_lats[1]))
lon_1 = np.argmin(np.abs(lons - reg_lons[0]))
lon_2 = np.argmin(np.abs(lons - reg_lons[1]))
print(lat_1, lat_2)
print(lon_1, lon_2)

reg_lats = lats[lat_1: lat_2+1]
reg_lons = lons[lon_1: lon_2+1]

dataset_var = var[:, lat_1:lat_2+1, lon_1:lon_2+1]
samplelons, samplelats = np.meshgrid(lons[lon_1:lon_2+1], lats[lat_1:lat_2+1])
# print(dataset_var[0, :, :].mask)
# print(np.argwhere(np.logical_xor(var[5, :, :].mask, var[0, :, :].mask)))

samplelons[dataset_var[0, :, :].mask] = np.nan
samplelats[dataset_var[0, :, :].mask] = np.nan
dataset_var[dataset_var[:, :, :].mask] = np.nan

print(dataset_var[0, :, :])
print(dataset_var.shape)
print(samplelats.shape)
print(samplelats)
print(samplelons)

temp_var = dataset_var.reshape((dataset_var.shape[0], -1))
print(temp_var.shape)
temp_var = np.transpose(temp_var, (1, 0))
# print(np.argwhere(np.isnan(temp_var)))
temp_var = np.delete(temp_var, np.argwhere(np.isnan(temp_var[:, 0])), 0)
print(temp_var.shape)
# print(np.argwhere(np.isnan(temp_var)))

# create the lat/lon list
temp_lat = samplelats.flatten()
temp_lon = samplelons.flatten()
print(temp_lat.shape)

# get rid of nan
temp_lat = temp_lat[~np.isnan(temp_lat)]
temp_lon = temp_lon[~np.isnan(temp_lon)]
print(temp_lat.shape)

fname = 'TRMM_precip_kmeans_nclusters'
ncluster = 6
outdir2 = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/obs/TRMM/clustering/overland/' + \
    str(ncluster)+' clusters/'
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

    # project the result onto the map using the temp_label and temp_lat/lon
    maps_res = np.empty((len(lats), len(lons)))
    maps_res[:] = np.nan

    for idx in range(len(temp_label)):
        lat_res = np.argmin(np.abs(lats - temp_lat[idx]))
        lon_res = np.argmin(np.abs(lons - temp_lon[idx]))
        maps_res[lat_res, lon_res] = temp_label[idx]

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
    if icluster == ncluster:
        plt.savefig(outdir2+fname+'_'+str(icluster)+'.pdf', bbox_inches='tight', dpi=3000)
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

kmeans_res = {}
for icluster in nclusters_list:
    kmeans_res[str(icluster)+' clusters'] = res[icluster-nclusters_list[0]].labels_
kmeans_res['kmeans_res_lats'] = temp_lat
kmeans_res['kmeans_res_lons'] = temp_lon
kmeans_res['reg_lats'] = reg_lats
kmeans_res['reg_lons'] = reg_lons

############################################################################
# plot for the different groups
############################################################################

index = np.arange(9)
bar_width = 0.8
opacity = 0.8
shape_type = ['', '', '', '', '..', '..', '..', '..', '//']

cluster_labels = res[ncluster-2].labels_
# cluster_labels = cluster_labels.reshape((lat_2-lat_1+1, lon_2-lon_1+1))
# cluster_labels = np.array([0,0,1,2,0,0,0,0,0,0,1,0,2,0,2,1,0,0,2,1,1,0])
print('Current number of clusterings is: '+str(ncluster))

# project the result onto the map using the temp_label and temp_lat/lon
maps_res = np.empty((len(lats), len(lons)))
maps_res[:] = np.nan

for idx in range(len(cluster_labels)):
    lat_res = np.argmin(np.abs(lats - temp_lat[idx]))
    lon_res = np.argmin(np.abs(lons - temp_lon[idx]))
    maps_res[lat_res, lon_res] = cluster_labels[idx]

maps_res = maps_res[lat_1: lat_2 + 1, lon_1: lon_2+1]

# # test the method to select cluster
# maps_res_3d = np.broadcast_to(maps_res == 0, dataset_var.shape)
# temp = np.ma.masked_where(maps_res_3d, dataset_var)
#
# print(np.argwhere(np.logical_xor(maps_res_3d[0, :, :], maps_res == 0)))
# print(maps_res_3d)
# print(dataset_var[maps_res_3d].shape)
# print(temp.shape)

dataset_monmean = np.ma.zeros((nyears*12, lat_2-lat_1+1, lon_2-lon_1+1))
for iyear in np.arange(nyears):
    for imon in range(12):
        dataset_monmean[iyear*12+imon, :, :] = np.ma.mean(
            dataset_var[((time.year == iniyear+iyear) & (time.month == (imon+1))), :, :], axis=0)

# print(dataset_monmean)

dataset_gpmonmean = np.ma.zeros((nyears*12, ncluster))
for idx in range(dataset_monmean.shape[0]):
    for icluster in range(ncluster):
        temp = dataset_monmean[idx, :, :]
        # print(np.argwhere(np.logical_xor(temp.mask, dataset_monmean.mask)))
        # print(temp)
        dataset_gpmonmean[idx, icluster] = np.ma.mean(temp[maps_res == icluster])
        # print(temp[maps_res == icluster].shape)
    # print(np.count_nonzero(~np.isnan(maps_res)))

print(dataset_gpmonmean)
kmeans_res['dataset_gpmonmean'] = dataset_gpmonmean

# ############################################################################
# # plot for monthly mean TS
# ############################################################################

monthts = np.arange((endyear-iniyear+1)*12) + 1
xlabel = 'Month'
ylabel = 'Precip (mm/day)'
xticks = np.arange(6, (endyear-iniyear+1)*12, 12)
xticknames = [str(iyear) for iyear in yearts]

# plot monthly mean ts for each group
for icluster in range(ncluster):
    groupname = 'group'+str(icluster)

    print('plot mothly mean ts for ' + groupname)

    title = str(iniyear)+' to '+str(endyear)+' TRMM Monthly mean precip TS in the group: '+str(icluster)
    fname = 'TRMM_prect_monthly_mean_ts_group_'+str(icluster)+'.png'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx in range(len(cluster_labels)):
        if cluster_labels[idx] == icluster:
            lat_res = np.argmin(np.abs(reg_lats - temp_lat[idx]))
            lon_res = np.argmin(np.abs(reg_lons - temp_lon[idx]))
            plt.plot(monthts, dataset_monmean[:, lat_res, lon_res], color='grey', marker='o',
                     markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)

    plt.plot(monthts, dataset_gpmonmean[:, icluster], color=colors[icluster], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, label=groupname)

    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(xticks, xticknames, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir2+fname, bbox_inches='tight', dpi=1000)
    plt.close(fig)

# plot monthly mean ts for all groups
print('plot mothly mean ts for all groups')
fig = plt.figure()
ax = fig.add_subplot(111)

title = str(iniyear)+' to '+str(endyear)+' TRMM Monthly mean precip TS'
fname = 'TRMM_prect_monthly_mean_ts_groupinone.png'

for idx in range(len(cluster_labels)):
    lat_res = np.argmin(np.abs(reg_lats - temp_lat[idx]))
    lon_res = np.argmin(np.abs(reg_lons - temp_lon[idx]))
    plt.plot(monthts, dataset_monmean[:, lat_res, lon_res], color=colors[cluster_labels[idx]], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)
lines = []
legendnames = []
for icluster in range(ncluster):
    iname = 'group'+str(icluster)
    legendnames.append(iname)
    lines += plt.plot(monthts, dataset_gpmonmean[:, icluster], color=colors[icluster], marker='o',
                      markersize=1, linestyle='-', linewidth=1.5, label=iname)

plt.legend(lines, legendnames, handlelength=4, fontsize=5)
plt.xticks(xticks, xticknames, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(xlabel, fontsize=8)
plt.ylabel(ylabel, fontsize=8)

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir2+fname, bbox_inches='tight', dpi=1000)
plt.close(fig)


# ############################################################################
# # plot for annual mean TS
# ############################################################################
xlabel = 'Year'
ylabel = 'Precip (mm/day)'

dataset_annmean = np.ma.zeros((nyears, lat_2-lat_1+1, lon_2-lon_1+1))
for iyear in np.arange(nyears):
    dataset_annmean[iyear, :, :] = np.ma.mean(dataset_var[(time.year == iniyear+iyear), :, :], axis=0)

# print(dataset_monmean)

dataset_gpannmean = np.ma.zeros((nyears, ncluster))
for idx in range(dataset_annmean.shape[0]):
    for icluster in range(ncluster):
        temp = dataset_annmean[idx, :, :]
        dataset_gpannmean[idx, icluster] = np.ma.mean(temp[maps_res == icluster])

print(dataset_gpannmean)
kmeans_res['dataset_gpannmean'] = dataset_gpannmean

# plot annual mean ts for each group
for icluster in range(ncluster):
    groupname = 'group'+str(icluster)

    print('plot annual mean ts for ' + groupname)

    title = str(iniyear)+' to '+str(endyear)+' TRMM Annual mean precip TS in the group: '+str(icluster)
    fname = 'TRMM_prect_annual_mean_ts_group_'+str(icluster)+'.png'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx in range(len(cluster_labels)):
        if cluster_labels[idx] == icluster:
            lat_res = np.argmin(np.abs(reg_lats - temp_lat[idx]))
            lon_res = np.argmin(np.abs(reg_lons - temp_lon[idx]))
            plt.plot(yearts, dataset_annmean[:, lat_res, lon_res], color='grey', marker='o',
                     markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)

    plt.plot(yearts, dataset_gpannmean[:, icluster], color=colors[icluster], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, label=groupname)

    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir2+fname, bbox_inches='tight', dpi=1000)
    plt.close(fig)

# plot annual mean ts for all groups
print('plot annual mean ts for all groups')
fig = plt.figure()
ax = fig.add_subplot(111)

title = str(iniyear)+' to '+str(endyear)+' TRMM Annual mean precip TS'
fname = 'TRMM_prect_annual_mean_ts_groupinone.png'

for idx in range(len(cluster_labels)):
    lat_res = np.argmin(np.abs(reg_lats - temp_lat[idx]))
    lon_res = np.argmin(np.abs(reg_lons - temp_lon[idx]))
    plt.plot(yearts, dataset_annmean[:, lat_res, lon_res], color=colors[cluster_labels[idx]], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)
lines = []
legendnames = []
for icluster in range(ncluster):
    iname = 'group'+str(icluster)
    legendnames.append(iname)
    lines += plt.plot(yearts, dataset_gpannmean[:, icluster], color=colors[icluster], marker='o',
                      markersize=1, linestyle='-', linewidth=1.5, label=iname)

plt.legend(lines, legendnames, handlelength=4, fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(xlabel, fontsize=8)
plt.ylabel(ylabel, fontsize=8)

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir2+fname, bbox_inches='tight', dpi=1000)
plt.close(fig)

# ############################################################################
# # plot for histogram
# ############################################################################
xlabel = 'Precip (mm/day)'
ylabel = 'Frequency'

# plot annual mean ts for each group
binarrays = []
for icluster in range(ncluster):
    groupname = 'group'+str(icluster)

    print('plot histogram for ' + groupname)

    maps_res_3d = np.broadcast_to(maps_res == icluster, dataset_var.shape)
    tempdata = dataset_var[maps_res_3d]
    binmax = np.amax(tempdata[~np.isnan(tempdata)])*2./3.
    binarray = np.arange(0, binmax, binmax/20)
    binarrays.append(binarray)

    title = str(iniyear)+' to '+str(endyear)+' TRMM Total precip distribution in the group: '+str(icluster)
    fname = 'TRMM_prect_hist_group_'+str(icluster)+'.png'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx in range(len(cluster_labels)):
        if cluster_labels[idx] == icluster:
            lat_res = np.argmin(np.abs(reg_lats - temp_lat[idx]))
            lon_res = np.argmin(np.abs(reg_lons - temp_lon[idx]))
            tempdata = dataset_var[:, lat_res, lon_res]
            y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarray, density=True)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            plt.plot(bincenters, y, c='grey', linestyle='-', linewidth=1.5, alpha=0.4)

    tempdata = dataset_var[maps_res_3d]
    y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarray, density=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters, y, c=colors[icluster], linestyle='-', linewidth=1.5, label=groupname)

    plt.yscale('log')
    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir2+fname, bbox_inches='tight', dpi=1000)
    plt.close(fig)

# plot annual mean ts for all groups
print('plot histogram for all groups')
fig = plt.figure()
ax = fig.add_subplot(111)

title = str(iniyear)+' to '+str(endyear)+' TRMM Total precip distribution'
fname = 'TRMM_prect_hist_groupinone.png'

for idx in range(len(cluster_labels)):
    lat_res = np.argmin(np.abs(reg_lats - temp_lat[idx]))
    lon_res = np.argmin(np.abs(reg_lons - temp_lon[idx]))
    tempdata = dataset_var[:, lat_res, lon_res]
    y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarrays[cluster_labels[idx]], density=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters, y, c=colors[cluster_labels[idx]], linestyle='-', linewidth=1.5, alpha=0.4)

lines = []
legendnames = []
for icluster in range(ncluster):
    iname = 'group'+str(icluster)
    legendnames.append(iname)
    maps_res_3d = np.broadcast_to(maps_res == icluster, dataset_var.shape)
    tempdata = dataset_var[maps_res_3d]
    y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarrays[icluster], density=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    lines += plt.plot(bincenters, y, c=colors[icluster], linestyle='-', linewidth=1.5, label=iname)

plt.yscale('log')
plt.legend(lines, legendnames, handlelength=4, fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(xlabel, fontsize=8)
plt.ylabel(ylabel, fontsize=8)

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir2+fname, bbox_inches='tight', dpi=300)
plt.close(fig)

# ############################################################################
# # plot for long term mean climatology
# ############################################################################
xlabel = 'Month'
ylabel = 'Precip (mm/day)'

dataset_climmean = np.ma.zeros((12, lat_2-lat_1+1, lon_2-lon_1+1))
for imon in range(12):
    dataset_climmean[imon, :, :] = np.ma.mean(dataset_var[(time.month == (imon+1)), :, :], axis=0)

# print(dataset_monmean)

dataset_gpclimmean = np.ma.zeros((12, ncluster))
for idx in range(dataset_climmean.shape[0]):
    for icluster in range(ncluster):
        temp = dataset_climmean[idx, :, :]
        dataset_gpclimmean[idx, icluster] = np.ma.mean(temp[maps_res == icluster])

print(dataset_gpclimmean)
kmeans_res['dataset_gpclimmean'] = dataset_gpclimmean

# plot climatology for each group
for icluster in range(ncluster):
    groupname = 'group'+str(icluster)

    print('plot climatology for ' + groupname)

    title = str(iniyear)+' to '+str(endyear)+' TRMM Seasonal cycle of precip in the group: '+str(icluster)
    fname = 'TRMM_prect_clim_mean_line_group_'+str(icluster)+'.png'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx in range(len(cluster_labels)):
        if cluster_labels[idx] == icluster:
            lat_res = np.argmin(np.abs(reg_lats - temp_lat[idx]))
            lon_res = np.argmin(np.abs(reg_lons - temp_lon[idx]))
            plt.plot(months, dataset_climmean[:, lat_res, lon_res], color='grey', marker='o',
                     markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)

    plt.plot(months, dataset_gpclimmean[:, icluster], color=colors[icluster], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, label=groupname)

    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir2+fname, bbox_inches='tight', dpi=1000)
    plt.close(fig)

# plot annual mean ts for all groups
print('plot annual mean ts for all groups')
fig = plt.figure()
ax = fig.add_subplot(111)

title = str(iniyear)+' to '+str(endyear)+' TRMM Seasonal cycle of precip'
fname = 'TRMM_prect_clim_mean_line_groupinone.png'

for idx in range(len(cluster_labels)):
    lat_res = np.argmin(np.abs(reg_lats - temp_lat[idx]))
    lon_res = np.argmin(np.abs(reg_lons - temp_lon[idx]))
    plt.plot(months, dataset_climmean[:, lat_res, lon_res], color=colors[cluster_labels[idx]], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)
lines = []
legendnames = []
for icluster in range(ncluster):
    iname = 'group'+str(icluster)
    legendnames.append(iname)
    lines += plt.plot(months, dataset_gpclimmean[:, icluster], color=colors[icluster], marker='o',
                      markersize=1, linestyle='-', linewidth=1.5, label=iname)

plt.legend(lines, legendnames, handlelength=4, fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(xlabel, fontsize=8)
plt.ylabel(ylabel, fontsize=8)

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir2+fname, bbox_inches='tight', dpi=1000)
plt.close(fig)


# test pickle data
pickle.dump(kmeans_res, open(outdir+'TRMM_kmeans_result_overland_'+str(ncluster)+'cluster.p', "wb"))
res_load = pickle.load(open(outdir+'TRMM_kmeans_result_overland_'+str(ncluster)+'cluster.p', "rb"))
print(res_load['4 clusters'])
print(res_load['kmeans_res_lats'].shape)
print(res_load['kmeans_res_lats'])
