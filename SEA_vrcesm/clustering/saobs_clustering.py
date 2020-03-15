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
from modules.datareader.mod_dataread_obs_pre import read_SAOBS_pre
from modules.stats.mod_stats_clim import mon2clim
import pandas as pd
import pickle
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')


############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/obs/SA-OBS/clustering/'

############################################################################
# set parameters
############################################################################
# variable info
varname = 'Total Precip'
varstr = 'prect'
var_unit = 'mm/day'

# time bounds
iniyear = 1990
endyear = 2005
yearts = np.arange(iniyear, endyear+1)
# yearts    = np.delete(yearts,9,None)
print(yearts)

# define number of clusters
ncluster = 3

# define regions
latbounds = [-15, 25]
lonbounds = [90, 145]

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]

# set data frequency
frequency = 'day'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# color for clusters
colors = ['red', 'blue', 'green', 'magenta']

outdir = outdir+str(iniyear)+'-'+str(endyear)+'/'

############################################################################
# read data
############################################################################

print('Reading SA-OBS data...')

# read SA-OBS
version = 'countries'
countries = ['Thailand', 'Vietnam', 'Myanmar', 'Cambodia']
dataset, obs_var, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_SAOBS_pre(
    version, iniyear, endyear, countries, missing_ratio=10, ignore_years=[1999])

print(dataset)
print(dataset.resample('A').mean())

############################################################################
# plot stations on the map
############################################################################
'''
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)

map = Basemap(llcrnrlon=91.0, llcrnrlat=6, urcrnrlon=110.0, urcrnrlat=22,
              lat_0=14., lon_0=100.5, resolution='h', epsg=4326)
#map = Basemap(projection='', lat_0=17, lon_0=100.5,resolution = 'l', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

# map.drawcoastlines(linewidth=0.3)
# map.drawcountries()
# map.bluemarble()
map.arcgisimage(service='World_Physical_Map', xpixels=50000)

latsmap = stnlats[0:]
lonsmap = stnlons[0:]
x, y = map(lonsmap, latsmap)

#plt.plot(x, y, 'ok', markersize=5)
#plt.text(x, y, stnnames, fontsize=10);

cs = map.scatter(x, y, s=15, marker="o", c='black', alpha=0.7)
labels = stnnames[0:]
count = 0
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+0.1, ypt+0.1, str(count+1)+':'+label, fontsize=5)
    count += 1

#plt.text(x, y, stnnames, fontsize=10);
#fig.subplots_adjust(bottom=0.2,wspace = 0.2,hspace = 0.2)

title = 'SA-OBS stations'
fname = 'SAOBS_station_locations.pdf'
plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight', dpi=3000)
plt.close(fig)
'''

############################################################################
# clustering
############################################################################
dataset_var = dataset.values
print(dataset_var.shape)
fname = 'SAOBS_prect_kmeans_nclusters'

# res_spectral = spectral_cluster(dataset_var, ncluster, stnlats, stnlons, stnnames, outdir+fname)
res = kmeans_cluster(dataset_var, ncluster, stnlats, stnlons, stnnames, outdir+fname, map_plot=True)

res_save = {}

for idx in range(len(res)):
    temp = res[idx]
    print('number of clusterings: '+str(idx+2))
    print(temp.labels_)
    res_save[str(idx+2)] = temp.labels_
print(stnhgts)

res_save['stanames'] = stnnames
res_save = pd.DataFrame(res_save)
res_save = res_save.set_index('stanames')
print(res_save)

# save pickle data
pickle.dump(res_save, open(outdir+'SA-OBS_'+str(iniyear)+'to'+str(endyear)+'_kmeans_clustering_results.p', "wb"))
res_save.to_csv(outdir+'SA-OBS_'+str(iniyear)+'to'+str(endyear)+'_kmeans_clustering_results.csv', sep=',', index=True)

############################################################################
# plot for the different groups
############################################################################

index = np.arange(9)
bar_width = 0.8
opacity = 0.8
shape_type = ['', '', '', '', '..', '..', '..', '..', '//']

cluster_labels = res[ncluster-2].labels_
# cluster_labels = np.array([0,0,1,2,0,0,0,0,0,0,1,0,2,0,2,1,0,0,2,1,1,0])
print('Current number of clusterings is: '+str(ncluster))
outdir = outdir+str(iniyear)+'-'+str(endyear)+' kmeans '+str(ncluster)+' clusters/'

for idx in range(ncluster):
    tempnames = stnnames[cluster_labels == idx]
    dataset['group'+str(idx)] = dataset[tempnames].mean(axis=1)

dataset_monmean = dataset.resample('M').mean()
print(dataset_monmean)
dataset_annmean = dataset.resample('A').mean()

############################################################################
# plot for monthly mean TS
############################################################################

monthts = np.arange((endyear-iniyear+1)*12) + 1
xlabel = 'Month'
ylabel = 'Precip (mm/day)'
xticks = np.arange(6, (endyear-iniyear+1)*12, 12)
xticknames = [str(iyear) for iyear in yearts]

# plot monthly mean ts for each group
for idx in range(ncluster):
    groupname = 'group'+str(idx)
    tempnames = stnnames[cluster_labels == idx]

    tempnames = np.append(tempnames, groupname)

    # print(dataset_monmean[tempnames])

    print('plot mothly mean ts for ' + groupname)

    title = str(iniyear)+' to '+str(endyear)+' SA-OBS Monthly mean precip TS in the group: '+str(idx)
    fname = 'SAOBS_prect_monthly_mean_ts_group_'+str(idx)+'.pdf'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for iname in tempnames[:-1]:
        plt.plot(monthts, dataset_monmean[iname], color=colors[idx], marker='o',
                 markersize=1, linestyle='-', linewidth=1.5, label=iname, alpha=0.4)
    plt.plot(monthts, dataset_monmean[groupname], color=colors[idx], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, label=groupname)

    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(xticks, xticknames, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

# plot monthly mean ts for all groups
print('plot mothly mean ts for all groups')
fig = plt.figure()
ax = fig.add_subplot(111)

title = str(iniyear)+' to '+str(endyear)+' SA-OBS Monthly mean precip TS'
fname = 'SAOBS_prect_monthly_mean_ts_groupinone.pdf'

for idx in range(len(stnnames)):
    iname = stnnames[idx]
    plt.plot(monthts, dataset_monmean[iname], color=colors[cluster_labels[idx]], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)
lines = []
legendnames = []
for idx in range(ncluster):
    iname = 'group'+str(idx)
    legendnames.append(iname)
    lines += plt.plot(monthts, dataset_monmean[iname], color=colors[idx], marker='o',
                      markersize=1, linestyle='-', linewidth=1.5, label=iname)

plt.legend(lines, legendnames, handlelength=4, fontsize=5)
plt.xticks(xticks, xticknames, fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(xlabel, fontsize=8)
plt.ylabel(ylabel, fontsize=8)

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)


############################################################################
# plot for annual mean TS
############################################################################
xlabel = 'Year'
ylabel = 'Precip (mm/day)'

# plot annual mean ts for each group
for idx in range(ncluster):
    groupname = 'group'+str(idx)
    tempnames = stnnames[cluster_labels == idx]

    tempnames = np.append(tempnames, groupname)

    # print(dataset_monmean[tempnames])

    print('plot annual mean ts for ' + groupname)

    title = str(iniyear)+' to '+str(endyear)+' SA-OBS Annual mean precip TS in the group: '+str(idx)
    fname = 'SAOBS_prect_annual_mean_ts_group_'+str(idx)+'.pdf'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for iname in tempnames[:-1]:
        plt.plot(yearts, dataset_annmean[iname], color=colors[idx], marker='o',
                 markersize=1, linestyle='-', linewidth=1.5, label=iname, alpha=0.4)
    plt.plot(yearts, dataset_annmean[groupname], color=colors[idx], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, label=groupname)

    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

# plot annual mean ts for all groups
print('plot annual mean ts for all groups')
fig = plt.figure()
ax = fig.add_subplot(111)

title = str(iniyear)+' to '+str(endyear)+' SA-OBS Annual mean precip TS'
fname = 'SAOBS_prect_annual_mean_ts_groupinone.pdf'

for idx in range(len(stnnames)):
    iname = stnnames[idx]
    plt.plot(yearts, dataset_annmean[iname], color=colors[cluster_labels[idx]], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)
lines = []
legendnames = []
for idx in range(ncluster):
    iname = 'group'+str(idx)
    legendnames.append(iname)
    lines += plt.plot(yearts, dataset_annmean[iname], color=colors[idx], marker='o',
                      markersize=1, linestyle='-', linewidth=1.5, label=iname)

plt.legend(lines, legendnames, handlelength=4, fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(xlabel, fontsize=8)
plt.ylabel(ylabel, fontsize=8)

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)

############################################################################
# plot for histogram
############################################################################
xlabel = 'Precip (mm/day)'
ylabel = 'Frequency'

# plot annual mean ts for each group
binarrays = []
for idx in range(ncluster):
    groupname = 'group'+str(idx)
    tempnames = stnnames[cluster_labels == idx]

    # print(dataset_monmean[tempnames])

    print('plot histogram for ' + groupname)

    tempdata = dataset[tempnames].values
    binmax = np.amax(tempdata[~np.isnan(tempdata)])*2./3.
    binarray = np.arange(0, binmax, binmax/20)
    binarrays.append(binarray)

    title = str(iniyear)+' to '+str(endyear)+' SA-OBS Total precip distribution in the group: '+str(idx)
    fname = 'SAOBS_prect_hist_group_'+str(idx)+'.pdf'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for iname in tempnames:
        tempdata = dataset[iname]
        y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarray, density=True)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters, y, c=colors[idx], linestyle='-', linewidth=1.5, label=iname, alpha=0.4)

    tempdata = dataset[tempnames].values
    y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarray, density=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters, y, c=colors[idx], linestyle='-', linewidth=1.5, label=groupname)

    plt.yscale('log')
    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

# plot annual mean ts for all groups
print('plot histogram for all groups')
fig = plt.figure()
ax = fig.add_subplot(111)

title = str(iniyear)+' to '+str(endyear)+' SA-OBS Total precip distribution'
fname = 'SAOBS_prect_hist_groupinone.pdf'

for idx in range(len(stnnames)):
    iname = stnnames[idx]
    tempdata = dataset[iname]
    y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarrays[cluster_labels[idx]], density=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters, y, c=colors[cluster_labels[idx]], linestyle='-', linewidth=1.5, label=iname, alpha=0.4)
lines = []
legendnames = []
for idx in range(ncluster):
    iname = 'group'+str(idx)
    legendnames.append(iname)
    tempnames = stnnames[cluster_labels == idx]
    tempdata = dataset[tempnames]
    y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarrays[idx], density=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    lines += plt.plot(bincenters, y, c=colors[idx], linestyle='-', linewidth=1.5, label=iname)

plt.yscale('log')
plt.legend(lines, legendnames, handlelength=4, fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(xlabel, fontsize=8)
plt.ylabel(ylabel, fontsize=8)

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)

############################################################################
# plot for long term mean climatology
############################################################################
xlabel = 'Month'
ylabel = 'Precip (mm/day)'

# plot climatology for each group
for idx in range(ncluster):
    groupname = 'group'+str(idx)
    tempnames = stnnames[cluster_labels == idx]
    tempnames = np.append(tempnames, groupname)

    # print(dataset_monmean[tempnames])

    print('plot climatology for ' + groupname)

    title = str(iniyear)+' to '+str(endyear)+' SA-OBS Seasonal cycle of precip in the group: '+str(idx)
    fname = 'SAOBS_prect_clim_mean_line_group_'+str(idx)+'.pdf'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for iname in tempnames[:-1]:
        tempdata = dataset_monmean[iname]
        temp_mean, temp_std = mon2clim(tempdata)
        plt.plot(months, temp_mean, color=colors[idx], marker='o', markersize=1,
                 linestyle='-', linewidth=1.5, label=iname, alpha=0.4)
    tempdata = dataset_monmean[groupname]
    temp_mean, temp_std = mon2clim(tempdata)
    plt.plot(months, temp_mean, color=colors[idx], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, label=groupname)

    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

# plot annual mean ts for all groups
print('plot annual mean ts for all groups')
fig = plt.figure()
ax = fig.add_subplot(111)

title = str(iniyear)+' to '+str(endyear)+' SA-OBS Seasonal cycle of precip'
fname = 'SAOBS_prect_clim_mean_line_groupinone.pdf'

for idx in range(len(stnnames)):
    iname = stnnames[idx]
    tempdata = dataset_monmean[iname]
    temp_mean, temp_std = mon2clim(tempdata)
    plt.plot(months, temp_mean, color=colors[cluster_labels[idx]], marker='o',
             markersize=1, linestyle='-', linewidth=1.5, alpha=0.4)
lines = []
legendnames = []
for idx in range(ncluster):
    iname = 'group'+str(idx)
    legendnames.append(iname)
    tempdata = dataset_monmean[iname]
    temp_mean, temp_std = mon2clim(tempdata)
    lines += plt.plot(months, temp_mean, color=colors[idx], marker='o',
                      markersize=1, linestyle='-', linewidth=1.5, label=iname)

plt.legend(lines, legendnames, handlelength=4, fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(xlabel, fontsize=8)
plt.ylabel(ylabel, fontsize=8)

plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)
