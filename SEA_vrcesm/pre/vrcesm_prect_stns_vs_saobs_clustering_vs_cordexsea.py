# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-plot basic analysis
# S3-calculate and plot extreme
#
# Written by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_obs_pre import read_SAOBS_pre
from modules.stats.mod_stats_clim import mon2clim
from modules.plot.mod_plt_bars import plot_bars
from modules.plot.mod_plt_lines import plot_lines
from modules.plot.mod_plt_findstns import data_findstns
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle
plt.switch_backend('agg')


############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/SA-OBS/clustering/'
kmeans_resdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/obs/SA-OBS/clustering/'

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
# print(yearts)

# select number of clusters
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
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
modelname = 'IPSL-IPSL-CM5A-LR'
cordex_var2, cordex_time2, cordex_lats2, cordex_lons2 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
modelname = 'MPI-M-MPI-ESM-MR'
cordex_var3, cordex_time3, cordex_lats3, cordex_lons3 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
modelname = 'MOHC-HadGEM2-ES'
cordex_var4, cordex_time4, cordex_lats4, cordex_lons4 = readcordex(
    varname, iniyear, endyear, project, modelname, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

# convert from kg/(m^2*s) to mm/day
cordex_var1 = cordex_var1 * 86400 * 1000 / 997
cordex_var2 = cordex_var2 * 86400 * 1000 / 997
cordex_var3 = cordex_var3 * 86400 * 1000 / 997
cordex_var4 = cordex_var4 * 86400 * 1000 / 997

# print(cordex_var4.shape)
# print(cordex_time4)
# read vrcesm

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

model_var1 = 86400 * 1000 * model_var1
model_var2 = 86400 * 1000 * model_var2
model_var3 = 86400 * 1000 * model_var3
model_var4 = 86400 * 1000 * model_var4

print(model_var1.shape)


print('Reading SA-OBS data...')

# read SA-OBS
version = 'countries'
countries = ['Thailand', 'Vietnam', 'Myanmar', 'Cambodia']
dataset1, obs_var1, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_SAOBS_pre(
    version, iniyear, endyear, countries, missing_ratio=10, ignore_years=[1999])

# find stations in gridded data
cordex_var1 = data_findstns(cordex_var1, cordex_time1, cordex_lats1, cordex_lons1, obs_var1, stnlats, stnlons, stnnames)
cordex_var2 = data_findstns(cordex_var2, cordex_time2, cordex_lats2, cordex_lons2, obs_var1, stnlats, stnlons, stnnames)
cordex_var3 = data_findstns(cordex_var3, cordex_time3, cordex_lats3, cordex_lons3, obs_var1, stnlats, stnlons, stnnames)
cordex_var4 = data_findstns(cordex_var4, cordex_time4, cordex_lats4, cordex_lons4, obs_var1, stnlats, stnlons, stnnames)

model_var1 = data_findstns(model_var1, model_time1, model_lats1, model_lons1, obs_var1, stnlats, stnlons, stnnames)
model_var2 = data_findstns(model_var2, model_time2, model_lats2, model_lons2, obs_var1, stnlats, stnlons, stnnames)
model_var3 = data_findstns(model_var3, model_time3, model_lats3, model_lons3, obs_var1, stnlats, stnlons, stnnames)
model_var4 = data_findstns(model_var4, model_time4, model_lats4, model_lons4, obs_var1, stnlats, stnlons, stnnames)
# print(cordex_var1.shape)
# print(model_var4)

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH',
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES', 'SACA']

colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod', 'darkcyan', 'darkmagenta', 'black']
line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-.', '-.', '-.', '-.', '-']

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'SACA']
cesm_colors = ['red', 'yellow', 'green', 'blue', 'black']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-']

############################################################################
# plot stations on the map
############################################################################

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)

map = Basemap(llcrnrlon=91.0, llcrnrlat=6, urcrnrlon=110.0, urcrnrlat=22,
              lat_0=14., lon_0=100.5, resolution='h', epsg=4326)
# map = Basemap(projection='', lat_0=17, lon_0=100.5,resolution = 'l', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

# map.drawcoastlines(linewidth=0.3)
# map.drawcountries()
# map.bluemarble()
map.arcgisimage(service='World_Physical_Map', xpixels=50000)

latsmap = stnlats[0:]
lonsmap = stnlons[0:]
x, y = map(lonsmap, latsmap)

# plt.plot(x, y, 'ok', markersize=5)
# plt.text(x, y, stnnames, fontsize=10);

cs = map.scatter(x, y, s=15, marker="o", c='black', alpha=0.7)
labels = stnnames[0:]
count = 0
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt+0.1, ypt+0.1, str(count+1)+':'+label, fontsize=5)
    count += 1

# plt.text(x, y, stnnames, fontsize=10);
# fig.subplots_adjust(bottom=0.2,wspace = 0.2,hspace = 0.2)

title = 'SACA stations'
fname = 'SAOBS_station_locations.pdf'
plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(outdir+fname, bbox_inches='tight', dpi=3000)
plt.close(fig)

outdir = outdir+str(iniyear)+'-'+str(endyear)+'/'+str(iniyear)+'-'+str(endyear)+' kmeans '+str(ncluster)+' clusters/'

############################################################################
# read kmean clustering result
############################################################################
kmeans_resdir = kmeans_resdir+str(iniyear)+'-'+str(endyear)+'/'

res_labels = pickle.load(open(kmeans_resdir+'SA-OBS_'+str(iniyear)+'to' +
                              str(endyear)+'_kmeans_clustering_results.p', "rb"))
cluster_labels = res_labels[str(ncluster)].values

############################################################################
# plot for the different groups
############################################################################

index = np.arange(9)
bar_width = 0.8
opacity = 0.8
shape_type = ['', '', '', '', '..', '..', '..', '..', '//']

# ncluster = 4
# cluster_labels = np.array([3, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 3, 0, 0, 2, 2, 0])
# ncluster = 3
# cluster_labels = np.array([0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 0, 0, 0, 2, 2, 0])
# cluster_labels = np.array([0,0,1,2,0,0,0,0,0,0,1,0,2,0,2,1,0,0,2,1,1,0])

'''
for idx in range(ncluster):
    tempnames = stnnames[cluster_labels==idx]
    print(tempnames)
    cordex_var1['group'+str(idx)] = cordex_var1[tempnames].mean(axis=1)
    cordex_var2['group'+str(idx)] = cordex_var2[tempnames].mean(axis=1)
    cordex_var3['group'+str(idx)] = cordex_var3[tempnames].mean(axis=1)
    cordex_var4['group'+str(idx)] = cordex_var4[tempnames].mean(axis=1)

    model_var1['group'+str(idx)] = model_var1[tempnames].mean(axis=1)
    model_var2['group'+str(idx)] = model_var2[tempnames].mean(axis=1)
    model_var3['group'+str(idx)] = model_var3[tempnames].mean(axis=1)
    model_var4['group'+str(idx)] = model_var4[tempnames].mean(axis=1)

    dataset1['group'+str(idx)] = dataset1[tempnames].mean(axis=1)


print(cordex_var1)
'''

# calculate annual maximum precip for each station
cordex_annmax1 = cordex_var1.resample('A').max()
cordex_annmax2 = cordex_var2.resample('A').max()
cordex_annmax3 = cordex_var3.resample('A').max()
cordex_annmax4 = cordex_var4.resample('A').max()

model_annmax1 = model_var1.resample('A').max()
model_annmax2 = model_var2.resample('A').max()
model_annmax3 = model_var3.resample('A').max()
model_annmax4 = model_var4.resample('A').max()

dataset_annmax1 = dataset1.resample('A').max()


# calculate annual mean precip for each station
cordex_annmean1 = cordex_var1.resample('A').mean()
cordex_annmean2 = cordex_var2.resample('A').mean()
cordex_annmean3 = cordex_var3.resample('A').mean()
cordex_annmean4 = cordex_var4.resample('A').mean()

model_annmean1 = model_var1.resample('A').mean()
model_annmean2 = model_var2.resample('A').mean()
model_annmean3 = model_var3.resample('A').mean()
model_annmean4 = model_var4.resample('A').mean()

dataset_annmean1 = dataset1.resample('A').mean()

# plot annual mean precip for each station
cordex_monmean1 = cordex_var1.resample('M').mean()
cordex_monmean2 = cordex_var2.resample('M').mean()
cordex_monmean3 = cordex_var3.resample('M').mean()
cordex_monmean4 = cordex_var4.resample('M').mean()

model_monmean1 = model_var1.resample('M').mean()
model_monmean2 = model_var2.resample('M').mean()
model_monmean3 = model_var3.resample('M').mean()
model_monmean4 = model_var4.resample('M').mean()

dataset_monmean1 = dataset1.resample('M').mean()

for idx in range(ncluster):
    tempnames = stnnames[cluster_labels == idx]
    print(tempnames)
    cordex_annmax1['group'+str(idx)] = cordex_annmax1[tempnames].mean(axis=1)
    cordex_annmax2['group'+str(idx)] = cordex_annmax2[tempnames].mean(axis=1)
    cordex_annmax3['group'+str(idx)] = cordex_annmax3[tempnames].mean(axis=1)
    cordex_annmax4['group'+str(idx)] = cordex_annmax4[tempnames].mean(axis=1)

    model_annmax1['group'+str(idx)] = model_annmax1[tempnames].mean(axis=1)
    model_annmax2['group'+str(idx)] = model_annmax2[tempnames].mean(axis=1)
    model_annmax3['group'+str(idx)] = model_annmax3[tempnames].mean(axis=1)
    model_annmax4['group'+str(idx)] = model_annmax4[tempnames].mean(axis=1)

    dataset_annmax1['group'+str(idx)] = dataset_annmax1[tempnames].mean(axis=1)

    cordex_annmean1['group'+str(idx)] = cordex_annmean1[tempnames].mean(axis=1)
    cordex_annmean2['group'+str(idx)] = cordex_annmean2[tempnames].mean(axis=1)
    cordex_annmean3['group'+str(idx)] = cordex_annmean3[tempnames].mean(axis=1)
    cordex_annmean4['group'+str(idx)] = cordex_annmean4[tempnames].mean(axis=1)

    model_annmean1['group'+str(idx)] = model_annmean1[tempnames].mean(axis=1)
    model_annmean2['group'+str(idx)] = model_annmean2[tempnames].mean(axis=1)
    model_annmean3['group'+str(idx)] = model_annmean3[tempnames].mean(axis=1)
    model_annmean4['group'+str(idx)] = model_annmean4[tempnames].mean(axis=1)

    dataset_annmean1['group'+str(idx)] = dataset_annmean1[tempnames].mean(axis=1)

    cordex_monmean1['group'+str(idx)] = cordex_monmean1[tempnames].mean(axis=1)
    cordex_monmean2['group'+str(idx)] = cordex_monmean2[tempnames].mean(axis=1)
    cordex_monmean3['group'+str(idx)] = cordex_monmean3[tempnames].mean(axis=1)
    cordex_monmean4['group'+str(idx)] = cordex_monmean4[tempnames].mean(axis=1)

    model_monmean1['group'+str(idx)] = model_monmean1[tempnames].mean(axis=1)
    model_monmean2['group'+str(idx)] = model_monmean2[tempnames].mean(axis=1)
    model_monmean3['group'+str(idx)] = model_monmean3[tempnames].mean(axis=1)
    model_monmean4['group'+str(idx)] = model_monmean4[tempnames].mean(axis=1)

    dataset_monmean1['group'+str(idx)] = dataset_monmean1[tempnames].mean(axis=1)

print(cordex_monmean1)

for idx in range(ncluster):
    groupname = 'group'+str(idx)
    tempnames = stnnames[cluster_labels == idx]

    print('Plotting the annaul max time series for '+groupname)
    # plot for annual max ts for each group
    # plot for both cesm and cordex
    plot_data = [model_annmax1[groupname].values, model_annmax2[groupname].values, model_annmax3[groupname].values, model_annmax4[groupname].values,
                 cordex_annmax1[groupname].values, cordex_annmax2[groupname].values, cordex_annmax3[groupname].values, cordex_annmax4[groupname].values, dataset_annmax1[groupname].values]

    xlabel = 'Models and SACA'
    ylabel = 'Precip (mm/day)'

    title = str(iniyear)+' to '+str(endyear)+'Annual Maximum precip in the group: '+str(idx)
    fname = 'vrcesm_prect_annual_max_line_vs_cordex_refSAOBS_station_group_'+str(idx)+'.pdf'
    plot_lines(yearts, plot_data, colors, line_types, legends, xlabel, ylabel, title, outdir+fname)

    # plot for cesm only
    plot_data = [model_annmax1[groupname].values, model_annmax2[groupname].values,
                 model_annmax3[groupname].values, model_annmax4[groupname].values, dataset_annmax1[groupname].values]

    title = str(iniyear)+' to '+str(endyear)+'Annual Maximum precip in the group: '+str(idx)
    fname = 'vrcesm_prect_annual_max_line_refSAOBS_group_'+str(idx)+'.pdf'
    plot_lines(yearts, plot_data, cesm_colors, cesm_line_types, cesm_legends, xlabel, ylabel, title, outdir+fname)

    # plot for mean annual max bar for each group
    cordex_annmax_mean1 = cordex_annmax1[groupname].mean(axis=0)
    cordex_annmax_mean2 = cordex_annmax2[groupname].mean(axis=0)
    cordex_annmax_mean3 = cordex_annmax3[groupname].mean(axis=0)
    cordex_annmax_mean4 = cordex_annmax4[groupname].mean(axis=0)

    model_annmax_mean1 = model_annmax1[groupname].mean(axis=0)
    model_annmax_mean2 = model_annmax2[groupname].mean(axis=0)
    model_annmax_mean3 = model_annmax3[groupname].mean(axis=0)
    model_annmax_mean4 = model_annmax4[groupname].mean(axis=0)

    dataset_annmax_mean1 = dataset_annmax1[groupname].mean(axis=0)

    cordex_annmax_std1 = cordex_annmax1[groupname].std(axis=0)
    cordex_annmax_std2 = cordex_annmax2[groupname].std(axis=0)
    cordex_annmax_std3 = cordex_annmax3[groupname].std(axis=0)
    cordex_annmax_std4 = cordex_annmax4[groupname].std(axis=0)

    model_annmax_std1 = model_annmax1[groupname].std(axis=0)
    model_annmax_std2 = model_annmax2[groupname].std(axis=0)
    model_annmax_std3 = model_annmax3[groupname].std(axis=0)
    model_annmax_std4 = model_annmax4[groupname].std(axis=0)

    dataset_annmax_std1 = dataset_annmax1[groupname].std(axis=0)

    print('Plotting the annaul max bar chart for '+groupname)
    # plot for both cesm and cordex
    plot_data = [model_annmax_mean1, model_annmax_mean2, model_annmax_mean3, model_annmax_mean4,
                 cordex_annmax_mean1, cordex_annmax_mean2, cordex_annmax_mean3, cordex_annmax_mean4, dataset_annmax_mean1]
    plot_err = [model_annmax_std1, model_annmax_std2, model_annmax_std3, model_annmax_std4,
                cordex_annmax_std1, cordex_annmax_std2, cordex_annmax_std3, cordex_annmax_std4, dataset_annmax_std1]

    xlabel = 'Models and SACA'
    ylabel = 'Precip (mm/day)'

    title = str(iniyear)+' to '+str(endyear)+' mean annual maximum precip in the group: '+str(idx)
    fname = 'vrcesm_prect_annual_max_bar_vs_cordex_refSAOBS_group_'+str(idx)+'.pdf'
    plot_bars(plot_data, plot_err, colors, legends, xlabel, ylabel, title, outdir+fname)

    # plot for cesm only
    plot_data = [model_annmax_mean1, model_annmax_mean2, model_annmax_mean3, model_annmax_mean4, dataset_annmax_mean1]
    plot_err = [model_annmax_std1, model_annmax_std2, model_annmax_std3, model_annmax_std4, dataset_annmax_std1]

    xlabel = 'Models and SACA'
    ylabel = 'Precip (mm/day)'

    title = str(iniyear)+' to '+str(endyear)+' mean annual maximum precip in the group: '+str(idx)
    fname = 'vrcesm_prect_annual_max_bar_vs_refSAOBS_group_'+str(idx)+'.pdf'
    plot_bars(plot_data, plot_err, cesm_colors, cesm_legends, xlabel, ylabel, title, outdir+fname)

    # plot annual mean ts for each station
    print('Plot the annaul mean time series for '+groupname)
    xlabel = 'Year'
    ylabel = 'Precip (mm/day)'

    # plot for both cesm and cordex
    plot_data = [model_annmean1[groupname].values, model_annmean2[groupname].values, model_annmean3[groupname].values, model_annmean4[groupname].values,
                 cordex_annmean1[groupname].values, cordex_annmean2[groupname].values, cordex_annmean3[groupname].values, cordex_annmean4[groupname].values, dataset_annmean1[groupname].values]

    title = str(iniyear)+' to '+str(endyear)+'Annual mean precip in the group: '+str(idx)
    fname = 'vrcesm_prect_annual_mean_line_vs_cordex_refSAOBS_group_'+str(idx)+'.pdf'
    plot_lines(yearts, plot_data, colors, line_types, legends, xlabel, ylabel, title, outdir+fname)

    # plot for cesm only
    plot_data = [model_annmean1[groupname].values, model_annmean2[groupname].values,
                 model_annmean3[groupname].values, model_annmean4[groupname].values, dataset_annmean1[groupname].values]

    title = str(iniyear)+' to '+str(endyear)+'Annual mean precip in the group: '+str(idx)
    fname = 'vrcesm_prect_annual_mean_line_refSAOBS_group_'+str(idx)+'.pdf'
    plot_lines(yearts, plot_data, cesm_colors, cesm_line_types, cesm_legends, xlabel, ylabel, title, outdir+fname)

    # plot for mean annual mean bar for each group
    cordex_annmean_mean1 = cordex_annmean1[groupname].mean(axis=0)
    cordex_annmean_mean2 = cordex_annmean2[groupname].mean(axis=0)
    cordex_annmean_mean3 = cordex_annmean3[groupname].mean(axis=0)
    cordex_annmean_mean4 = cordex_annmean4[groupname].mean(axis=0)

    model_annmean_mean1 = model_annmean1[groupname].mean(axis=0)
    model_annmean_mean2 = model_annmean2[groupname].mean(axis=0)
    model_annmean_mean3 = model_annmean3[groupname].mean(axis=0)
    model_annmean_mean4 = model_annmean4[groupname].mean(axis=0)

    dataset_annmean_mean1 = dataset_annmean1[groupname].mean(axis=0)

    cordex_annmean_std1 = cordex_annmean1[groupname].std(axis=0)
    cordex_annmean_std2 = cordex_annmean2[groupname].std(axis=0)
    cordex_annmean_std3 = cordex_annmean3[groupname].std(axis=0)
    cordex_annmean_std4 = cordex_annmean4[groupname].std(axis=0)

    model_annmean_std1 = model_annmean1[groupname].std(axis=0)
    model_annmean_std2 = model_annmean2[groupname].std(axis=0)
    model_annmean_std3 = model_annmean3[groupname].std(axis=0)
    model_annmean_std4 = model_annmean4[groupname].std(axis=0)

    dataset_annmean_std1 = dataset_annmean1[groupname].std(axis=0)

    print('Plot the mean annaul mean for '+groupname)
    # plot for both cesm and cordex
    plot_data = [model_annmean_mean1, model_annmean_mean2, model_annmean_mean3, model_annmean_mean4,
                 cordex_annmean_mean1, cordex_annmean_mean2, cordex_annmean_mean3, cordex_annmean_mean4, dataset_annmean_mean1]
    plot_err = [model_annmean_std1, model_annmean_std2, model_annmean_std3, model_annmean_std4,
                cordex_annmean_std1, cordex_annmean_std2, cordex_annmean_std3, cordex_annmean_std4, dataset_annmean_std1]

    xlabel = 'Models and SACA'
    ylabel = 'Precip (mm/day)'

    title = str(iniyear)+' to '+str(endyear)+' mean annual mean precip in the group: '+str(idx)
    fname = 'vrcesm_prect_annual_mean_bar_vs_cordex_refSAOBS_group_'+str(idx)+'.pdf'
    plot_bars(plot_data, plot_err, colors, legends, xlabel, ylabel, title, outdir+fname)

    # plot for cesm only
    plot_data = [model_annmean_mean1, model_annmean_mean2,
                 model_annmean_mean3, model_annmean_mean4, dataset_annmean_mean1]
    plot_err = [model_annmean_std1, model_annmean_std2, model_annmean_std3, model_annmean_std4, dataset_annmean_std1]

    xlabel = 'Models and SACA'
    ylabel = 'Precip (mm/day)'

    title = str(iniyear)+' to '+str(endyear)+' mean annual mean precip in the group: '+str(idx)
    fname = 'vrcesm_prect_annual_mean_bar_vs_refSAOBS_group_'+str(idx)+'.pdf'
    plot_bars(plot_data, plot_err, cesm_colors, cesm_legends, xlabel, ylabel, title, outdir+fname)

    # plot monthly mean ts for each group
    monthts = np.arange((endyear-iniyear+1)*12) + 1
    xlabel = 'Month'
    ylabel = 'Precip (mm/day)'
    xticks = np.arange(6, (endyear-iniyear+1)*12, 12)
    xticknames = [str(iyear) for iyear in yearts]

    print('Plot the monthly mean time series for '+groupname)
    # plot for both cesm and cordex
    plot_data = [model_monmean1[groupname].values, model_monmean2[groupname].values, model_monmean3[groupname].values, model_monmean4[groupname].values,
                 cordex_monmean1[groupname].values, cordex_monmean2[groupname].values, cordex_monmean3[groupname].values, cordex_monmean4[groupname].values, dataset_monmean1[groupname].values]

    title = str(iniyear)+' to '+str(endyear)+'Monthly mean precip in the group: '+str(idx)
    fname = 'vrcesm_prect_monthly_mean_line_vs_cordex_refSAOBS_group_'+str(idx)+'.pdf'
    plot_lines(monthts, plot_data, colors, line_types, legends, xlabel, ylabel,
               title, outdir+fname, xticks=xticks, xticknames=xticknames)

    # plot for cesm only
    plot_data = [model_monmean1[groupname].values, model_monmean2[groupname].values,
                 model_monmean3[groupname].values, model_monmean4[groupname].values, dataset_monmean1[groupname].values]

    title = str(iniyear)+' to '+str(endyear)+'Monthly mean precip in the group: '+str(idx)
    fname = 'vrcesm_prect_monthly_mean_line_refSAOBS_group_'+str(idx)+'.pdf'
    plot_lines(monthts, plot_data, cesm_colors, cesm_line_types, cesm_legends, xlabel,
               ylabel, title, outdir+fname, xticks=xticks, xticknames=xticknames)

    # plot climatological mean for each group
    xlabel = 'Month'
    ylabel = 'Precip (mm/day)'

    print('Plot the seasonalities for '+groupname)
    cordex_mean1, codex_std1 = mon2clim(cordex_monmean1[groupname].values)
    cordex_mean2, codex_std2 = mon2clim(cordex_monmean2[groupname].values)
    cordex_mean3, codex_std3 = mon2clim(cordex_monmean3[groupname].values)
    cordex_mean4, codex_std4 = mon2clim(cordex_monmean4[groupname].values)

    model_mean1, model_std1 = mon2clim(model_monmean1[groupname].values)
    model_mean2, model_std2 = mon2clim(model_monmean2[groupname].values)
    model_mean3, model_std3 = mon2clim(model_monmean3[groupname].values)
    model_mean4, model_std4 = mon2clim(model_monmean4[groupname].values)

    dataset_mean1, dataset_std1 = mon2clim(dataset_monmean1[groupname].values)

    # plot for both cesm and cordex
    plot_data = [model_mean1, model_mean2, model_mean3, model_mean4,
                 cordex_mean1, cordex_mean2, cordex_mean3, cordex_mean4, dataset_mean1]
    plot_err = [model_std1, model_std2, model_std3, model_std4,
                codex_std1, codex_std2, codex_std3, codex_std4, dataset_std1]

    title = str(iniyear)+' to '+str(endyear)+' Seasonal cycle of precip in the group: '+str(idx)
    fname = 'vrcesm_prect_clim_mean_line_vs_cordex_refSAOBS_group_'+str(idx)+'.pdf'
    plot_lines(months, plot_data, colors, line_types, legends, xlabel, ylabel, title, outdir+fname, yerr=plot_err)

    # plot for cesm only
    plot_data = [model_mean1, model_mean2, model_mean3, model_mean4, dataset_mean1]
    plot_err = [model_std1, model_std2, model_std3, model_std4, dataset_std1]

    title = str(iniyear)+' to '+str(endyear)+'Seasonal cycle of precip in the group: '+str(idx)
    fname = 'vrcesm_prect_clim_mean_line_refSAOBS_group_'+str(idx)+'.pdf'
    plot_lines(months, plot_data, cesm_colors, cesm_line_types, cesm_legends,
               xlabel, ylabel, title, outdir+fname, yerr=plot_err)

    # plot histogram for each group
    print('Plot the precip histogram for '+groupname)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    tempdata = dataset1[tempnames].values
    # print(tempdata)
    # print(len(tempdata))
    binmax = np.amax(tempdata[~np.isnan(tempdata)])*1./2.
    # print(binmax)
    binarray = np.arange(0, binmax, binmax/30)

    plot_data = [model_var1[tempnames].values, model_var2[tempnames].values, model_var3[tempnames].values, model_var4[tempnames].values,
                 cordex_var1[tempnames].values, cordex_var2[tempnames].values, cordex_var3[tempnames].values, cordex_var4[tempnames].values, dataset1[tempnames].values]

    for ii in range(9):
        tempdata = plot_data[ii]
        y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarray)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        # print(bincenters)
        plt.plot(bincenters, y, c=colors[ii], linestyle=line_types[ii], linewidth=1.5, label=legends[ii])

    plt.yscale('log')
    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("Days")
    plt.xlabel("Precip(mm/day)")

    title = str(iniyear)+' to '+str(endyear)+' Total precip distribution in the group: '+str(idx)
    fname = 'vrcesm_prect_hist_vs_cordex_refSAOBS_group_'+str(idx)+'.pdf'

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

    # plot for only cesm
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_data = [model_var1[tempnames].values, model_var2[tempnames].values,
                 model_var3[tempnames].values, model_var4[tempnames].values, dataset1[tempnames].values]

    for ii in range(5):
        tempdata = plot_data[ii]
        y, binEdges = np.histogram(tempdata[~np.isnan(tempdata)], bins=binarray)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        # print(bincenters)
        plt.plot(bincenters, y, c=cesm_colors[ii], linestyle=cesm_line_types[ii], linewidth=1.5, label=cesm_legends[ii])

    plt.yscale('log')
    plt.legend(handlelength=4, fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("Days")
    plt.xlabel("Precip(mm/day)")

    title = str(iniyear)+' to '+str(endyear)+' Total precip distribution in the group: '+str(idx)
    fname = 'vrcesm_prect_hist_refSAOBS_group_'+str(idx)+'.pdf'

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)
