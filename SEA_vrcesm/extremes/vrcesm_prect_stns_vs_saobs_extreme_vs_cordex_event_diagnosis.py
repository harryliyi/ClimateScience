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
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/SA-OBS/extremes/'
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

# select number of clusters
ncluster = 3

# define regions
latbounds = [-15, 25]
lonbounds = [90, 145]

# mainland Southeast Asia
reg_lats = [10, 25]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'day'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# percentile
percentile = 97

# set up nbins
nbins = 30

# check if map need to be plot
map_plot = False

outdir = outdir+str(iniyear)+'-'+str(endyear)+'/'+str(iniyear)+'-'+str(endyear)+' kmeans '+str(ncluster)+' clusters/'
kmeans_resdir = kmeans_resdir+str(iniyear)+'-'+str(endyear)+'/'

############################################################################
# functions
############################################################################


def plot_hist(plot_data, ref_data, percentile, colors, line_types, legends, varname, var_unit, title, fname, **kwargs):

    arranges = {1: [1, 1], 2: [2, 1], 3: [3, 1], 4: [2, 2], 5: [3, 2],
                6: [3, 2], 8: [2, 4], 9: [3, 3], 10: [3, 4], 12: [3, 4]}
    nfigs = len(plot_data)

    if 'density' in kwargs:
        density = kwargs['density']
    else:
        density = True

    if 'extremes' in kwargs:
        extremes = kwargs['extremes']
    else:
        extremes = False

    if 'extreme_fractions' in kwargs:
        extreme_fractions = kwargs['extreme_fractions']
    else:
        extreme_fractions = []

    if nfigs not in arranges:
        print('plot_2Dcontour: Error! Too many Sub-figures, the maximum number is 9!')
        return -1

    plt.clf()
    fig = plt.figure()

    ref_hist, ref_edges = np.histogram(ref_data, bins=30, density=density)
    ref_thres = np.percentile(ref_data, percentile)
    ref_bins = ref_edges[:-1]+(ref_edges[1]-ref_edges[0])/2

    for idx in range(len(plot_data)):
        # calculate the hist and threshold
        model_hist, model_edges = np.histogram(plot_data[idx], bins=ref_edges, density=density)

        if not extremes:
            model_thres = np.percentile(plot_data[idx], percentile)
            model_thres = round(model_thres, 2)

        ax = fig.add_subplot(arranges[nfigs][0], arranges[nfigs][1], idx+1)

        if extremes:
            if len(extreme_fractions) != 0:
                ax.set_title(legends[idx]+' \n events percent: '+str(round(extreme_fractions[idx], 2)), fontsize=5, pad=2)
            else:
                ax.set_title(legends[idx], fontsize=5, pad=2)
        else:
            ax.set_title(legends[idx]+' \n '+str(percentile)+'th percentile: '+str(model_thres), fontsize=5, pad=2)

        ax.plot(ref_bins, model_hist, color=colors[idx], linestyle=line_types[idx], linewidth=1.5, label=legends[idx])
        ax.plot(ref_bins, ref_hist, color='black', linestyle='-', linewidth=1., label=legends[-1])

        if not extremes:
            ax.axvline(x=model_thres, color=colors[idx], linestyle=line_types[idx], linewidth=1.)
            ax.axvline(x=ref_thres, color='black', linestyle='-', linewidth=1.)
            ax.set_yscale('log')

        if (len(plot_data)-idx) <= arranges[nfigs][1]:
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel(varname+' ['+var_unit+']', fontsize=5)
        else:
            ax.set_xticklabels([])

        ax.minorticks_off()
        if (idx % (arranges[nfigs][1]) == 0):
            ax.yaxis.set_tick_params(labelsize=5)
            if extremes:
                ax.set_ylabel('Counts', fontsize=5)
            else:
                ax.set_ylabel('Frequency', fontsize=5)
        else:
            ax.set_yticklabels([])

    fig.subplots_adjust(hspace=0.15)

    # add title
    plt.suptitle(title, fontsize=7, y=0.95)

    # save figure
    plt.savefig(fname, bbox_inches='tight')
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

# print(cordex_var1.shape)
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

# read Observations

print('Reading SA-OBS data...')
ref_name = 'SA-OBS'

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
           'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES', 'SA-OBS']

colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod', 'darkcyan', 'darkmagenta', 'black']
line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-.', '-.', '-.', '-.', '-']

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'SA-OBS']

cesm_colors = ['red', 'yellow', 'green', 'blue', 'black']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed',  '-']

############################################################################
# fit the extremes
############################################################################

res_labels = pickle.load(open(kmeans_resdir+'SA-OBS_'+str(iniyear)+'to' +
                              str(endyear)+'_kmeans_clustering_results.p', "rb"))
kmeans_labels = res_labels[str(ncluster)].values

# plot clustering result on map
if map_plot:
    print('Plot clustering result on map...')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    map = Basemap(llcrnrlon=91.0, llcrnrlat=6, urcrnrlon=110.0, urcrnrlat=22,
                  lat_0=14., lon_0=100.5, resolution='h', epsg=4326)
    # map = Basemap(projection='', lat_0=17, lon_0=100.5,resolution = 'l', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

    # map.drawcoastlines(linewidth=0.3)
    # map.drawcountries()
    map.arcgisimage(service='World_Physical_Map', xpixels=50000)

    latsmap = stnlats[0:]
    lonsmap = stnlons[0:]
    x, y = map(lonsmap, latsmap)

    cs = map.scatter(x, y, s=20, marker="o", c=kmeans_labels,
                     cmap=plt.cm.get_cmap('viridis', ncluster), alpha=0.7)

    labels = stnnames[0:]
    count = 0
    for label, xpt, ypt in zip(labels, x, y):
        plt.text(xpt+0.1, ypt+0.1, str(count+1)+':'+label, fontsize=5)
        count += 1

    fig.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    cbar_ax = fig.add_axes([0.2, 0.17, 0.6, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal', ticks=range(ncluster))
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('Cluster labels')
    plt.clim(-0.5, ncluster-0.5)

    title = 'K-means clustering: '+str(ncluster)+' groups'
    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+'SAOBS_prect_kmeans_nclusters_'+str(ncluster)+'.pdf', bbox_inches='tight', dpi=3000)
    plt.close(fig)


############################################################################
# plot for each group
############################################################################
datasets = [model_var1, model_var2, model_var3, model_var4,
            cordex_var1, cordex_var2, cordex_var3, cordex_var4,
            dataset1]

for idx in range(ncluster):
    print('Plotting for group'+str(idx)+'...')

    ############################################################################
    # plot histogram
    print('Plotting histogram for each group...')
    print('Total precipitation histogram...')

    select_group = (kmeans_labels == idx)
    # print(select_group)

    # pull data out of dataframe
    dataset_ts1 = np.array(dataset1.values[:, select_group])
    # print(dataset_ts1.shape)
    dataset_ts1 = dataset_ts1.flatten()
    # print(dataset_ts1)

    model_ts1 = model_var1.values[:, select_group].flatten()
    model_ts2 = model_var2.values[:, select_group].flatten()
    model_ts3 = model_var3.values[:, select_group].flatten()
    model_ts4 = model_var4.values[:, select_group].flatten()

    cordex_ts1 = cordex_var1.values[:, select_group].flatten()
    cordex_ts2 = cordex_var2.values[:, select_group].flatten()
    cordex_ts3 = cordex_var3.values[:, select_group].flatten()
    cordex_ts4 = cordex_var4.values[:, select_group].flatten()

    # remove NaN values
    # print(len(dataset_ts1))
    dataset_ts1 = dataset_ts1[~np.isnan(dataset_ts1)]
    # print(len(dataset_ts1))

    model_ts1 = model_ts1[~np.isnan(model_ts1)]
    model_ts2 = model_ts2[~np.isnan(model_ts2)]
    model_ts3 = model_ts3[~np.isnan(model_ts3)]
    model_ts4 = model_ts4[~np.isnan(model_ts4)]

    cordex_ts1 = cordex_ts1[~np.isnan(cordex_ts1)]
    cordex_ts2 = cordex_ts2[~np.isnan(cordex_ts2)]
    cordex_ts3 = cordex_ts3[~np.isnan(cordex_ts3)]
    cordex_ts4 = cordex_ts4[~np.isnan(cordex_ts4)]

    # calculate the percentile
    model_percent1 = np.percentile(model_ts1, percentile)
    model_percent2 = np.percentile(model_ts2, percentile)
    model_percent3 = np.percentile(model_ts3, percentile)
    model_percent4 = np.percentile(model_ts4, percentile)

    cordex_percent1 = np.percentile(cordex_ts1, percentile)
    cordex_percent2 = np.percentile(cordex_ts2, percentile)
    cordex_percent3 = np.percentile(cordex_ts3, percentile)
    cordex_percent4 = np.percentile(cordex_ts4, percentile)

    dataset_percent1 = np.percentile(dataset_ts1, percentile)

    extreme_thres = [model_percent1, model_percent2, model_percent3, model_percent4, cordex_percent1,
                     cordex_percent2, cordex_percent3, cordex_percent4, dataset_percent1]

    print(extreme_thres)

    # cesm and cordex
    plot_data = [model_ts1, model_ts2, model_ts3, model_ts4,
                 cordex_ts1, cordex_ts2, cordex_ts3, cordex_ts4]

    ref_data = dataset_ts1

    # frenquency distribution
    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' Precipitation frequency distribution vs '+ref_name+' over '+reg_name
    fname = 'vrcesm_prect_'+str(percentile)+'th_hist_vs_cordex_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, ref_data, percentile, colors, line_types, legends, varname, var_unit, title, outdir+fname)

    # cesm only
    plot_data = [model_ts1, model_ts2, model_ts3, model_ts4]

    ref_data = dataset_ts1

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' Precipitation frequency distribution vs '+ref_name+' over '+reg_name
    fname = 'vrcesm_prect_'+str(percentile)+'th_hist_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, ref_data, percentile, cesm_colors, cesm_line_types,
              cesm_legends, varname, var_unit, title, outdir+fname)

    print('Extreme precipitation histogram...')

    model_ts_sub1 = model_ts1[model_ts1 > dataset_percent1]
    model_ts_sub2 = model_ts2[model_ts2 > dataset_percent1]
    model_ts_sub3 = model_ts3[model_ts3 > dataset_percent1]
    model_ts_sub4 = model_ts4[model_ts4 > dataset_percent1]

    cordex_ts_sub1 = cordex_ts1[cordex_ts1 > dataset_percent1]
    cordex_ts_sub2 = cordex_ts2[cordex_ts2 > dataset_percent1]
    cordex_ts_sub3 = cordex_ts3[cordex_ts3 > dataset_percent1]
    cordex_ts_sub4 = cordex_ts4[cordex_ts4 > dataset_percent1]

    dataset_ts_sub1 = dataset_ts1[dataset_ts1 > dataset_percent1]

    extreme_fractions = []
    extreme_fractions.append(len(model_ts_sub1)/len(model_ts1) * 100)
    extreme_fractions.append(len(model_ts_sub2)/len(model_ts2) * 100)
    extreme_fractions.append(len(model_ts_sub3)/len(model_ts3) * 100)
    extreme_fractions.append(len(model_ts_sub4)/len(model_ts4) * 100)

    extreme_fractions.append(len(cordex_ts_sub1)/len(cordex_ts1) * 100)
    extreme_fractions.append(len(cordex_ts_sub2)/len(cordex_ts2) * 100)
    extreme_fractions.append(len(cordex_ts_sub3)/len(cordex_ts3) * 100)
    extreme_fractions.append(len(cordex_ts_sub4)/len(cordex_ts4) * 100)

    # cesm and cordex
    plot_data = [model_ts_sub1, model_ts_sub2, model_ts_sub3, model_ts_sub4,
                 cordex_ts_sub1, cordex_ts_sub2, cordex_ts_sub3, cordex_ts_sub4]

    ref_data = dataset_ts_sub1

    # frenquency distribution
    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' Precipitation exceeds '+str(round(dataset_percent1, 2))+'( '+str(percentile)+'th '+reg_name+') counts distribution over '+reg_name
    fname = 'vrcesm_prect_'+str(percentile)+'th_hist_extremes_vs_cordex_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, ref_data, percentile, colors, line_types, legends, varname, var_unit, title, outdir+fname, density=False, extremes=True, extreme_fractions=extreme_fractions)

    # cesm only
    plot_data = [model_ts_sub1, model_ts_sub2, model_ts_sub3, model_ts_sub4]

    ref_data = dataset_ts_sub1

    # frenquency distribution
    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' Precipitation exceeds '+str(round(dataset_percent1, 2))+'( '+str(percentile)+'th '+reg_name+') counts distribution over '+reg_name
    fname = 'vrcesm_prect_'+str(percentile)+'th_hist_extremes_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, ref_data, percentile, colors, line_types, legends, varname, var_unit, title, outdir+fname, density=False, extremes=True, extreme_fractions=extreme_fractions[:4])

    ############################################################################
    # plot month distribution of extreme events

    print('Counts for extreme events in each month...')

    ndatasets = len(legends)

    event_counts = np.zeros((ndatasets, len(months)))

    for idataset in range(ndatasets):
        tempset = datasets[idataset]
        # print(tempset)
        # print(tempset.index)
        # print(tempset.values.shape)
        # print(select_group.shape)

        for imon in range(len(months)):
            imon_data = tempset[(tempset.index.month == months[imon])]
            imon_data = imon_data.values[:, select_group]
            # print(imon_data.shape)

            imon_data_ts = imon_data.flatten()
            imon_data_ts = imon_data_ts[~np.isnan(imon_data_ts)]
            imon_data_ts = imon_data_ts[imon_data_ts > extreme_thres[idataset]]

            event_counts[idataset, imon] = len(imon_data_ts)

    print(event_counts)
    print(event_counts[0])

    # line plot for cesm and cordex
    xlabel = 'Month'
    ylabel = 'Counts'
    title = 'Group '+str(idx)+': '+str(iniyear) + ' to ' + str(endyear) + ' time distribution of CESM and CORDEX Precip '+str(percentile)+'th extreme events'
    fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_line_time_distribution_vs_cordex_group_'+str(idx)+'.pdf'
    plot_lines(months, event_counts, colors, line_types,
               legends, xlabel, ylabel, title, outdir+fname, xticks=months, xticknames=monnames)

    # line plot for cesm only
    plot_data = [event_counts[0], event_counts[1], event_counts[2], event_counts[3], event_counts[-1]]

    title = 'Group '+str(idx)+': '+str(iniyear) + ' to ' + str(endyear) + ' time distribution of CESM Precip '+str(percentile)+'th extreme events'
    fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_line_time_distribution_group_'+str(idx)+'.pdf'
    plot_lines(months, plot_data, cesm_colors, cesm_line_types,
               cesm_legends, xlabel, ylabel, title, outdir+fname, xticks=months, xticknames=monnames)

    # bar plot for cesm only
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar_width = 0.85/len(plot_data)
    index = months - bar_width*len(plot_data)/2
    opacity = 0.8

    for ibar in range(len(plot_data)):
        plt.bar(index+ibar*bar_width, plot_data[ibar], bar_width, alpha=opacity, color=cesm_colors[ibar], label=cesm_legends[ibar])

    plt.legend(handlelength=4, fontsize=5)

    plt.xticks(months, monnames, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)

    title = 'Group '+str(idx)+': '+str(iniyear) + ' to ' + str(endyear) + ' time distribution of CESM Precip '+str(percentile)+'th extreme events'
    fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_bar_time_distribution_group_'+str(idx)+'.pdf'

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)
