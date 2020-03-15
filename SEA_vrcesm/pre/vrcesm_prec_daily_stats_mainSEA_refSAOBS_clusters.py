# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-calculate extreme
# S3-plot contour
#
# Written by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.plot.mod_plt_regrid import data_regrid
from modules.datareader.mod_dataread_obs_pre import read_SAOBS_pre
from modules.plot.mod_plt_lines import plot_lines
from modules.plot.mod_plt_findstns import data_findstns
from modules.datareader.mod_dataread_vrcesm import readvrcesm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pickle
plt.switch_backend('agg')

############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/VRseasia_vs_ne30andfv09/pre/SA-OBS/'
kmeans_resdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/obs/SA-OBS/clustering/'

############################################################################
# set parameters
############################################################################
# variable info
var_unit = 'mm/day'


# time bounds
iniyear = 1990
endyear = 2005

# select number of clusters
ncluster = 3

# define regions
latbounds = [0, 25]
lonbounds = [90, 120]
reg_name = 'mainSEA'

# # mainland Southeast Asia
# reg_lats = [0, 20]
# reg_lons = [60, 130]

# set data frequency
frequency = 'day'

# set prect threshold
premin = 17
premax = 30

# 17mm is 95th percentile for all resolution, 39-30mm is the 99th percentile

# set percentile
percentile = 99

# select bins for histogram
nbins = 30

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monstr = 'Months'

# create Seasons for plot
seasons = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
seasnames = ['DJF', 'MAM', 'JJA', 'SON']
seastr = 'Seasons'

# plot legend
cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
                'CESM-fv1.9x2.5']

cesm_colors = ['red', 'yellow', 'green', 'blue']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed']

outdir = outdir+str(iniyear)+'-'+str(endyear)+'/'+str(iniyear)+'-'+str(endyear)+' kmeans '+str(ncluster)+' clusters/'
kmeans_resdir = kmeans_resdir+str(iniyear)+'-'+str(endyear)+'/'

############################################################################
# plotting functions
############################################################################


def plot_hist(plot_data, ref_data, percentile, colors, line_types, legends, varname, var_unit, title, fname, **kwargs):
    plt.clf()
    fig = plt.figure()

    ax = fig.add_subplot(111)

    if 'density' in kwargs:
        density = kwargs['density']
    else:
        density = True

    ref_hist, ref_edges = np.histogram(ref_data, bins=nbins, density=density)
    ref_bins = ref_edges[:-1]+(ref_edges[1]-ref_edges[0])/2

    for ii in range(len(plot_data)):
        tempdata = plot_data[ii]
        tempdata = tempdata[~np.isnan(tempdata)]
        temp_hist, temp_edges = np.histogram(tempdata, bins=ref_edges, density=density)
        ax.plot(ref_bins, temp_hist, c=colors[ii], linestyle=line_types[ii], linewidth=1.5, label=legends[ii])

    plt.legend(handlelength=4, fontsize=5)
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.set_xlabel(varname+' ['+var_unit+']', fontsize=5)

    if not density:
        ax.set_ylabel('Counts', fontsize=5)
    else:
        ax.set_yscale('log')
        ax.set_ylabel('Frequency', fontsize=5)

    ax.minorticks_off()

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.close(fig)


def plot_hist_bar(plot_data, bin_edges, bin_labels, colors, legends, varname, var_unit, title, fname, **kwargs):

    ndatasets = len(plot_data)
    bar_width = 0.8/ndatasets
    opacity = 0.8
    index = np.arange(len(bin_labels))

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if 'density' in kwargs:
        density = kwargs['density']
    else:
        density = True

    for ii in range(len(plot_data)):
        tempdata = plot_data[ii]
        tempdata = tempdata[~np.isnan(tempdata)]
        temp_hist, temp_edges = np.histogram(tempdata, bins=bin_edges, density=density)
        # print(index)
        # print(temp_hist)
        ax.bar(index+(bar_width/2-0.4+bar_width*ii), temp_hist, bar_width, color=colors[ii], label=legends[ii], alpha=opacity)

    ax.set_xticks(index)
    ax.set_xticklabels(bin_labels, fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_xlabel(varname+' ['+var_unit+']', fontsize=7)

    if not density:
        ax.set_ylabel('Counts', fontsize=7)
    else:
        ax.set_yscale('log')
        ax.set_ylabel('Frequency', fontsize=7)

    ax.minorticks_off()
    ax.legend(handlelength=4, fontsize=5)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.close(fig)


def plot_bars(plot_data, bin_labels, colors, legends, varname, var_unit, title, fname):
    ndatasets = len(plot_data)
    bar_width = 0.8/ndatasets
    opacity = 0.8
    index = np.arange(len(bin_labels))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for ii in range(len(plot_data)):
        ax.bar(index+(bar_width/2-0.4+bar_width*ii), plot_data[ii], bar_width, color=colors[ii], label=legends[ii], alpha=opacity)

    ax.set_xticks(index)
    ax.set_xticklabels(bin_labels, fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_xlabel(varname+' ['+var_unit+']', fontsize=7)
    ax.set_ylabel('As % of total precipitation', fontsize=7)
    ax.legend(fontsize=5)

    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.close(fig)


def plot_scatter(plot_datax, plot_datay, extreme_thres, xlabel, ylabel, colors, line_types, legends, varname, var_unit, title, fname, **kwargs):
    plt.clf()
    fig = plt.figure()

    for idx in range(len(plot_datax)):
        ax = fig.add_subplot(2, 2, idx+1)
        model_thres = round(extreme_thres[idx], 2)
        ax.set_title(legends[idx]+' \n '+str(percentile)+'th percentile: '+str(model_thres)+var_unit, fontsize=5, pad=2)

        ax.plot(plot_datax[idx], plot_datay[idx], 'o', color=colors[idx], markersize=1.5,)
        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_xlim(0, max(np.amax(plot_datax[idx]), np.amax(plot_datay[idx])))
        ax.set_ylim(0, max(np.amax(plot_datax[idx]), np.amax(plot_datay[idx])))
        if (len(plot_datax)-idx) <= 2:
            ax.set_xlabel(xlabel, fontsize=5)
        if (idx % (2) == 0):
            ax.set_ylabel(ylabel, fontsize=5, labelpad=0.7)

    fig.subplots_adjust(wspace=0.15, hspace=0.2)
    plt.suptitle(title, fontsize=7, y=0.95)
    plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.close(fig)


def plot_together(plot_data1, plot_data2, plot_data3, plot_data4, colors, legends, tsstr, ts, tsname, bin_labels, title, fname):
    plt.clf()
    fig = plt.figure()

    # Time distribution of counts of Extreme events
    ndatasets = len(plot_data1)
    bar_width = 0.8/ndatasets
    opacity = 0.8
    index = np.arange(len(ts))

    ax = fig.add_subplot(221)
    for ii in range(len(plot_data1)):
        ax.bar(index+(bar_width/2-0.4+bar_width*ii), plot_data1[ii], bar_width, color=colors[ii], label=legends[ii], alpha=opacity)

    ax.set_title('Time distribution of extreme events', fontsize=5)
    ax.set_xticks(index)
    ax.set_xticklabels(tsname, fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xlabel(tsstr, fontsize=5)
    ax.set_ylabel('Counts', fontsize=5)
    ax.legend(fontsize=5)

    # Climatological mean of extreme precc and precl
    ndatasets = len(plot_data2)
    bar_width = 0.8/ndatasets
    opacity = 0.8
    index = np.arange(len(ts))

    ax = fig.add_subplot(222)
    for ii in range(len(plot_data2)-1):
        ax.bar(index+(bar_width/2-0.4+bar_width*ii), plot_data2[ii][0], bar_width, color=colors[ii], label=legends[ii]+'-precc', edgecolor='black', linewidth=0.5, alpha=opacity, hatch='///////')
        ax.bar(index+(bar_width/2-0.4+bar_width*ii), plot_data2[ii][1], bar_width, bottom=plot_data2[ii][0], color=colors[ii], label=legends[ii]+'-precl', edgecolor='black',  linewidth=0.5, alpha=opacity)

    mpl.rcParams['hatch.linewidth'] = 0.05
    ax.bar(index+(bar_width/2-0.4+bar_width*(ndatasets-1)), plot_data2[-1], bar_width, color=colors[-1], label=legends[-1], alpha=opacity)

    ax.set_title('Climatological mean of extreme precip', fontsize=5)
    ax.set_xticks(index)
    ax.set_xticklabels(tsname, fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    # ax.set_xlabel(tsstr, fontsize=5)
    ax.set_ylabel('Precip [mm/day]', fontsize=5, labelpad=-0.5)
    ax.legend(['precc', 'precl'], handlelength=5, fontsize=5)

    # Histogram bar for precc
    ndatasets = len(plot_data3)
    bar_width = 0.8/ndatasets
    opacity = 0.8
    index = np.arange(len(bin_labels))

    ax = fig.add_subplot(223)
    for ii in range(len(plot_data3)):
        ax.bar(index+(bar_width/2-0.4+bar_width*ii), plot_data3[ii], bar_width, color=colors[ii], label=legends[ii], alpha=opacity)

    ax.set_title('Histogram of convective precip', fontsize=5)
    ax.set_xticks(index)
    ax.set_xticklabels(bin_labels, fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.set_xlabel('Convective Precip [mm/day]', fontsize=5)

    ax.set_yscale('log')
    ax.set_ylabel('Frequency', fontsize=5)

    ax.minorticks_off()
    # ax.legend(handlelength=4, fontsize=5)

    # Histogram bar for precl
    ndatasets = len(plot_data4)
    bar_width = 0.8/ndatasets
    opacity = 0.8
    index = np.arange(len(bin_labels))

    ax = fig.add_subplot(224)
    for ii in range(len(plot_data4)):
        ax.bar(index+(bar_width/2-0.4+bar_width*ii), plot_data4[ii], bar_width, color=colors[ii], label=legends[ii], alpha=opacity)

    ax.set_title('Histogram of large-scale precip', fontsize=5)
    ax.set_xticks(index)
    ax.set_xticklabels(bin_labels, fontsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    ax.set_xlabel('Large-scale Precip [mm/day]', fontsize=5)

    ax.set_yscale('log')
    # ax.set_ylabel('Frequency', fontsize=5, labelpad=-2)

    ax.minorticks_off()
    # ax.legend(handlelength=4, fontsize=5)

    fig.subplots_adjust(wspace=0.15, hspace=0.2)
    plt.suptitle(title, fontsize=7, y=0.95)
    plt.savefig(fname, bbox_inches='tight', dpi=600)
    plt.close(fig)


def get_monmean(var, ts):
    res = []

    for itime in ts:
        if np.isscalar(itime):
            temp = var.values[var.index.month == itime, :].flatten()
        else:
            temp = var.values[np.in1d(var.index.month, itime), :].flatten()

        temp = temp[~np.isnan(temp)]
        res.append(np.mean(temp))

    return res


def get_extremean(var, ref_var, threshold, ts):
    res = []

    for itime in ts:
        if np.isscalar(itime):
            temp = var.values[var.index.month == itime, :].flatten()
            ref_temp = ref_var.values[ref_var.index.month == itime, :].flatten()
        else:
            temp = var.values[np.in1d(var.index.month, itime), :].flatten()
            ref_temp = ref_var.values[np.in1d(ref_var.index.month, itime), :].flatten()

        select_data = (~np.isnan(temp)) & (~np.isnan(ref_temp))
        temp = temp[select_data]
        ref_temp = ref_temp[select_data]

        temp = temp[ref_temp > threshold]
        if len(temp) == 0:
            res.append(0.)
        else:
            res.append(np.mean(temp))

    return res


def get_extrecounts(var, threshold, ts):
    res = []

    for itime in ts:
        if np.isscalar(itime):
            temp = var.values[var.index.month == itime, :].flatten()
        else:
            temp = var.values[np.in1d(var.index.month, itime), :].flatten()

        # print(var[np.in1d(var.index.month, itime)])

        temp = temp[~np.isnan(temp)]
        res.append(len(temp[temp > threshold]))

    return res


############################################################################
# read data
############################################################################


# read vrcesm
print('Reading VRCESM data...')

resolution = 'fv02'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
precc_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    'PRECC', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
precl_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    'PRECL', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
prect_var1, model_time1, model_lats1, model_lons1 = readvrcesm(
    'PRECT', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
precc_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    'PRECC', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
precl_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    'PRECL', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
prect_var2, model_time2, model_lats2, model_lons2 = readvrcesm(
    'PRECT', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
precc_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    'PRECC', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
precl_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    'PRECL', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
prect_var3, model_time3, model_lats3, model_lons3 = readvrcesm(
    'PRECT', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
precc_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    'PRECC', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
precl_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    'PRECL', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])
prect_var4, model_time4, model_lats4, model_lons4 = readvrcesm(
    'PRECT', iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

print(prect_var1.shape)

# read Observations

print('Reading SA-OBS data...')

ref_name = 'SA-OBS'

# read SA-OBS
version = 'countries'
countries = ['Thailand', 'Vietnam', 'Myanmar', 'Cambodia']
dataset1, obs_var1, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_SAOBS_pre(
    version, iniyear, endyear, countries, missing_ratio=10, ignore_years=[1999])

print(obs_var1.shape)

precc_var1 = 86400 * 1000 * data_findstns(precc_var1, model_time1, model_lats1,
                                          model_lons1, obs_var1, stnlats, stnlons, stnnames)
precc_var2 = 86400 * 1000 * data_findstns(precc_var2, model_time2, model_lats2,
                                          model_lons2, obs_var1, stnlats, stnlons, stnnames)
precc_var3 = 86400 * 1000 * data_findstns(precc_var3, model_time3, model_lats3,
                                          model_lons3, obs_var1, stnlats, stnlons, stnnames)
precc_var4 = 86400 * 1000 * data_findstns(precc_var4, model_time4, model_lats4,
                                          model_lons4, obs_var1, stnlats, stnlons, stnnames)

precl_var1 = 86400 * 1000 * data_findstns(precl_var1, model_time1, model_lats1,
                                          model_lons1, obs_var1, stnlats, stnlons, stnnames)
precl_var2 = 86400 * 1000 * data_findstns(precl_var2, model_time2, model_lats2,
                                          model_lons2, obs_var1, stnlats, stnlons, stnnames)
precl_var3 = 86400 * 1000 * data_findstns(precl_var3, model_time3, model_lats3,
                                          model_lons3, obs_var1, stnlats, stnlons, stnnames)
precl_var4 = 86400 * 1000 * data_findstns(precl_var4, model_time4, model_lats4,
                                          model_lons4, obs_var1, stnlats, stnlons, stnnames)

prect_var1 = 86400 * 1000 * data_findstns(prect_var1, model_time1, model_lats1,
                                          model_lons1, obs_var1, stnlats, stnlons, stnnames)
prect_var2 = 86400 * 1000 * data_findstns(prect_var2, model_time2, model_lats2,
                                          model_lons2, obs_var1, stnlats, stnlons, stnnames)
prect_var3 = 86400 * 1000 * data_findstns(prect_var3, model_time3, model_lats3,
                                          model_lons3, obs_var1, stnlats, stnlons, stnnames)
prect_var4 = 86400 * 1000 * data_findstns(prect_var4, model_time4, model_lats4,
                                          model_lons4, obs_var1, stnlats, stnlons, stnnames)

############################################################################
# read clustering results
############################################################################

res_labels = pickle.load(open(kmeans_resdir+'SA-OBS_'+str(iniyear)+'to' +
                              str(endyear)+'_kmeans_clustering_results.p', "rb"))
kmeans_labels = res_labels[str(ncluster)].values


############################################################################
# plot histogram for total precc and precl
############################################################################

for idx in range(ncluster):
    print('Plotting for group'+str(idx)+'...')

    select_group = (kmeans_labels == idx)

    precc_var_gp1 = precc_var1.loc[:, stnnames[select_group]]
    precc_var_gp2 = precc_var2.loc[:, stnnames[select_group]]
    precc_var_gp3 = precc_var3.loc[:, stnnames[select_group]]
    precc_var_gp4 = precc_var4.loc[:, stnnames[select_group]]
    print(precc_var_gp1)

    precl_var_gp1 = precl_var1.loc[:, stnnames[select_group]]
    precl_var_gp2 = precl_var2.loc[:, stnnames[select_group]]
    precl_var_gp3 = precl_var3.loc[:, stnnames[select_group]]
    precl_var_gp4 = precl_var4.loc[:, stnnames[select_group]]

    prect_var_gp1 = prect_var1.loc[:, stnnames[select_group]]
    prect_var_gp2 = prect_var2.loc[:, stnnames[select_group]]
    prect_var_gp3 = prect_var3.loc[:, stnnames[select_group]]
    prect_var_gp4 = prect_var4.loc[:, stnnames[select_group]]

    dataset_gp1 = dataset1.loc[:, stnnames[select_group]]

    precc_ts1 = precc_var_gp1.values.flatten()
    precc_ts2 = precc_var_gp2.values.flatten()
    precc_ts3 = precc_var_gp3.values.flatten()
    precc_ts4 = precc_var_gp4.values.flatten()

    precl_ts1 = precl_var_gp1.values.flatten()
    precl_ts2 = precl_var_gp2.values.flatten()
    precl_ts3 = precl_var_gp3.values.flatten()
    precl_ts4 = precl_var_gp4.values.flatten()

    prect_ts1 = prect_var_gp1.values.flatten()
    prect_ts2 = prect_var_gp2.values.flatten()
    prect_ts3 = prect_var_gp3.values.flatten()
    prect_ts4 = prect_var_gp4.values.flatten()

    dataset_ts1 = dataset_gp1.values.flatten()

    precc_ts1 = precc_ts1[~np.isnan(precc_ts1)]
    precc_ts2 = precc_ts2[~np.isnan(precc_ts2)]
    precc_ts3 = precc_ts3[~np.isnan(precc_ts3)]
    precc_ts4 = precc_ts4[~np.isnan(precc_ts4)]

    precl_ts1 = precl_ts1[~np.isnan(precl_ts1)]
    precl_ts2 = precl_ts2[~np.isnan(precl_ts2)]
    precl_ts3 = precl_ts3[~np.isnan(precl_ts3)]
    precl_ts4 = precl_ts4[~np.isnan(precl_ts4)]

    prect_ts1 = prect_ts1[~np.isnan(prect_ts1)]
    prect_ts2 = prect_ts2[~np.isnan(prect_ts2)]
    prect_ts3 = prect_ts3[~np.isnan(prect_ts3)]
    prect_ts4 = prect_ts4[~np.isnan(prect_ts4)]
    dataset_ts1 = dataset_ts1[~np.isnan(dataset_ts1)]

    prect_percent1 = np.percentile(prect_ts1, percentile)
    prect_percent2 = np.percentile(prect_ts2, percentile)
    prect_percent3 = np.percentile(prect_ts3, percentile)
    prect_percent4 = np.percentile(prect_ts4, percentile)
    dataset_percent1 = np.percentile(dataset_ts1, percentile)
    extreme_thres = [prect_percent1, prect_percent2, prect_percent3, prect_percent4, dataset_percent1]

    print('Plotting histogram for total precc and precl...')

    # precc
    varname = 'Convective Precip'
    varstr = 'precc'
    var_unit = 'mm/day'
    plot_data = [precc_ts1, precc_ts2, precc_ts3, precc_ts4]

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+varstr+'_'+reg_name+'_hist_refSAOBS_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, prect_ts1, percentile, cesm_colors, cesm_line_types,
              cesm_legends, varname, var_unit, title, outdir+fname, density=True)

    # precl
    varname = 'Large-scale Precip'
    varstr = 'precl'
    var_unit = 'mm/day'
    plot_data = [precl_ts1, precl_ts2, precl_ts3, precl_ts4]

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+varstr+'_'+reg_name+'_hist_refSAOBS_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, prect_ts1, percentile, cesm_colors, cesm_line_types,
              cesm_legends, varname, var_unit, title, outdir+fname, density=True)

    # precc and precl together
    varname = 'Precip'
    varstr = 'precc_and_precl'
    var_unit = 'mm/day'
    plot_data = [precc_ts1, precc_ts2, precc_ts3, precc_ts4, precl_ts1, precl_ts2, precl_ts3, precl_ts4]

    legends = ['precc-CESM-vrseasia', 'precc-CESM-ne30', 'precc-CESM-fv0.9x1.25', 'precc-CESM-fv1.9x2.5',
               'precl-CESM-vrseasia', 'precl-CESM-ne30', 'precl-CESM-fv0.9x1.25', 'precl-CESM-fv1.9x2.5']
    colors = ['red', 'yellow', 'green', 'blue', 'red', 'yellow', 'green', 'blue']
    line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-', '-', '-', '-']
    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+varstr+'_'+reg_name+'_hist_refSAOBS_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, prect_ts1, percentile, colors, line_types,
              legends, varname, var_unit, title, outdir+fname, density=True)

    ############################################################################
    # plot histogram bar for total prect
    ############################################################################

    varname = 'Total Precip'
    varstr = 'prect'
    var_unit = 'mm/day'
    plot_data = [prect_ts1, prect_ts2, prect_ts3, prect_ts4, dataset_ts1]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
               'CESM-fv1.9x2.5', 'SA-OBS']

    colors = ['red', 'yellow', 'green', 'blue', 'black']

    bin_edges = [0, 1, 5, 10, 20, 40, np.amax(dataset_ts1)+1]
    bin_labels = ['<1', '1-5', '5-10', '10-20', '20-40', '>40']

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+varstr+'_'+reg_name+'_hist_bar_refSAOBS_group_'+str(idx)+'.pdf'

    plot_hist_bar(plot_data, bin_edges, bin_labels, colors, legends, varname, var_unit, title, outdir+fname)

    ############################################################################
    # plot contribution bar for total prect
    ############################################################################
    varname = 'Total Precip'
    varstr = 'prect'
    var_unit = 'mm/day'
    vars = [prect_ts1, prect_ts2, prect_ts3, prect_ts4, dataset_ts1]

    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
               'CESM-fv1.9x2.5', 'SA-OBS']

    colors = ['red', 'yellow', 'green', 'blue', 'black']

    bin_edges = [0, 1, 5, 10, 20, 40, np.amax(dataset_ts1)+1]
    bin_labels = ['<1', '1-5', '5-10', '10-20', '20-40', '>40']

    plot_data = []
    for idata in vars:
        res = []
        for idx_bin in range(len(bin_edges)-1):
            temp = idata[(idata >= bin_edges[idx_bin]) & (idata < bin_edges[idx_bin+1])]
            res.append(np.sum(temp)/np.sum(idata)*100)

        plot_data.append(res)

    print(plot_data)
    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+varstr+'_'+reg_name+'_contribution_bar_refSAOBS_group_'+str(idx)+'.pdf'

    plot_bars(plot_data, bin_labels, colors, legends, varname, var_unit, title, outdir+fname)

    ############################################################################
    # plot histogram for extreme precc and precl
    ############################################################################

    print('Plotting histogram for Extreme precipitation...')

    precc_ts_sub1 = precc_ts1[prect_ts1 > prect_percent1]
    precc_ts_sub2 = precc_ts2[prect_ts2 > prect_percent2]
    precc_ts_sub3 = precc_ts3[prect_ts3 > prect_percent3]
    precc_ts_sub4 = precc_ts4[prect_ts4 > prect_percent4]

    precl_ts_sub1 = precl_ts1[prect_ts1 > prect_percent1]
    precl_ts_sub2 = precl_ts2[prect_ts2 > prect_percent2]
    precl_ts_sub3 = precl_ts3[prect_ts3 > prect_percent3]
    precl_ts_sub4 = precl_ts4[prect_ts4 > prect_percent4]

    ref_data = np.concatenate((precc_ts_sub1, precc_ts_sub2, precc_ts_sub3, precc_ts_sub4,
                               precl_ts_sub1, precl_ts_sub2, precl_ts_sub3, precl_ts_sub4), axis=None)
    print(ref_data.shape)

    # precc
    varname = 'Convective Precip'
    varstr = 'precc'
    var_unit = 'mm/day'
    plot_data = [precc_ts_sub1, precc_ts_sub2, precc_ts_sub3, precc_ts_sub4]

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+str(percentile)+'th '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_tp_'+varstr+'_'+reg_name+'_hist_refSAOBS_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, ref_data, percentile, cesm_colors, cesm_line_types,
              cesm_legends, varname, var_unit, title, outdir+fname, density=False)

    # precl
    varname = 'Large-scale Precip'
    varstr = 'precl'
    var_unit = 'mm/day'
    plot_data = [precl_ts_sub1, precl_ts_sub2, precl_ts_sub3, precl_ts_sub4]

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+str(percentile)+'th '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_tp_'+varstr+'_'+reg_name+'_hist_refSAOBS_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, ref_data, percentile, cesm_colors, cesm_line_types,
              cesm_legends, varname, var_unit, title, outdir+fname, density=False)

    # precc and precl together
    varname = 'Precip'
    varstr = 'precc_and_precl'
    var_unit = 'mm/day'
    plot_data = [precc_ts_sub1, precc_ts_sub2, precc_ts_sub3, precc_ts_sub4,
                 precl_ts_sub1, precl_ts_sub2, precl_ts_sub3, precl_ts_sub4]

    legends = ['precc-CESM-vrseasia', 'precc-CESM-ne30', 'precc-CESM-fv0.9x1.25', 'precc-CESM-fv1.9x2.5',
               'precl-CESM-vrseasia', 'precl-CESM-ne30', 'precl-CESM-fv0.9x1.25', 'precl-CESM-fv1.9x2.5']
    colors = ['red', 'yellow', 'green', 'blue', 'red', 'yellow', 'green', 'blue']
    line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-', '-', '-', '-']

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+str(percentile)+'th '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_tp_'+varstr+'_'+reg_name+'_hist_refSAOBS_group_'+str(idx)+'.pdf'

    plot_hist(plot_data, ref_data, percentile, colors, line_types,
              legends, varname, var_unit, title, outdir+fname, density=False)

    ############################################################################
    # plot scattering for precip
    ############################################################################

    prect_ts_sub1 = prect_ts1[prect_ts1 > prect_percent1]
    prect_ts_sub2 = prect_ts2[prect_ts2 > prect_percent2]
    prect_ts_sub3 = prect_ts3[prect_ts3 > prect_percent3]
    prect_ts_sub4 = prect_ts4[prect_ts4 > prect_percent4]

    # precc vs precl
    print('Scatter plot for extreme precc vs precl..')
    varstr = 'precc_vs_precl'
    xlabel = 'Convective Precip'
    ylabel = 'Large-scale Precip'

    plot_datax = [precc_ts_sub1, precc_ts_sub2, precc_ts_sub3, precc_ts_sub4]
    plot_datay = [precl_ts_sub1, precl_ts_sub2, precl_ts_sub3, precl_ts_sub4]

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+str(percentile)+'th '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_tp_'+varstr+'_'+reg_name+'_scatter_refSAOBS_group_'+str(idx)+'.png'

    plot_scatter(plot_datax, plot_datay, extreme_thres, xlabel, ylabel, cesm_colors, cesm_line_types,
                 cesm_legends, varname, var_unit, title, outdir+fname)

    # prect vs precl
    print('Scatter plot for extreme prect vs precl..')
    varstr = 'prect_vs_precl'
    xlabel = 'Total Precip'
    ylabel = 'Large-scale Precip'

    plot_datax = [prect_ts_sub1, prect_ts_sub2, prect_ts_sub3, prect_ts_sub4]
    plot_datay = [precl_ts_sub1, precl_ts_sub2, precl_ts_sub3, precl_ts_sub4]

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+str(percentile)+'th '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_tp_'+varstr+'_'+reg_name+'_scatter_refSAOBS_group_'+str(idx)+'.png'

    plot_scatter(plot_datax, plot_datay, extreme_thres, xlabel, ylabel, cesm_colors, cesm_line_types,
                 cesm_legends, varname, var_unit, title, outdir+fname)

    # prect vs precc
    print('Scatter plot for extreme prect vs precc..')
    varstr = 'prect_vs_precc'
    xlabel = 'Total Precip'
    ylabel = 'Convective Precip'

    plot_datax = [prect_ts_sub1, prect_ts_sub2, prect_ts_sub3, prect_ts_sub4]
    plot_datay = [precc_ts_sub1, precc_ts_sub2, precc_ts_sub3, precc_ts_sub4]

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' '+str(percentile)+'th '+varname+' distribution over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_tp_'+varstr+'_'+reg_name+'_scatter_refSAOBS_group_'+str(idx)+'.png'

    plot_scatter(plot_datax, plot_datay, extreme_thres, xlabel, ylabel, cesm_colors, cesm_line_types,
                 cesm_legends, varname, var_unit, title, outdir+fname)

    ############################################################################
    # plot climatological mean
    ############################################################################

    print('Plot climatological mean for precc and precl')

    # precc
    vars = [precc_var_gp1, precc_var_gp2, precc_var_gp3, precc_var_gp4]
    plot_data = []
    for idata in vars:
        # print(idata[idata.index.month == 1])
        res = get_monmean(idata, months)
        plot_data.append(res)

    # print(plot_data)

    varname = 'Convective Precip'
    varstr = 'precc'

    xlabel = 'Month'
    ylabel = varname

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' monthly mean '+varname+' over '+reg_name
    fname = 'vrseasia_'+varstr+'_'+reg_name+'_clim_line_refSAOBS_group_'+str(idx)+'.pdf'

    plot_lines(months, plot_data, cesm_colors, cesm_line_types, cesm_legends, xlabel, ylabel, title, outdir+fname, xticks=months, xticknames=monnames)

    # precc extrems
    vars = [precc_var_gp1, precc_var_gp2, precc_var_gp3, precc_var_gp4]
    ref_vars = [prect_var_gp1, prect_var_gp2, prect_var_gp3, prect_var_gp4]

    plot_data = []
    for idx_data, idata in enumerate(vars):
        res = get_extremean(idata, ref_vars[idx_data], extreme_thres[idx_data], months)
        plot_data.append(res)

    # print(plot_data)

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' monthly mean '+' '+str(percentile)+'th '+varname+' over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_tp_'+varstr+'_'+reg_name+'_clim_line_refSAOBS_group_'+str(idx)+'.pdf'

    plot_lines(months, plot_data, cesm_colors, cesm_line_types, cesm_legends, xlabel, ylabel, title, outdir+fname, xticks=months, xticknames=monnames)

    # precl
    vars = [precl_var_gp1, precl_var_gp2, precl_var_gp3, precl_var_gp4]
    plot_data = []
    for idata in vars:
        # print(idata[idata.index.month == 1])
        res = get_monmean(idata, months)
        plot_data.append(res)

    # print(plot_data)

    varname = 'Large-scale Precip'
    varstr = 'precl'

    xlabel = 'Month'
    ylabel = varname

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' monthly mean '+varname+' over '+reg_name
    fname = 'vrseasia_'+varstr+'_'+reg_name+'_clim_line_refSAOBS_group_'+str(idx)+'.pdf'

    plot_lines(months, plot_data, cesm_colors, cesm_line_types, cesm_legends, xlabel, ylabel, title, outdir+fname, xticks=months, xticknames=monnames)

    # precc extrems
    vars = [precl_var_gp1, precl_var_gp2, precl_var_gp3, precl_var_gp4]
    ref_vars = [prect_var_gp1, prect_var_gp2, prect_var_gp3, prect_var_gp4]

    plot_data = []
    for idx_data, idata in enumerate(vars):
        res = get_extremean(idata, ref_vars[idx_data], extreme_thres[idx_data], months)
        plot_data.append(res)

    # print(plot_data)

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' monthly mean '+' '+str(percentile)+'th '+varname+' over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_tp_'+varstr+'_'+reg_name+'_clim_line_refSAOBS_group_'+str(idx)+'.pdf'

    plot_lines(months, plot_data, cesm_colors, cesm_line_types, cesm_legends, xlabel, ylabel, title, outdir+fname, xticks=months, xticknames=monnames)

    ############################################################################
    # plot 99th observed precip together
    ############################################################################
    plt.clf()
    fig = plt.figure()

    ts = seasons
    tsname = seasnames
    tsstr = seastr

    # Monthly counts of Extreme events
    ax = fig.add_subplot(221)

    vars = [prect_var_gp1, prect_var_gp2, prect_var_gp3, prect_var_gp4, dataset_gp1]

    plot_data1 = []
    for idx_data, idata in enumerate(vars):
        res = get_extrecounts(idata, extreme_thres[-1], ts)
        plot_data1.append(res)

    # print(plot_data1)

    # Climatological mean of extreme precc and precl
    var_precc = [precc_var_gp1, precc_var_gp2, precc_var_gp3, precc_var_gp4]
    var_precl = [precl_var_gp1, precl_var_gp2, precl_var_gp3, precl_var_gp4]
    ref_vars = [prect_var_gp1, prect_var_gp2, prect_var_gp3, prect_var_gp4]

    plot_data2 = []
    for idx_data, idata in enumerate(ref_vars):
        res_precc = get_extremean(var_precc[idx_data], idata, extreme_thres[-1], ts)
        res_precl = get_extremean(var_precl[idx_data], idata, extreme_thres[-1], ts)

        # res_prect = get_extremean(idata, idata, extreme_thres[idx_data], months)
        # print(np.array(res_precc)+np.array(res_precl))
        # print(res_prect)

        plot_data2.append([res_precc, res_precl])

    res_obs = get_extremean(dataset_gp1, dataset_gp1, extreme_thres[-1], ts)
    plot_data2.append(res_obs)

    # print(plot_data2)

    # Histogram bar of precc

    vars = [precc_ts1, precc_ts2, precc_ts3, precc_ts4]
    ref_vars = [prect_ts1, prect_ts2, prect_ts3, prect_ts4]

    bin_edges = [0, 1, 5, 10, 20, 50, np.amax(dataset_ts1)+1]
    bin_labels = ['<1', '1-5', '5-10', '10-20', '20-50', '>50']

    plot_data3 = []
    for idx_data, idata in enumerate(vars):
        ref_data = ref_vars[idx_data]
        tempdata = idata[~np.isnan(idata)]
        ref_data = ref_data[~np.isnan(ref_data)]
        tempdata = tempdata[ref_data > extreme_thres[-1]]
        temp_hist, temp_edges = np.histogram(tempdata, bins=bin_edges, density=True)
        plot_data3.append(temp_hist)

    # Histogram bar of precl

    vars = [precl_ts1, precl_ts2, precl_ts3, precl_ts4]
    ref_vars = [prect_ts1, prect_ts2, prect_ts3, prect_ts4]

    bin_edges = [0, 1, 5, 10, 20, 50, np.amax(dataset_ts1)+1]
    bin_labels = ['<1', '1-5', '5-10', '10-20', '20-50', '>50']

    plot_data4 = []
    for idx_data, idata in enumerate(vars):
        ref_data = ref_vars[idx_data]
        tempdata = idata[~np.isnan(idata)]
        ref_data = ref_data[~np.isnan(ref_data)]
        tempdata = tempdata[ref_data > extreme_thres[-1]]
        temp_hist, temp_edges = np.histogram(tempdata, bins=bin_edges, density=True)
        plot_data4.append(temp_hist)

    # create plot
    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
               'CESM-fv1.9x2.5', 'SA-OBS']

    colors = ['red', 'yellow', 'green', 'blue', 'black']

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' climatological mean of observed '+str(percentile)+'th precip over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_SAOBStp_summary_'+reg_name+'_refSAOBS_group_'+str(idx)+'.pdf'

    plot_together(plot_data1, plot_data2, plot_data3, plot_data4, colors, legends, tsstr, ts, tsname, bin_labels, title, outdir+fname)

    ############################################################################
    # plot 99th model precip together
    ############################################################################
    plt.clf()
    fig = plt.figure()

    ts = seasons
    tsname = seasnames
    tsstr = seastr

    # Monthly counts of Extreme events
    ax = fig.add_subplot(221)

    vars = [prect_var_gp1, prect_var_gp2, prect_var_gp3, prect_var_gp4, dataset_gp1]

    plot_data1 = []
    for idx_data, idata in enumerate(vars):
        res = get_extrecounts(idata, extreme_thres[idx_data], ts)
        plot_data1.append(res)

    # print(plot_data1)

    # Climatological mean of extreme precc and precl
    var_precc = [precc_var_gp1, precc_var_gp2, precc_var_gp3, precc_var_gp4]
    var_precl = [precl_var_gp1, precl_var_gp2, precl_var_gp3, precl_var_gp4]
    ref_vars = [prect_var_gp1, prect_var_gp2, prect_var_gp3, prect_var_gp4]

    plot_data2 = []
    for idx_data, idata in enumerate(ref_vars):
        res_precc = get_extremean(var_precc[idx_data], idata, extreme_thres[idx_data], ts)
        res_precl = get_extremean(var_precl[idx_data], idata, extreme_thres[idx_data], ts)

        # res_prect = get_extremean(idata, idata, extreme_thres[idx_data], months)
        # print(np.array(res_precc)+np.array(res_precl))
        # print(res_prect)

        plot_data2.append([res_precc, res_precl])

    res_obs = get_extremean(dataset_gp1, dataset_gp1, extreme_thres[-1], ts)
    plot_data2.append(res_obs)

    # print(plot_data2)

    # Histogram bar of precc

    vars = [precc_ts1, precc_ts2, precc_ts3, precc_ts4]
    ref_vars = [prect_ts1, prect_ts2, prect_ts3, prect_ts4]

    bin_edges = [0, 1, 5, 10, 20, 50, np.amax(dataset_ts1)+1]
    bin_labels = ['<1', '1-5', '5-10', '10-20', '20-50', '>50']

    plot_data3 = []
    for idx_data, idata in enumerate(vars):
        ref_data = ref_vars[idx_data]
        tempdata = idata[~np.isnan(idata)]
        ref_data = ref_data[~np.isnan(ref_data)]
        tempdata = tempdata[ref_data > extreme_thres[idx_data]]
        temp_hist, temp_edges = np.histogram(tempdata, bins=bin_edges, density=True)
        plot_data3.append(temp_hist)

    # Histogram bar of precl

    vars = [precl_ts1, precl_ts2, precl_ts3, precl_ts4]
    ref_vars = [prect_ts1, prect_ts2, prect_ts3, prect_ts4]

    bin_edges = [0, 1, 5, 10, 20, 50, np.amax(dataset_ts1)+1]
    bin_labels = ['<1', '1-5', '5-10', '10-20', '20-50', '>50']

    plot_data4 = []
    for idx_data, idata in enumerate(vars):
        ref_data = ref_vars[idx_data]
        tempdata = idata[~np.isnan(idata)]
        ref_data = ref_data[~np.isnan(ref_data)]
        tempdata = tempdata[ref_data > extreme_thres[idx_data]]
        temp_hist, temp_edges = np.histogram(tempdata, bins=bin_edges, density=True)
        plot_data4.append(temp_hist)

    # create plot
    legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25',
               'CESM-fv1.9x2.5', 'SA-OBS']

    colors = ['red', 'yellow', 'green', 'blue', 'black']

    title = 'Group '+str(idx)+': '+str(iniyear)+' to '+str(endyear)+' climatological mean of  simulated '+str(percentile)+'th precip over '+reg_name
    fname = 'vrseasia_'+str(percentile)+'th_modeltp_summary_'+reg_name+'_refSAOBS_group_'+str(idx)+'.pdf'

    plot_together(plot_data1, plot_data2, plot_data3, plot_data4, colors, legends, tsstr, ts, tsname, bin_labels, title, outdir+fname)
