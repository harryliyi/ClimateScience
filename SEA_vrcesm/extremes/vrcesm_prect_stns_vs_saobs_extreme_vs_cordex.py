# This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
# Several steps are inplemented:
# S1-read precip data from vrcesm and CORDEX-SEA
# S2-plot basic analysis
# S3-calculate and plot extreme
#
# Written by Harry Li

# import libraries

import pathmagic  # noqa: F401
from modules.stats.mod_stats_extremes import cdfcheck, pdfcheck, gpdfit_moment
from modules.datareader.mod_dataread_obs_pre import read_SAOBS_pre
from modules.plot.mod_plt_lines import plot_lines
from modules.plot.mod_plt_findstns import data_findstns
from modules.datareader.mod_dataread_vrcesm import readvrcesm
from modules.datareader.mod_dataread_cordex_sea import readcordex
from scipy.stats import genpareto as gpd
import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# set data frequency
frequency = 'day'

# create months for plot
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# percentile
percentile = 99

# set up nbins
nbins = 50

outdir = outdir+str(iniyear)+'-'+str(endyear)+'/'+str(iniyear)+'-'+str(endyear)+' kmeans '+str(ncluster)+' clusters/'
kmeans_resdir = kmeans_resdir+str(iniyear)+'-'+str(endyear)+'/'

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

        ax1 = plt.subplot(3, 1, 1)
        x, xbins, ypdf = pdfcheck(temp_data, nbins, temp_fit)
        anderson = ss.anderson_ksamp([temp_data, gpd.rvs(temp_fit[0], 0., temp_fit[1], size=5000)])
        ax1.plot(x, ypdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
        ax1.hist(temp_data, bins=xbins, alpha=0.5, density=True, label='Empirical')
        ax1.set_title('pdf check, ks p-value ='+str(tmep_ks.pvalue), fontsize=5, pad=2)
        ax1.legend(loc='upper right', fontsize=8)
        # set x/y tick label size
        ax1.xaxis.set_tick_params(labelsize=4)
        ax1.yaxis.set_tick_params(labelsize=4)

        ax2 = fig.add_subplot(3, 1, 2)
        temp_res = ss.probplot(temp_data, dist=ss.genpareto, sparams=(temp_fit[0], 0., temp_fit[1]))
        ax2.scatter(temp_res[0][0], temp_res[0][1], c='b', s=3)
        ax2.plot(temp_res[0][0], temp_res[0][0], linestyle='solid', linewidth=1.5, color='r')
        ax2.set_title('qq-plot', fontsize=5, pad=2)
        # ax2.legend(loc='upper right', fontsize=8)
        # set x/y tick label size
        ax2.xaxis.set_tick_params(labelsize=4)
        ax2.yaxis.set_tick_params(labelsize=4)
        ax2.set_xlabel('Theratical quantiles', fontsize=4)
        ax2.set_ylabel('Ordered values', fontsize=4)

        ax3 = fig.add_subplot(3, 1, 3)
        x, xbins, ycdf = cdfcheck(temp_data, nbins, temp_fit)
        ax3.plot(x, ycdf, linestyle='solid', linewidth=1.5, label='GPD Fit')
        ax3.hist(temp_data, bins=xbins, alpha=0.5, density=True, histtype='step', cumulative=True, label='Empirical')
        anderson = ss.anderson_ksamp([temp_data, gpd.rvs(temp_fit[0], 0., temp_fit[1], size=5000)])
        ax3.set_title('cdf check, AD sig level='+str(anderson.significance_level), fontsize=5, pad=2)
        ax3.legend(loc='lower right', fontsize=8)
        # set x/y tick label size
        ax3.xaxis.set_tick_params(labelsize=4)
        ax3.yaxis.set_tick_params(labelsize=4)
        ax3.set_xlabel('Probability', fontsize=4)
        ax3.set_ylabel('Precip (mm/day)', fontsize=4)

        title = 'Gourp: '+str(groupidx)+' '+str(iniyear)+' to '+str(endyear)+' ' + \
            legends[idx]+' '+str(percentile)+'th percentile precip GPD fit'
        plt.suptitle(title, fontsize=10, y=0.95)

        fig.subplots_adjust(hspace=0.2)
        fname = str(percentile)+'th_extremes_GPD_fit_group'+str(groupidx)+'_'+str(idx+1)+'.pdf'
        plt.savefig(outdir+fname, bbox_inches='tight')
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

print(cordex_var1.shape)
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
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed',  '-']

############################################################################
# cluster the data
############################################################################
# plot annual maximum precip for each station
cordex_annmax1 = cordex_var1.resample('A').max()
cordex_annmax2 = cordex_var2.resample('A').max()
cordex_annmax3 = cordex_var3.resample('A').max()
cordex_annmax4 = cordex_var4.resample('A').max()

model_annmax1 = model_var1.resample('A').max()
model_annmax2 = model_var2.resample('A').max()
model_annmax3 = model_var3.resample('A').max()
model_annmax4 = model_var4.resample('A').max()

dataset_annmax1 = dataset1.resample('A').max()

# plot annual maximum precip for each station
cordex_annmean1 = cordex_var1.resample('A').mean()
cordex_annmean2 = cordex_var2.resample('A').mean()
cordex_annmean3 = cordex_var3.resample('A').mean()
cordex_annmean4 = cordex_var4.resample('A').mean()

model_annmean1 = model_var1.resample('A').mean()
model_annmean2 = model_var2.resample('A').mean()
model_annmean3 = model_var3.resample('A').mean()
model_annmean4 = model_var4.resample('A').mean()

dataset_annmean1 = dataset1.resample('A').mean()

# plot monthly mean precip for each station
cordex_monmean1 = cordex_var1.resample('M').mean()
cordex_monmean2 = cordex_var2.resample('M').mean()
cordex_monmean3 = cordex_var3.resample('M').mean()
cordex_monmean4 = cordex_var4.resample('M').mean()

model_monmean1 = model_var1.resample('M').mean()
model_monmean2 = model_var2.resample('M').mean()
model_monmean3 = model_var3.resample('M').mean()
model_monmean4 = model_var4.resample('M').mean()

dataset_monmean1 = dataset1.resample('M').mean()


'''
#clustering
dataset_var = dataset1.values
print(dataset_var.shape)
fname = 'SAOBS_prect_kmeans_nclusters'
ncluster = 5
kmeans = kmeans_cluster(dataset_var,ncluster,stnlats,stnlons,outdir+fname)
print(kmeans.labels_)

index = np.arange(9)
bar_width = 0.8
opacity = 0.8
shape_type = ['','','','','..','..','..','..','//']

for idx in range(ncluster):
    select_group = (kmeans.labels_==idx)

    #annual mean
    cordex_annmean_mean1 = np.mean(np.mean(cordex_annmean1.values[:,select_group],axis=1),axis=0)
    cordex_annmean_mean2 = np.mean(np.mean(cordex_annmean2.values[:,select_group],axis=1),axis=0)
    cordex_annmean_mean3 = np.mean(np.mean(cordex_annmean3.values[:,select_group],axis=1),axis=0)
    cordex_annmean_mean4 = np.mean(np.mean(cordex_annmean4.values[:,select_group],axis=1),axis=0)
    #print(cordex_annmean1.values[:,select_group])

    model_annmean_mean1 = np.mean(np.mean(model_annmean1.values[:,select_group],axis=1),axis=0)
    model_annmean_mean2 = np.mean(np.mean(model_annmean2.values[:,select_group],axis=1),axis=0)
    model_annmean_mean3 = np.mean(np.mean(model_annmean3.values[:,select_group],axis=1),axis=0)
    model_annmean_mean4 = np.mean(np.mean(model_annmean4.values[:,select_group],axis=1),axis=0)

    dataset_annmean_mean1 = np.mean(np.mean(dataset_annmean1.values[:,select_group],axis=1),axis=0)

    cordex_annmean_std1 = np.std(np.mean(cordex_annmean1.values[:,select_group],axis=1),axis=0)
    cordex_annmean_std2 = np.std(np.mean(cordex_annmean2.values[:,select_group],axis=1),axis=0)
    cordex_annmean_std3 = np.std(np.mean(cordex_annmean3.values[:,select_group],axis=1),axis=0)
    cordex_annmean_std4 = np.std(np.mean(cordex_annmean4.values[:,select_group],axis=1),axis=0)

    model_annmean_std1 = np.std(np.mean(model_annmean1.values[:,select_group],axis=1),axis=0)
    model_annmean_std2 = np.std(np.mean(model_annmean2.values[:,select_group],axis=1),axis=0)
    model_annmean_std3 = np.std(np.mean(model_annmean3.values[:,select_group],axis=1),axis=0)
    model_annmean_std4 = np.std(np.mean(model_annmean4.values[:,select_group],axis=1),axis=0)

    dataset_annmean_std1 = np.std(np.mean(dataset_annmean1.values[:,select_group],axis=1),axis=0)

    plot_data = [model_annmean_mean1,model_annmean_mean2,model_annmean_mean3,model_annmean_mean4,cordex_annmean_mean1,cordex_annmean_mean2,cordex_annmean_mean3,cordex_annmean_mean4,dataset_annmean_mean1]
    plot_err  = [model_annmean_std1,model_annmean_std2,model_annmean_std3,model_annmean_std4,cordex_annmean_std1,cordex_annmean_std2,cordex_annmean_std3,cordex_annmean_std4,dataset_annmean_std1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(index, plot_data, bar_width,alpha=opacity,color=colors)
    plt.errorbar(index,plot_data,yerr=plot_err,elinewidth=0.5,ecolor='black',fmt='none',alpha=opacity)

    plt.xticks(index, legends,fontsize=5,rotation=60)
    plt.yticks(fontsize=6)
    plt.xlabel('Models and SACA', fontsize=8)
    plt.ylabel('Precip (mm/day)', fontsize=8)

    title = str(iniyear)+' to '+str(endyear)+' annual mean precip in the group: '+str(idx)
    plt.suptitle(title,fontsize=9,y=0.95)
    fname = 'vrcesm_prect_annual_mean_bar_vs_cordex_refSAOBS_group_'+str(idx)+'.pdf'

    plt.savefig(outdir+fname,bbox_inches='tight')
    plt.close(fig)


    #annual max
    cordex_annmax_mean1 = np.mean(np.mean(cordex_annmax1.values[:,select_group],axis=1),axis=0)
    cordex_annmax_mean2 = np.mean(np.mean(cordex_annmax2.values[:,select_group],axis=1),axis=0)
    cordex_annmax_mean3 = np.mean(np.mean(cordex_annmax3.values[:,select_group],axis=1),axis=0)
    cordex_annmax_mean4 = np.mean(np.mean(cordex_annmax4.values[:,select_group],axis=1),axis=0)

    model_annmax_mean1 = np.mean(np.mean(model_annmax1.values[:,select_group],axis=1),axis=0)
    model_annmax_mean2 = np.mean(np.mean(model_annmax2.values[:,select_group],axis=1),axis=0)
    model_annmax_mean3 = np.mean(np.mean(model_annmax3.values[:,select_group],axis=1),axis=0)
    model_annmax_mean4 = np.mean(np.mean(model_annmax4.values[:,select_group],axis=1),axis=0)

    dataset_annmax_mean1 = np.mean(np.mean(dataset_annmax1.values[:,select_group],axis=1),axis=0)

    cordex_annmax_std1 = np.std(np.mean(cordex_annmax1.values[:,select_group],axis=1),axis=0)
    cordex_annmax_std2 = np.std(np.mean(cordex_annmax2.values[:,select_group],axis=1),axis=0)
    cordex_annmax_std3 = np.std(np.mean(cordex_annmax3.values[:,select_group],axis=1),axis=0)
    cordex_annmax_std4 = np.std(np.mean(cordex_annmax4.values[:,select_group],axis=1),axis=0)

    model_annmax_std1 = np.std(np.mean(model_annmax1.values[:,select_group],axis=1),axis=0)
    model_annmax_std2 = np.std(np.mean(model_annmax2.values[:,select_group],axis=1),axis=0)
    model_annmax_std3 = np.std(np.mean(model_annmax3.values[:,select_group],axis=1),axis=0)
    model_annmax_std4 = np.std(np.mean(model_annmax4.values[:,select_group],axis=1),axis=0)

    dataset_annmax_std1 = np.std(np.mean(dataset_annmax1.values[:,select_group],axis=1),axis=0)

    plot_data = [model_annmax_mean1,model_annmax_mean2,model_annmax_mean3,model_annmax_mean4,cordex_annmax_mean1,cordex_annmax_mean2,cordex_annmax_mean3,cordex_annmax_mean4,dataset_annmax_mean1]
    plot_err = [model_annmax_std1,model_annmax_std2,model_annmax_std3,model_annmax_std4,cordex_annmax_std1,cordex_annmax_std2,cordex_annmax_std3,cordex_annmax_std4,dataset_annmax_std1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(index, plot_data, bar_width,alpha=opacity,color=colors)
    plt.errorbar(index,plot_data,yerr=plot_err,elinewidth=0.5,ecolor='black',fmt='none',alpha=opacity)

    plt.xticks(index, legends,fontsize=5,rotation=60)
    plt.yticks(fontsize=6)
    plt.xlabel('Models and SACA', fontsize=8)
    plt.ylabel('Precip (mm/day)', fontsize=8)

    title = str(iniyear)+' to '+str(endyear)+' annual maximum precip in the group: '+str(idx)
    plt.suptitle(title,fontsize=9,y=0.95)
    fname = 'vrcesm_prect_annual_max_bar_vs_cordex_refSAOBS_group_'+str(idx)+'.pdf'

    plt.savefig(outdir+fname,bbox_inches='tight')
    plt.close(fig)
'''


############################################################################
# fit the extremes
############################################################################
# ncluster = 5
# kmeans_labels = np.array([0,0,3,1,0,0,4,0,0,3,3,0,2,0,2,3,0,0,4,3,3,0])  #2000-2005
# ncluster = 4
# kmeans_labels = np.array([3, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 3, 0, 0, 2, 2, 0])
# ncluster = 3
# kmeans_labels = np.array([0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 0, 0, 0, 2, 2, 0])
# kmeans_labels = kmeans.labels_

res_save = {}

res_labels = pickle.load(open(kmeans_resdir+'SA-OBS_'+str(iniyear)+'to' +
                              str(endyear)+'_kmeans_clustering_results.p', "rb"))
kmeans_labels = res_labels[str(ncluster)].values

for idx in range(ncluster):
    select_group = (kmeans_labels == idx)
    # print(select_group)

    # pull data out of dataframe
    dataset_ts1 = np.array(dataset1.values[:, select_group])
    # print(dataset_ts1.shape)
    dataset_ta1 = dataset_ts1.flatten()
    print(dataset_ts1)

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

    # test the mean residual life plot for SA-OBS
    dataset_max1 = np.amax(dataset_ts1)
    test_thresholds = np.arange(0, 1.*dataset_max1/2, 1.)

    # create array for mean excess and 95% confidence bounds
    test_me = np.zeros(len(test_thresholds))
    test_meup = np.zeros(len(test_thresholds))
    test_mebot = np.zeros(len(test_thresholds))
    test_menum = np.zeros(len(test_thresholds))
    test_c = np.zeros(len(test_thresholds))
    test_sigma = np.zeros(len(test_thresholds))

    for ii, ithres in enumerate(test_thresholds):
        # print('current threshold: '+str(ithres))
        dataset_ts1_sub = dataset_ts1[dataset_ts1 > ithres] - ithres
        test_len = len(dataset_ts1_sub)
        test_me[ii] = np.mean(dataset_ts1_sub)
        test_std = np.std(dataset_ts1_sub)
        test_menum[ii] = len(dataset_ts1_sub)
        test_meup[ii] = test_me[ii] + 1.96*(test_std/np.sqrt(test_len))
        test_mebot[ii] = test_me[ii] - 1.96*(test_std/np.sqrt(test_len))

        data1_fit = gpd.fit(dataset_ts1_sub)
        data1_fit = gpd.fit(dataset_ts1_sub)

        # print(data1_fit)
        data1_fit = gpdfit_moment(dataset_ts1_sub)
        # print(data1_fit)
        tempscore = ss.kstest(dataset_ts1_sub, 'genpareto', args=[
                              data1_fit[0], 0., data1_fit[1]], alternative='two-sided')
        # print(tempscore)
        test_c[ii] = data1_fit[0]
        test_sigma[ii] = data1_fit[1]

    print(test_thresholds[(np.abs(test_menum - 500)).argmin()])
    print(ss.percentileofscore(dataset_ts1, test_thresholds[(np.abs(test_menum - 500)).argmin()]))
    dataset_percent1 = np.percentile(dataset_ts1, percentile)

    # mean residual life
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(test_thresholds, test_me, linestyle='solid', linewidth=0.7, c='k')
    plt.plot(test_thresholds, test_meup, linestyle='dashed', linewidth=0.7, c='k')
    plt.plot(test_thresholds, test_mebot, linestyle='dashed', linewidth=0.7, c='k')

    plt.ylabel("Mean Excess")
    plt.xlabel("Thresholds")
    ax.set_title(str(percentile)+"th percentile: "+str(dataset_percent1)+",  500 obs needs threshold below: " +
                 str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=8)

    title = 'Mean residual life plot for SACA in group: '+str(idx)
    plt.suptitle(title, fontsize=10, y=0.95)

    fname = 'saobs_prect_'+str(percentile)+'th_mean_residual_life_plot_group_'+str(idx)+'.pdf'
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

    # c
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(test_thresholds, test_c, linestyle='solid', linewidth=1.5, c='k')

    plt.ylabel('Xi')
    plt.xlabel("Thresholds")
    ax.set_title(str(percentile)+"th percentile: "+str(dataset_percent1)+",  500 obs needs threshold below: " +
                 str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=8)

    title = 'Estimated parameter (Xi) for SACA in group: '+str(idx)
    plt.suptitle(title, fontsize=10, y=0.95)

    fname = 'saobs_prect_'+str(percentile)+'th_xi_plot_group_'+str(idx)+'.pdf'
    plt.savefig(outdir+fname, bbox_inches='tight')
    plt.close(fig)

    # sigma
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(test_thresholds, test_sigma, linestyle='solid', linewidth=1.5, c='k')

    plt.ylabel('Xi')
    plt.xlabel("Thresholds")
    ax.set_title(str(percentile)+"th percentile: "+str(dataset_percent1)+",  500 obs needs threshold below: " +
                 str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=8)

    title = 'Estimated parameter (sigma) for SACA in group: '+str(idx)
    plt.suptitle(title, fontsize=10, y=0.95)

    fname = 'saobs_prect_'+str(percentile)+'th_sigma_plot_group_'+str(idx)+'.pdf'
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

    dataset_percent1 = np.percentile(dataset_ts1, percentile)

    print([model_percent1, model_percent2, model_percent3, model_percent4, cordex_percent1,
           cordex_percent2, cordex_percent3, cordex_percent4, dataset_percent1])

    model_ts1_sub = model_ts1[model_ts1 > model_percent1] - model_percent1
    model_ts2_sub = model_ts2[model_ts2 > model_percent2] - model_percent2
    model_ts3_sub = model_ts3[model_ts3 > model_percent3] - model_percent3
    model_ts4_sub = model_ts4[model_ts4 > model_percent4] - model_percent4

    cordex_ts1_sub = cordex_ts1[cordex_ts1 > cordex_percent1] - cordex_percent1
    cordex_ts2_sub = cordex_ts2[cordex_ts2 > cordex_percent2] - cordex_percent2
    cordex_ts3_sub = cordex_ts3[cordex_ts3 > cordex_percent3] - cordex_percent3
    cordex_ts4_sub = cordex_ts4[cordex_ts4 > cordex_percent4] - cordex_percent4

    dataset_ts1_sub = dataset_ts1[dataset_ts1 > dataset_percent1] - dataset_percent1

    # fit data
    model_fit1 = gpdfit_moment(model_ts1_sub)
    model_fit2 = gpdfit_moment(model_ts2_sub)
    model_fit3 = gpdfit_moment(model_ts3_sub)
    model_fit4 = gpdfit_moment(model_ts4_sub)

    cordex_fit1 = gpdfit_moment(cordex_ts1_sub)
    cordex_fit2 = gpdfit_moment(cordex_ts2_sub)
    cordex_fit3 = gpdfit_moment(cordex_ts3_sub)
    cordex_fit4 = gpdfit_moment(cordex_ts4_sub)

    dataset_fit1 = gpdfit_moment(dataset_ts1_sub)

    print(model_fit1)
    print(model_fit2)
    print(model_fit3)
    print(model_fit4)

    print(cordex_fit1)
    print(cordex_fit2)
    print(cordex_fit3)
    print(cordex_fit4)

    print(dataset_fit1)

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

    dataset_ks1 = ss.kstest(dataset_ts1_sub, 'genpareto', args=[
                            dataset_fit1[0], 0., dataset_fit1[1]], alternative='two-sided')

    plot_data = [model_ts1_sub, model_ts2_sub, model_ts3_sub, model_ts4_sub,
                 cordex_ts1_sub, cordex_ts2_sub, cordex_ts3_sub, cordex_ts4_sub, dataset_ts1_sub]
    plot_fit = [model_fit1, model_fit2, model_fit3, model_fit4,
                cordex_fit1, cordex_fit2, cordex_fit3, cordex_fit4, dataset_fit1]
    plot_ks = [model_ks1, model_ks2, model_ks3, model_ks4, cordex_ks1, cordex_ks2, cordex_ks3, cordex_ks4, dataset_ks1]

    plot_fitcheck(plot_data, plot_fit, plot_ks, legends, idx)

    # calculate N-year return level
    year_returns = np.arange(1., 101., 1.)

    model_return1 = np.zeros(len(year_returns))
    model_return2 = np.zeros(len(year_returns))
    model_return3 = np.zeros(len(year_returns))
    model_return4 = np.zeros(len(year_returns))

    cordex_return1 = np.zeros(len(year_returns))
    cordex_return2 = np.zeros(len(year_returns))
    cordex_return3 = np.zeros(len(year_returns))
    cordex_return4 = np.zeros(len(year_returns))

    dataset_return1 = np.zeros(len(year_returns))

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

        dataset_return1[ii] = dataset_percent1 + dataset_fit1[1] / \
            dataset_fit1[0] * ((m*obsrate) ** dataset_fit1[0] - 1)

    select_years = [10, 20, 30,  50, 75, 100,  125, 150]
    model_select1 = np.zeros(len(select_years))
    model_select2 = np.zeros(len(select_years))
    model_select3 = np.zeros(len(select_years))
    model_select4 = np.zeros(len(select_years))

    dataset_select1 = np.zeros(len(select_years))

    for ii, iyear in enumerate(select_years):
        m = iyear * 365.
        obsrate = (100.-percentile)/100.

        model_select1[ii] = model_percent1 + model_fit1[1] / model_fit1[0] * ((m*obsrate) ** model_fit1[0] - 1)
        model_select2[ii] = model_percent2 + model_fit2[1] / model_fit2[0] * ((m*obsrate) ** model_fit2[0] - 1)
        model_select3[ii] = model_percent3 + model_fit3[1] / model_fit3[0] * ((m*obsrate) ** model_fit3[0] - 1)
        model_select4[ii] = model_percent4 + model_fit4[1] / model_fit4[0] * ((m*obsrate) ** model_fit4[0] - 1)

        dataset_select1[ii] = dataset_percent1 + dataset_fit1[1] / \
            dataset_fit1[0] * ((m*obsrate) ** dataset_fit1[0] - 1)

    print(select_years)
    print(model_select1)
    print(model_select2)
    print(model_select3)
    print(model_select4)
    print(dataset_select1)

    res_save['years'] = select_years
    res_save['cluster'+str(idx)+':vrseasia'] = model_select1
    res_save['cluster'+str(idx)+':ne30'] = model_select2
    res_save['cluster'+str(idx)+':fv0.9x1.25'] = model_select3
    res_save['cluster'+str(idx)+':fv1.25x2.5'] = model_select4
    res_save['cluster'+str(idx)+':SA-OBS'] = dataset_select1

    xticks = [2, 3, 10, 20, 50, 75, 100]
    # plot for both cesm and cordex
    plot_data = [model_return1, model_return2, model_return3, model_return4,
                 cordex_return1, cordex_return2, cordex_return3, cordex_return4, dataset_return1]

    xlabel = 'Return years'
    ylabel = 'Precip(mm/day)'
    title = 'Group '+str(idx)+' Precip extreme return levels'
    fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels_vs_cordex_group_'+str(idx)+'.pdf'

    plot_lines(year_returns, plot_data, colors, line_types, legends, xlabel,
               ylabel, title, outdir+fname, xlog=True, xticks=xticks)

    xlabel = 'SACA Return levels'
    ylabel = 'Model Return levels'
    title = 'Group '+str(idx)+' Precip extreme return levels vs SACA'
    fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels_vs_cordex_refSAOBS_group_'+str(idx)+'.pdf'
    plot_lines(dataset_return1, plot_data, colors, line_types, legends, xlabel, ylabel, title, outdir+fname)

    # plot for cesm only
    plot_data = [model_return1, model_return2, model_return3, model_return4, dataset_return1]

    xlabel = 'Return years'
    ylabel = 'Precip(mm/day)'
    title = 'Group '+str(idx)+' Precip extreme return levels'
    fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels_group_'+str(idx)+'.pdf'

    plot_lines(year_returns, plot_data, cesm_colors, cesm_line_types,
               cesm_legends, xlabel, ylabel, title, outdir+fname, xlog=True, xticks=xticks)

    xlabel = 'SACA Return levels'
    ylabel = 'Model Return levels'
    title = 'Group '+str(idx)+' Precip extreme return levels vs SACA'
    fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels_refSAOBS_group_'+str(idx)+'.pdf'
    plot_lines(dataset_return1, plot_data, cesm_colors, cesm_line_types,
               cesm_legends, xlabel, ylabel, title, outdir+fname)

res_save = pd.DataFrame(res_save)
res_save = res_save.set_index('years')
# save pickle data
pickle.dump(res_save, open(outdir+'SA-OBS_'+str(iniyear)+'to'+str(endyear)+'_kmeans_clustering_retrun_levels.p', "wb"))
res_save.to_csv(outdir+'SA-OBS_'+str(iniyear)+'to'+str(endyear) +
                '_kmeans_clustering_retrun_levels.csv', sep=',', index=True)
