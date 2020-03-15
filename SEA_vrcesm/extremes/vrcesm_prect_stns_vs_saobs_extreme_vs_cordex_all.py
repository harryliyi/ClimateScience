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
import matplotlib.pyplot as plt
plt.switch_backend('agg')

############################################################################
# setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/SA-OBS/extremes/'

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

# define regions
latbounds = [0, 25]
lonbounds = [90, 120]

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

outdir = outdir+str(iniyear)+'-'+str(endyear)+'/all stations/'

############################################################################
# functions
############################################################################


def plot_fitcheck(plot_data, plot_fit, plot_ks, legends):

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
        ax1.set_title('pdf check, ks p-value ='+str(tmep_ks.pvalue), fontsize=7, pad=2)
        ax1.legend(loc='upper right', fontsize=8)
        # set x/y tick label size
        ax1.xaxis.set_tick_params(labelsize=5)
        ax1.yaxis.set_tick_params(labelsize=5)
        ax1.set_xlabel('Frequency', fontsize=5)
        ax1.set_ylabel('Precip (mm/day)', fontsize=5)

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
        anderson = ss.anderson_ksamp([temp_data, gpd.rvs(temp_fit[0], 0., temp_fit[1], size=5000)])
        ax3.set_title('cdf check, AD sig level='+str(anderson.significance_level), fontsize=7, pad=2)
        ax3.legend(loc='lower right', fontsize=8)
        # set x/y tick label size
        ax3.xaxis.set_tick_params(labelsize=5)
        ax3.yaxis.set_tick_params(labelsize=5)
        ax3.set_xlabel('Probability', fontsize=5)
        ax3.set_ylabel('Precip (mm/day)', fontsize=5)

        title = str(iniyear)+' to '+str(endyear)+' '+legends[idx]+' '+str(percentile)+'th percentile precip GPD fit'
        plt.suptitle(title, fontsize=10, y=0.95)

        fig.subplots_adjust(hspace=0.4)
        fname = str(percentile)+'th_extremes_GPD_fit_'+str(idx+1)+'.pdf'
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

# convert from kg/(m^2*s) to mm/day
cordex_var1 = cordex_var1 * 86400 * 1000 / 997
cordex_var2 = cordex_var2 * 86400 * 1000 / 997
cordex_var3 = cordex_var3 * 86400 * 1000 / 997
cordex_var4 = cordex_var4 * 86400 * 1000 / 997

print('Reading VRCESM data...')

varname = 'PRECT'

resolution = 'fv02'
varfname = 'PREC'
case = 'vrseasia_AMIP_1979_to_2005'
model_var1, model_time1, model_lats1, model_lons1 = readvrcesm(varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

resolution = 'fv09'
varfname = 'PREC'
case = 'ne30_ne30_AMIP_1979_to_2005'
model_var2, model_time2, model_lats2, model_lons2 = readvrcesm(varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

resolution = 'fv09'
varfname = 'PREC'
case = 'f09_f09_AMIP_1979_to_2005'
model_var3, model_time3, model_lats3, model_lons3 = readvrcesm(varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

resolution = 'fv19'
varfname = 'PREC'
case = 'f19_f19_AMIP_1979_to_2005'
model_var4, model_time4, model_lats4, model_lons4 = readvrcesm(varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds, oceanmask=0, ignore_years=[1999])

model_var1 = 86400 * 1000 * model_var1
model_var2 = 86400 * 1000 * model_var2
model_var3 = 86400 * 1000 * model_var3
model_var4 = 86400 * 1000 * model_var4

# read Observations

print('Reading SA-OBS data...')

# read SA-OBS
version = 'countries'
countries = ['Thailand', 'Vietnam', 'Myanmar', 'Cambodia']
dataset1, obs_var1, stnids, stnnames, countrynames, stnlats, stnlons, stnhgts, stnmiss = read_SAOBS_pre(version, iniyear, endyear, countries, missing_ratio=10, ignore_years=[1999])

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

legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'CORDEX-ICHEC-EC-EARTH', 'CORDEX-IPSL-IPSL-CM5A-LR', 'CORDEX-MPI-M-MPI-ESM-MR', 'CORDEX-MOHC-HadGEM2-ES', 'SA-OBS']

colors = ['red', 'yellow', 'green', 'blue', 'tomato', 'goldenrod', 'darkcyan', 'darkmagenta', 'black']
line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-.', '-.', '-.', '-.', '-']

cesm_legends = ['CESM-vrseasia', 'CESM-ne30', 'CESM-fv0.9x1.25', 'CESM-fv1.9x2.5', 'SA-OBS']

cesm_colors = ['red', 'yellow', 'green', 'blue', 'black']
cesm_line_types = ['dashed', 'dashed', 'dashed', 'dashed', '-']


dataset_ts1 = np.array(dataset1.values)
dataset_ta1 = dataset_ts1.flatten()
print(dataset_ts1)

model_ts1 = model_var1.values.flatten()
model_ts2 = model_var2.values.flatten()
model_ts3 = model_var3.values.flatten()
model_ts4 = model_var4.values.flatten()

cordex_ts1 = cordex_var1.values.flatten()
cordex_ts2 = cordex_var2.values.flatten()
cordex_ts3 = cordex_var3.values.flatten()
cordex_ts4 = cordex_var4.values.flatten()

dataset_ts1 = dataset_ts1[~np.isnan(dataset_ts1)]

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
    print('current threshold: '+str(ithres))
    dataset_ts1_sub = dataset_ts1[dataset_ts1 > ithres] - ithres
    test_len = len(dataset_ts1_sub)
    test_me[ii] = np.mean(dataset_ts1_sub)
    test_std = np.std(dataset_ts1_sub)
    test_menum[ii] = len(dataset_ts1_sub)
    test_meup[ii] = test_me[ii] + 1.96*(test_std/np.sqrt(test_len))
    test_mebot[ii] = test_me[ii] - 1.96*(test_std/np.sqrt(test_len))

    data1_fit = gpd.fit(dataset_ts1_sub)
    data1_fit = gpd.fit(dataset_ts1_sub)

    print(data1_fit)
    data1_fit = gpdfit_moment(dataset_ts1_sub)
    print(data1_fit)
    tempscore = ss.kstest(dataset_ts1_sub, 'genpareto', args=[data1_fit[0], 0., data1_fit[1]], alternative='two-sided')
    print(tempscore)
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
ax.set_title(str(percentile)+"th percentile: "+str(dataset_percent1)+",  500 obs needs threshold below: "+str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=8)

title = 'Mean residual life plot for SA-OBS'
plt.suptitle(title, fontsize=10, y=0.95)

fname = 'saobs_prect_'+str(percentile)+'th_mean_residual_life_plot.pdf'
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)


# c
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(test_thresholds, test_c, linestyle='solid', linewidth=1.5, c='k')

plt.ylabel('Shape factor')
plt.xlabel("Thresholds")
ax.set_title(str(percentile)+"th percentile: "+str(dataset_percent1)+",  500 obs needs threshold below: "+str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=8)

title = 'Estimated parameter (Xi) for SA-OBS'
plt.suptitle(title, fontsize=10, y=0.95)

fname = 'saobs_prect_'+str(percentile)+'th_xi_plot.pdf'
plt.savefig(outdir+fname, bbox_inches='tight')
plt.close(fig)


# sigma
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(test_thresholds, test_sigma, linestyle='solid', linewidth=1.5, c='k')

plt.ylabel('Scale')
plt.xlabel("Thresholds")
ax.set_title(str(percentile)+"th percentile: "+str(dataset_percent1)+",  500 obs needs threshold below: "+str(test_thresholds[(np.abs(test_menum - 2000)).argmin()]), fontsize=8)

title = 'Estimated parameter (sigma) for SA-OBS'
plt.suptitle(title, fontsize=10, y=0.95)

fname = 'saobs_prect_'+str(percentile)+'th_sigma_plot.pdf'
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

print([model_percent1, model_percent2, model_percent3, model_percent4, cordex_percent1, cordex_percent2, cordex_percent3, cordex_percent4, dataset_percent1])


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

cordex_ks1 = ss.kstest(cordex_ts1_sub, 'genpareto', args=[cordex_fit1[0], 0., cordex_fit1[1]], alternative='two-sided')
cordex_ks2 = ss.kstest(cordex_ts2_sub, 'genpareto', args=[cordex_fit2[0], 0., cordex_fit2[1]], alternative='two-sided')
cordex_ks3 = ss.kstest(cordex_ts3_sub, 'genpareto', args=[cordex_fit3[0], 0., cordex_fit3[1]], alternative='two-sided')
cordex_ks4 = ss.kstest(cordex_ts4_sub, 'genpareto', args=[cordex_fit4[0], 0., cordex_fit4[1]], alternative='two-sided')

dataset_ks1 = ss.kstest(dataset_ts1_sub, 'genpareto', args=[dataset_fit1[0], 0., dataset_fit1[1]], alternative='two-sided')

plot_data = [model_ts1_sub, model_ts2_sub, model_ts3_sub, model_ts4_sub, cordex_ts1_sub, cordex_ts2_sub, cordex_ts3_sub, cordex_ts4_sub, dataset_ts1_sub]
plot_fit = [model_fit1, model_fit2, model_fit3, model_fit4, cordex_fit1, cordex_fit2, cordex_fit3, cordex_fit4, dataset_fit1]
plot_ks = [model_ks1, model_ks2, model_ks3, model_ks4, cordex_ks1, cordex_ks2, cordex_ks3, cordex_ks4, dataset_ks1]

plot_fitcheck(plot_data, plot_fit, plot_ks, legends)


# calculate N-year return level
year_returns = np.arange(1., 201., 1.)

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

    dataset_return1[ii] = dataset_percent1 + dataset_fit1[1] / dataset_fit1[0] * ((m*obsrate) ** dataset_fit1[0] - 1)


# plot for both cesm and cordex
plot_data = [model_return1, model_return2, model_return3, model_return4, cordex_return1, cordex_return2, cordex_return3, cordex_return4, dataset_return1]

xlabel = 'Return years'
ylabel = 'Precip(mm/day)'
title = 'Precip extreme return levels'
fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels_vs_cordexsea.pdf'
plot_lines(year_returns, plot_data, colors, line_types, legends, xlabel, ylabel, title, outdir+fname)


xlabel = 'SA-OBS Return levels'
ylabel = 'Model Return levels'
title = 'Precip extreme return levels vs SA-OBS'
fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels_vs_cordexsea_refSAOBS.pdf'
plot_lines(dataset_return1, plot_data, colors, line_types, legends, xlabel, ylabel, title, outdir+fname)


# plot for cesm only
plot_data = [model_return1, model_return2, model_return3, model_return4, dataset_return1]
xlabel = 'Return years'
ylabel = 'Precip(mm/day)'
title = 'Precip extreme return levels'
fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels.pdf'
plot_lines(year_returns, plot_data, cesm_colors, cesm_line_types, cesm_legends, xlabel, ylabel, title, outdir+fname)

xlabel = 'SA-OBS Return levels'
ylabel = 'Model Return levels'
title = 'Precip extreme return levels vs SA-OBS'
fname = 'vrcesm_prect_'+str(percentile)+'th_extremes_return_levels_refSAOBS.pdf'
plot_lines(dataset_return1, plot_data, cesm_colors, cesm_line_types, cesm_legends, xlabel, ylabel, title, outdir+fname)
