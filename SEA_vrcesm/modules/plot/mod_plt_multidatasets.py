'''
# This is a module that used to calculate monthly, seasonal and annual mean of multi-datasets

# Written by Harry Li
'''

# modules that the function needs
import pathmagic  # noqa: F401
import numpy as np
import pandas as pd
from modules.plot.mod_plt_regrid import data_regrid
from modules.plot.mod_plt_contour import plot_2Dcontour
from modules.stats.mod_stats_clim import mon2clim, getstats_2D_ttest, getstats_2D_ftest

################################################################################################
# calculate and plot monthly, annual and seasonal mean
################################################################################################


def plt_multidatasets_2Dcontour_mean(project, region, iniyear, endyear, varname, varstr, var_unit, model_vars, model_lons, model_lats, lonbounds,
                                     latbounds, legends, mean_colormap, mean_clevs, var_colormap, var_clevs, outdir, fname, **kwargs):

    # plot for monthly and annual mean
    print('Plotting for monthly and annual mean contour...')

    # create the month and annual mean list
    plot_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plot_list.append('Annual')

    if 'model_masks' in kwargs:
        model_masks = kwargs['model_masks']

    # calculate monthly and annual mean, wrap muti-datasets into one list
    datasets_mean = []
    datasets_std = []
    for idx in range(len(model_vars)):
        curr_var = model_vars[idx]
        curr_mean, curr_std = mon2clim(curr_var[:, :, :], opt=2)

        # mask the calculated results if necessary
        if 'model_masks' in kwargs:
            curr_mean[np.broadcast_to(model_masks[idx], curr_mean.shape)] = np.nan
            curr_std[np.broadcast_to(model_masks[idx], curr_std.shape)] = np.nan

        datasets_mean.append(curr_mean)
        datasets_std.append(curr_std)

    # print(len(datasets_mean))

    # plot the monthly and annual mean
    for idx, imonname in enumerate(plot_list):
        print('plotting for '+imonname+' mean...')

        # warp data into a list and plot for means
        plot_data = []
        for idata in range(len(datasets_mean)):
            idataset = datasets_mean[idata]
            # print(idataset.shape)
            plot_data.append(idataset[idx, :, :])

        # print(len(plot_data))
        title = str(iniyear)+'-'+str(endyear)+' '+imonname+' averaged '+varname

        # set different file names for monthly and annual mean
        if idx != 12:
            outname = project+'_'+varstr+'_'+region+'_monthly_mean_'+fname+'_'+str(idx+1)
        else:
            outname = project+'_'+varstr+'_'+region+'_annual_mean_'+fname

        plot_2Dcontour(plot_data, model_lons, model_lats, mean_colormap, mean_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=0)

        # plot for variabilities
        plot_data = []
        for istd in datasets_std:
            plot_data.append(istd[idx, :, :])

        title = str(iniyear)+'-'+str(endyear)+' '+imonname+' averaged '+varname+' variability'

        if idx != 12:
            outname = project+'_'+varstr+'_'+region+'_monthly_mean_var_'+fname+'_'+str(idx+1)
        else:
            outname = project+'_'+varstr+'_'+region+'_annual_mean_var_'+fname

        plot_2Dcontour(plot_data, model_lons, model_lats, var_colormap, var_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=0)

    ############################################################################
    # plot for seasonal mean contours
    print('Plotting for seasonal mean contour...')
    seasons_list = ['DJF', 'MAM', 'JJA', 'SON']
    # seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8],  'JJAS': [6, 7, 8, 9], 'AMJ': [4, 5, 6], 'SON': [9, 10, 11]}

    for idx, iseason in enumerate(seasons_list):
        print('plotting for '+iseason+' mean...')

        # calculate seasonal mean, wrap muti-datasets into one list
        datasets_mean = []
        datasets_std = []
        for imodel in range(len(model_vars)):
            curr_var = model_vars[imodel]
            curr_mean, curr_std = mon2clim(curr_var[:, :, :], opt=3, season=iseason)

            # mask the calculated results if necessary
            if 'model_masks' in kwargs:
                curr_mean[np.broadcast_to(model_masks[imodel], curr_mean.shape)] = np.nan
                curr_std[np.broadcast_to(model_masks[imodel], curr_std.shape)] = np.nan

            datasets_mean.append(curr_mean)
            datasets_std.append(curr_std)

        # set title and file names for seasonal mean
        title = str(iniyear)+'-'+str(endyear)+' '+iseason+' averaged '+varname
        outname = project+'_'+varstr+'_'+region+'_seasonal_mean_'+fname+'_'+str(idx+1)

        plot_2Dcontour(datasets_mean, model_lons, model_lats, mean_colormap, mean_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=0)

        # plot for variability
        title = str(iniyear)+'-'+str(endyear)+' '+iseason+' averaged '+varname+' variability'
        outname = project+'_'+varstr+'_'+region+'_seasonal_mean_var_'+fname+'_'+str(idx+1)

        plot_2Dcontour(datasets_std, model_lons, model_lats, var_colormap, var_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=0)


################################################################################################
# calculate and plot monthly, annual and seasonal difference with the reference dataset
################################################################################################

def plt_multidatasets_2Dcontour_bias(project, region, iniyear, endyear, varname, varstr, var_unit, model_vars, model_lons, model_lats, reference, referencestr,
                                     ref_var, ref_lon, ref_lat, lonbounds, latbounds, legends, diff_colormap, diff_clevs, var_colormap, var_clevs, outdir, fname, **kwargs):

    # plot for monthly and annual mean difference
    print('Plotting for monthly and annual mean differences against '+reference+'...')

    # create the month and annual mean list
    plot_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plot_list.append('Annual')

    if 'model_masks' in kwargs:
        model_masks = kwargs['model_masks']

    # regrid and calculate monthly and annual mean
    ref_var_regrids = []
    ref_var_regrids_mean = []
    ref_var_regrids_std = []
    datasets_mean = []
    datasets_std = []
    for imodel, imodel_var in enumerate(model_vars):
        # regrid reference data
        # print(imodel_var.shape)
        lonsout, latsout = np.meshgrid(model_lons[imodel], model_lats[imodel])
        temp_regrid = data_regrid(ref_var, ref_lon, ref_lat, lonsout, latsout)
        ref_var_regrids.append(temp_regrid)

        # calculate seasonal and annual mean and std for model output
        temp_mean, temp_std = mon2clim(imodel_var[:, :, :], opt=2)
        datasets_mean.append(temp_mean)
        datasets_std.append(temp_std)

        # calculate seasonal and annual mean and std for regrided reference data
        temp_mean, temp_std = mon2clim(temp_regrid[:, :, :], opt=2)
        ref_var_regrids_mean.append(temp_mean)
        ref_var_regrids_std.append(temp_std)

    # calculate degree of freedom
    expdf = (endyear-iniyear)
    refdf = (endyear-iniyear)

    for idx, imonname in enumerate(plot_list):
        print('plotting for '+imonname+' differences...')

        # create list to plot data
        plot_data = []
        plot_test = []
        plot_var_data = []
        plot_var_test = []

        # calculate the mean difference and t-test results
        for imodel in range(len(model_vars)):
            model_mean = datasets_mean[imodel]
            model_std = datasets_std[imodel]
            ref_mean = ref_var_regrids_mean[imodel]
            ref_std = ref_var_regrids_std[imodel]

            # calculate monthly mean differences
            curr_diff, curr_ttest = getstats_2D_ttest(
                model_mean[idx, :, :], ref_mean[idx, :, :], model_std[idx, :, :], ref_std[idx, :, :], expdf, refdf)

            # calculate the variability difference
            curr_var_diff, curr_var_ftest = getstats_2D_ftest(model_std[idx, :, :], ref_std[idx, :, :])

            # mask the calculated results if necessary
            if 'model_masks' in kwargs:
                curr_diff[model_masks[imodel]] = np.nan
                curr_ttest[model_masks[imodel]] = np.nan
                curr_var_diff[model_masks[imodel]] = np.nan
                curr_var_ftest[model_masks[imodel]] = np.nan

            plot_data.append(curr_diff)
            plot_test.append(curr_ttest)
            plot_var_data.append(curr_var_diff)
            plot_var_test.append(curr_var_ftest)

        # print(len(plot_data))

        # plot for monthly difference
        title = str(iniyear)+'-'+str(endyear)+' '+imonname+' averaged '+varname+' differece (Ref as '+reference+')'

        if idx != 12:
            outname = project+'_'+varstr+'_'+region+'_monthly_mean_'+fname+'_ref'+referencestr+'_wtsig_'+str(idx+1)
        else:
            outname = project+'_'+varstr+'_'+region+'_annual_mean_'+fname+'_ref'+referencestr+'_wtsig'

        plot_2Dcontour(plot_data, model_lons, model_lats, diff_colormap, diff_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=1, sig_test=plot_test)

        # without significance levels
        if idx != 12:
            outname = project+'_'+varstr+'_'+region+'_monthly_mean_'+fname+'_ref'+referencestr+'_nosig_'+str(idx+1)
        else:
            outname = project+'_'+varstr+'_'+region+'_annual_mean_'+fname+'_ref'+referencestr+'_nosig'

        plot_2Dcontour(plot_data, model_lons, model_lats, diff_colormap, diff_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=0)

        # plot for variability
        title = str(iniyear)+'-'+str(endyear)+' '+imonname+' averaged '+varname+' variability differece (Ref as '+reference+')'

        if idx != 12:
            outname = project+'_'+varstr+'_'+region+'_monthly_mean_var_'+fname+'_ref'+referencestr+'_wtsig_'+str(idx+1)
        else:
            outname = project+'_'+varstr+'_'+region+'_annual_mean_var_'+fname+'_ref'+referencestr+'_wtsig'

        plot_2Dcontour(plot_var_data, model_lons, model_lats, var_colormap, var_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=1, sig_test=plot_var_test, sig_thres=2.2693)

        # without significance levels
        if idx != 12:
            outname = project+'_'+varstr+'_'+region+'_monthly_mean_var_'+fname+'_ref'+referencestr+'_nosig_'+str(idx+1)
        else:
            outname = project+'_'+varstr+'_'+region+'_annual_mean_var_'+fname+'_ref'+referencestr+'_nosig'

        plot_2Dcontour(plot_var_data, model_lons, model_lats, var_colormap, var_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=0)

    ############################################################################
    # plot for seasonal mean differences
    print('Plotting for seasonal mean differences against '+reference+'...')

    # calculate degree of freedom
    expdf = (endyear-iniyear) * 3
    refdf = (endyear-iniyear) * 3

    seasons_list = ['DJF', 'MAM', 'JJA', 'SON']
    # seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8],  'JJAS': [6, 7, 8, 9], 'AMJ': [4, 5, 6], 'SON': [9, 10, 11]}

    for idx, iseason in enumerate(seasons_list):
        print('plotting for '+iseason+' differences...')

        # create list to plot data
        plot_data = []
        plot_test = []
        plot_var_data = []
        plot_var_test = []

        # calculate the mean difference and t-test results
        for imodel in range(len(model_vars)):
            # calculate seasonal mean
            model_mean, model_std = mon2clim(model_vars[imodel], opt=3, season=iseason)
            ref_mean, ref_std = mon2clim(ref_var_regrids[imodel], opt=3, season=iseason)

            # calculate seasonal mean differencs
            curr_diff, curr_ttest = getstats_2D_ttest(model_mean, ref_mean, model_std, ref_std, expdf, refdf)
            # calculate the variability difference
            curr_var_diff, curr_var_ftest = getstats_2D_ftest(model_std, ref_std)

            # mask the calculated results if necessary
            if 'model_masks' in kwargs:
                curr_diff[model_masks[imodel]] = np.nan
                curr_ttest[model_masks[imodel]] = np.nan
                curr_var_diff[model_masks[imodel]] = np.nan
                curr_var_ftest[model_masks[imodel]] = np.nan

            plot_data.append(curr_diff)
            plot_test.append(curr_ttest)
            plot_var_data.append(curr_var_diff)
            plot_var_test.append(curr_var_ftest)

        # plot for seasonal difference
        title = str(iniyear)+'-'+str(endyear)+' '+iseason+' averaged '+varname+' differece (Ref as '+reference+')'
        outname = project+'_'+varstr+'_'+region+'_seasonal_mean_'+fname+'_ref'+referencestr+'_wtsig_'+str(idx+1)

        plot_2Dcontour(plot_data, model_lons, model_lats, diff_colormap, diff_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=1, sig_test=plot_test)

        # without significance levels
        outname = project+'_'+varstr+'_'+region+'_seasonal_mean_'+fname+'_ref'+referencestr+'_nosig_'+str(idx+1)

        plot_2Dcontour(plot_data, model_lons, model_lats, diff_colormap, diff_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=0)

        # plot for variability
        title = str(iniyear)+'-'+str(endyear)+' '+iseason+' averaged '+varname+' variability differece (Ref as '+reference+')'
        outname = project+'_'+varstr+'_'+region+'_seasonal_mean_var_'+fname+'_ref'+referencestr+'_wtsig_'+str(idx+1)

        plot_2Dcontour(plot_var_data, model_lons, model_lats, var_colormap, var_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=1, sig_test=plot_var_test, sig_thres=2.2693)

        # without significance levels
        outname = project+'_'+varstr+'_'+region+'_seasonal_mean_var_'+fname+'_ref'+referencestr+'_nosig_'+str(idx+1)

        plot_2Dcontour(plot_var_data, model_lons, model_lats, var_colormap, var_clevs, legends, lonbounds,
                       latbounds, varname, var_unit, title, outdir+outname, opt=0)
