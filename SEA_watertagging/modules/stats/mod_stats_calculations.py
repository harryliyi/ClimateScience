'''
# This is a module that used to calculate climatological statistics for multidatasets and comparison between multidatasets and reference data

# Written by Harry Li
'''

# modules that the function needs
import pathmagic  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats as ss
from modules.plot.mod_plt_regrid import data_regrid

################################################################################################
# basic stats function
################################################################################################


def stats_rmsd(var, ref_var):
    if (var.shape != ref_var.shape):
        print('stats_rmse: Error! Two datasets have different size!')
        return -1
    else:
        return np.sqrt(np.nanmean((var-ref_var)**2))


def stats_msd(var, ref_var):
    if (var.shape != ref_var.shape):
        print('stats_rmse: Error! Two datasets have different size!')
        return -1
    else:
        var = var.flatten()
        ref_var = ref_var.flatten()
        var[np.isnan(ref_var)] = np.nan
        ref_var[np.isnan(var)] = np.nan
        n = np.count_nonzero(~np.isnan(ref_var))

        diff = np.nanmean(var-ref_var)
        std = np.sqrt((np.nanvar(var)+np.nanvar(ref_var))/2)
        t = diff/(std*np.sqrt(2/n))
        df = 2 * np.count_nonzero(~np.isnan(ref_var)) - 2  # degree of freedom
        pval = 2 * (1 - ss.t.cdf(np.abs(t), df=df))

        # print(t, pval)
        # print(ss.ttest_ind(var[~np.isnan(var)], ref_var[~np.isnan(ref_var)]))

        return diff, pval, t, df


def stats_mrd(var, ref_var):
    if (var.shape != ref_var.shape):
        print('stats_rmse: Error! Two datasets have different size!')
        return -1
    else:
        return np.nanmean(np.abs(var-ref_var))/np.nanmean(ref_var)


def stats_corr(var, ref_var):
    if (var.shape != ref_var.shape):
        print('stats_rmse: Error! Two datasets have different size!')
        return -1
    else:
        var = var.flatten()
        ref_var = ref_var.flatten()
        var[np.isnan(ref_var)] = np.nan
        ref_var[np.isnan(var)] = np.nan
        df = np.count_nonzero(~np.isnan(ref_var)) - 2  # degree of freedom
        r = ss.pearsonr(var[~np.isnan(var)], ref_var[~np.isnan(ref_var)])[0]
        pval = ss.pearsonr(var[~np.isnan(var)], ref_var[~np.isnan(ref_var)])[1]
        t = r*np.sqrt(df/(1-r**2))
        # print(2 * (1 - ss.t.cdf(np.abs(t), df=df)))
        # print(pval)
        # print(ss.pearsonr(var[~np.isnan(var)], ref_var[~np.isnan(ref_var)]))
        # print(np.corrcoef(var, ref_var))

        return r, pval, t, df

################################################################################################
# calculate the statistics for single variable in multi-datasets
################################################################################################


def cal_stats_clim_multidatasets(datasets, datasets_times, legends, name_list, idx_list, **kwargs):

    dim_selections = ['3D', 'temperal', 'spatial']

    index = []
    for iname in name_list:
        for idim in dim_selections:
            index.append(iname+'-'+idim)

    # print(index)

    result = {'time_list': index}
    result = pd.DataFrame(result)
    result = result.set_index('time_list')

    for idata, idata_var in enumerate(datasets):
        print('Calculating the stats for '+legends[idata]+'...')
        idata_time = datasets_times[idata]

        result_mean = []
        result_std = []
        for idx, itime in enumerate(name_list):
            temp_3d = idata_var[np.in1d(idata_time.month, idx_list[idx]), :, :]
            # print(idata_time[np.in1d(idata_time.month, idx_list[idx])])
            # print(temp_3d.shape)
            # print(temp_3d[0, :, :])

            # 3d mean and std
            result_mean.append(np.nanmean(temp_3d[:, :, :]))
            result_std.append(np.nanstd(temp_3d[:, :, :]))

            # temperal mean and std
            temp = np.nanmean(temp_3d.reshape(temp_3d.shape[0], -1), axis=1)
            result_mean.append(np.nanmean(temp))
            result_std.append(np.nanstd(temp))

            # spatial mean and std
            temp = np.nanmean(temp_3d, axis=0)
            result_mean.append(np.nanmean(temp))
            result_std.append(np.nanstd(temp))

        result[legends[idata]+'-mean'] = result_mean
        result[legends[idata]+'-std'] = result_std

    return result

################################################################################################
# calculate the statistics between multi-datasets and reference dataset
################################################################################################


def cal_stats_clim_multidatasets_reference_bias(datasets, datasets_times, datasets_lons, datasets_lats, legends,
                                                ref_data, ref_time, ref_lon, ref_lat, ref_legend, name_list, idx_list, **kwargs):
    print('Calculating the stats against '+ref_legend+'...')

    dim_selections = ['3D', 'temperal', 'spatial']

    index = []
    for iname in name_list:
        for idim in dim_selections:
            index.append(iname+'-'+idim)

    # print(index)

    result = {'time_list': index}
    result = pd.DataFrame(result)
    result = result.set_index('time_list')

    for idata, idata_var in enumerate(datasets):
        print('Calculating the stats for '+legends[idata]+'...')
        idata_time = datasets_times[idata]

        lonsout, latsout = np.meshgrid(datasets_lons[idata], datasets_lats[idata])
        ref_data_regrid = data_regrid(ref_data, ref_lon, ref_lat, lonsout, latsout)

        result_rmsd = []
        result_msd = []
        result_msd_pval = []
        result_msd_tscore = []
        result_msd_df = []
        result_mrd = []
        result_corr = []
        result_corr_pval = []
        result_corr_tscore = []
        result_corr_df = []
        for idx, itime in enumerate(name_list):
            temp_3d = idata_var[np.in1d(idata_time.month, idx_list[idx]), :, :]
            temp_ref_3d = ref_data_regrid[np.in1d(ref_time.month, idx_list[idx]), :, :]

            # 3d mean and std
            # RMSD
            result_rmsd.append(stats_rmsd(temp_3d, temp_ref_3d))
            # MSD
            diff, pval, ts, df = stats_msd(temp_3d, temp_ref_3d)
            result_msd.append(diff)
            result_msd_pval.append(pval)
            result_msd_tscore.append(ts)
            result_msd_df.append(df)
            # MRD
            result_mrd.append(stats_mrd(temp_3d, temp_ref_3d))
            # Pearson correlation coefficient
            r, pval, ts, df = stats_corr(temp_3d, temp_ref_3d)
            result_corr.append(r)
            result_corr_pval.append(pval)
            result_corr_tscore.append(ts)
            result_corr_df.append(df)

            # temperal mean and std
            temp = np.nanmean(temp_3d.reshape(temp_3d.shape[0], -1), axis=1)
            temp_ref = np.nanmean(temp_ref_3d.reshape(temp_3d.shape[0], -1), axis=1)
            # RMSD
            result_rmsd.append(stats_rmsd(temp, temp_ref))
            # MSD
            diff, pval, ts, df = stats_msd(temp, temp_ref)
            result_msd.append(diff)
            result_msd_pval.append(pval)
            result_msd_tscore.append(ts)
            result_msd_df.append(df)
            # MRD
            result_mrd.append(stats_mrd(temp, temp_ref))
            # Pearson correlation coefficient
            r, pval, ts, df = stats_corr(temp, temp_ref)
            result_corr.append(r)
            result_corr_pval.append(pval)
            result_corr_tscore.append(ts)
            result_corr_df.append(df)

            # spatial mean and std
            temp = np.nanmean(temp_3d, axis=0)
            temp_ref = np.nanmean(temp_ref_3d, axis=0)
            # RMSD
            result_rmsd.append(stats_rmsd(temp, temp_ref))
            # MSD
            diff, pval, ts, df = stats_msd(temp, temp_ref)
            result_msd.append(diff)
            result_msd_pval.append(pval)
            result_msd_tscore.append(ts)
            result_msd_df.append(df)
            # MRD
            result_mrd.append(stats_mrd(temp, temp_ref))
            # Pearson correlation coefficient
            r, pval, ts, df = stats_corr(temp, temp_ref)
            result_corr.append(r)
            result_corr_pval.append(pval)
            result_corr_tscore.append(ts)
            result_corr_df.append(df)

        result[legends[idata]+'-rmsd'] = result_rmsd
        result[legends[idata]+'-msd'] = result_msd
        result[legends[idata]+'-msd_pval'] = result_msd_pval
        result[legends[idata]+'-msd_tscore'] = result_msd_tscore
        result[legends[idata]+'-msd_df'] = result_msd_df
        result[legends[idata]+'-mrd'] = result_mrd
        result[legends[idata]+'-corr'] = result_corr
        result[legends[idata]+'-corr_pval'] = result_corr_pval
        result[legends[idata]+'-corr_tscore'] = result_corr_tscore
        result[legends[idata]+'-corr_df'] = result_corr_df

    return result
