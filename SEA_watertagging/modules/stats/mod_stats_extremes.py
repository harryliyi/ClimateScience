'''
#This is a module that used to calculate climatological mean

#Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
from scipy.optimize import curve_fit

# define a function to calculate root mean square error


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# define gpd function


def pdffunc(x, c, loc, scale):
    return 1. / scale * (1. + c * ((x-loc)/scale)) ** (-1. - 1./c)


def cdffunc(x, c, loc, scale):
    return 1. - (1. + c * ((x-loc)/scale)) ** (- 1./c)


def pdffunc_noloc(x, c, scale):
    return 1. / scale * (1. + c * (x/scale)) ** (-1. - 1./c)


def cdffunc_noloc(x, c, scale):
    return 1. - (1. + c * (x/scale)) ** (- 1./c)

# define fitting process and ks test


def kstest(rs, ibins):
    rsmin = np.amin(rs)
    rsmax = np.amax(rs)

    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2., rsmax, (rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin, rsmax + (rsmax-rsmin)/ibins/2., (rsmax-rsmin)/ibins)

    hist, bin_edges = np.histogram(rs, bins=xbins, density=True)
    hist_tofitcdf = np.cumsum(hist)*(rsmax-rsmin)/ibins

    popt, pcov = curve_fit(cdffunc_noloc, x, hist_tofitcdf)
    tempscore = ss.kstest(rs, 'genpareto', args=[popt[0], 0., popt[1]], alternative='two-sided')

    return popt, pcov, tempscore

# check fitting


def cdfcheck(rs, ibins, popt):
    rsmin = np.amin(rs)
    rsmax = np.amax(rs)

    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2., rsmax, (rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin, rsmax + (rsmax-rsmin)/ibins/2., (rsmax-rsmin)/ibins)

    ycdf = cdffunc_noloc(x, popt[0], popt[1])

    return x, xbins, ycdf


def pdfcheck(rs, ibins, popt):
    rsmin = np.amin(rs)
    rsmax = np.amax(rs)

    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2., rsmax, (rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin, rsmax + (rsmax-rsmin)/ibins/2., (rsmax-rsmin)/ibins)

    ypdf = pdffunc_noloc(x, popt[0], popt[1])

    return x, xbins, ypdf

# moment method


def gpdfit_moment(rs):
    rs_mean = np.mean(rs)
    rs_var = np.var(rs)

    xi = 0.5*(1-rs_mean*rs_mean/rs_var)
    sigma = rs_mean*(1-xi)

    return [xi, sigma]


'''
calculate the N-day consecutive maximum
Description on parameters:
1) var: the input daily dataset, it can be 1-4 dimensions but the first dimension needs to be time
2) time: the datetime seires for the first dimension
3) ndays: the length of rolling sum (for N-day consecutive sum)
4) freq: the frequency of out put it can be either monthly or annual
'''


def climdex_RxNday(var, time, ndays, freq='monthly'):
    if len(var.shape) > 1:
        var_shape = var.shape
        size = np.prod(var_shape[1:])
        # print(size)
        var = var.reshape((var_shape[0], size))
        # print(var.shape)

    df = pd.DataFrame.from_records(var, index=time)
    # print(df)
    if ndays != 1:
        df = df.rolling(window=ndays, center=True,).sum()

    if freq == 'monthly':
        df_max = df.groupby(pd.Grouper(freq='M')).max()
        df_idxmax = df.groupby(pd.Grouper(freq='M')).idxmax()
    if freq == 'annual':
        # df_max = df.resample('A').max()
        # df_idxmax = df.resample('A').idxmax()
        df_max = df.groupby(pd.Grouper(freq='A')).max()
        df_idxmax = df.groupby(pd.Grouper(freq='A')).idxmax()

    print(df_max)
    # print(df_idxmax)
    ntimes = df_max.shape[0]
    if len(var.shape) > 1:
        print(np.concatenate([np.array([ntimes]), var_shape[1:]]))
        nsizes = np.concatenate([np.array([ntimes]), var_shape[1:]])
        df_max = df_max.values.reshape((nsizes))
        df_idxmax = df_idxmax.values.reshape((nsizes))
    else:
        df_max = df_max.values
        df_idxmax = df_idxmax.values

    #print(df_max[:, 0, 0])
    #print(df_idxmax[:, 0, 0])
    return df_max, df_idxmax
