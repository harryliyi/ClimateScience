#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.stats as ss
from scipy.stats import genpareto as gpd
from scipy.stats import genextreme as gev
from scipy.optimize import curve_fit

nbins = 3000
c = 0.1
loc = 10.0
scale = 2.5


def pdffunc(x, c, loc, scale):
    return (1. + c * ((x-loc)/scale)) ** (-1. - 1./c)

def cdffunc(x, c, loc, scale):
    return 1. - (1. + c * ((x-loc)/scale)) ** (- 1./c)

def cdffunc_noloc(x, c, scale):
    return 1. - (1. + c * (x/scale)) ** (- 1./c)

print('GPD parameters: ')
print([c,loc,scale])
rs = gpd.rvs(c=c,loc=loc,scale=scale,size=5000)

rsmax = np.amax(rs)
rsmin = np.amin(rs)
rs = rs - loc

rs_fit = gpd.fit(rs)
rs_fit = gpd.fit(rs)
print('scipy.stats.gpd fit: ')
print(rs_fit)
#print(gpd.fit(rs))

rsmax = np.amax(rs)
rsmin = np.amin(rs)

x = np.arange(rsmin+(rsmax-rsmin)/nbins/2.,rsmax,(rsmax-rsmin)/nbins)
xbins = np.arange(rsmin,rsmax + (rsmax-rsmin)/nbins/2.,(rsmax-rsmin)/nbins)
#print(x)
#print(xbins)

hist, bin_edges = np.histogram(rs, bins=xbins, density=True)
#print(hist)
ydata = pdffunc(x,c,loc,scale)
#print(ydata)


x_tofit = x[hist!=0]
hist_tofit = hist[hist!=0]

#print(x_tofit)
#print(hist_tofit)
popt, pcov = curve_fit(pdffunc, x_tofit, hist_tofit)
#popt[1] = popt[1] +rsmin
print('curve_fit pdf based on hist:')
print(popt)
print(pcov)

hist_tofitcdf = np.cumsum(hist)*(rsmax-rsmin)/nbins
print(hist_tofitcdf)
popt, pcov = curve_fit(cdffunc, x, hist_tofitcdf)
print('curve_fit cdf based on hist:')
print(popt)
print(pcov)

print('Kolmogorov-Smirnov test:')
print(ss.kstest(rs,'genpareto',args=popt, alternative = 'two-sided'))
#print(ss.kstest(rs,'genpareto',args=[c,loc,scale], alternative = 'two-sided'))
rs_sort = np.sort(rs)
distance = [max(cdffunc(irs,popt[0],popt[1],popt[2])-1.*(ii)/len(rs_sort),(1.+ii)/len(rs_sort)-cdffunc(irs,popt[0],popt[1],popt[2]))  for ii,irs in enumerate(rs_sort)]
print(np.amax(distance))

print('k-sample Anderson-Darling test: ')
print(ss.anderson_ksamp([rs,gpd.rvs(popt[0],popt[1],popt[2],size=3000)]))

#rs_sort = np.sort(rs)
#rs_ecdf = [1. * ii/len(rs_sort) for ii,irs in enumerate(rs_sort)]
#rs_unique = np.unique(rs_sort)
#print(len(rs_sort))
#print(len(rs_unique))

#print(rs_sort)
#print(rs_ecdf)
#popt, pcov = curve_fit(cdffunc, rs_sort, rs_ecdf)
#print('curve_fit cdf based on ecdf:')
#print(popt)
#print(pcov)

ksscore = 0.
bestfit = []
count = 0
print('check nbins sensitivity: ')
for ibins in np.arange(1.*len(rs)/3,1.*len(rs)-2,1.): #(1.*len(rs)-1.*len(rs)/3)/1000):
    ibins = int(ibins)
    x = np.arange(rsmin+(rsmax-rsmin)/ibins/2.,rsmax,(rsmax-rsmin)/ibins)
    xbins = np.arange(rsmin,rsmax + (rsmax-rsmin)/ibins/2.,(rsmax-rsmin)/ibins)

    hist, bin_edges = np.histogram(rs, bins=xbins, density=True)
    hist_tofitcdf = np.cumsum(hist)*(rsmax-rsmin)/ibins

    #x = x - rsmin
    #xbins = xbins - rsmin
    popt, pcov = curve_fit(cdffunc_noloc, x, hist_tofitcdf)

    tempscore = ss.kstest(rs,'genpareto',args=[popt[0],0.,popt[1]], alternative = 'two-sided')
    if tempscore.pvalue>ksscore:
        ksscore=tempscore.pvalue
        bestfit=popt
        bestcov=pcov
    count = count + 1
    print('No.'+str(count)+'('+str(ibins)+' bins): '+str(tempscore.statistic)+',  '+str(tempscore.pvalue))
    print(popt)


print('best fit:')
print(bestfit)
print(bestcov)
print(ss.kstest(rs,'genpareto',args=[bestfit[0],0.,bestfit[1]], alternative = 'two-sided'))
