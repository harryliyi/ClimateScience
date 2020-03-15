'''
#This is a module that used to select single grid data that is nearest to the stations from grided data sets

#Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd


def data_findstns(var, time, lats, lons, stnvar, stnlats, stnlons, stnnames):

    res = {}
    res['Date'] = time
    # print(len(time))
    for idx in range(len(stnlats)):
        ilat = np.argmin(np.abs(lats - stnlats[idx]))
        ilon = np.argmin(np.abs(lons - stnlats[idx]))

        temp = var[:, ilat, ilon]
        # print(len(temp))
        if len(temp) == len(stnvar[idx, :]):
            temp[np.isnan(stnvar[idx, :])] = np.NaN

        res[stnnames[idx]] = temp

    res = pd.DataFrame.from_dict(res)
    res = res.set_index(['Date'])

    return res
