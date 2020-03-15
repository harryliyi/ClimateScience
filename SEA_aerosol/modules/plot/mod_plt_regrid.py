'''
#This is a module that used to regrid the 3D data

#Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd
from mpl_toolkits import basemap


def data_regrid(var, lonin, latin, lonout, latout):
    if (lonout.shape == 1)and(latout.shape == 1):
        lonout, latout = np.meshgrid(lonout, latout)

    res = []
    for idx in range(var.shape[0]):
        tempinterp = basemap.interp(var[idx, :, :], lonin, latin, lonout, latout, order=1)
        res.append(tempinterp)

    res = np.ma.array(res)

    return res
