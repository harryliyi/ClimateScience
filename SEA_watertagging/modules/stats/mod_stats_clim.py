'''
#This is a module that used to calculate climatological mean

#Written by Harry Li
'''

# modules that the function needs
import numpy as np
import pandas as pd


################################################################################################
# calculate climatological mean from monthly mean data
################################################################################################
'''
This is a function used to calculate climatological mean and variability

Description on parameters:
1) var: the variable that need to be analyzed, it must be either 1-dimension (time) or 3-dimension (time,lat,lon)
        the first dimension must be integer times 12

2) opt: 0: the return will be seasonality with standard deviation, if input is 3d variable, horizontal mean will be calculated
        1: the return will be annual mean time series with standard deviation, if input is 3d variable, horizontal mean will be calculated
        2: the return will be monthly (0-11) + annual (12) mean contour with standard deviation, the 3d input is required
        3: the return will be seasonal mean contour with standard deviation, the 3d input and selected season is required
        4: the return will be seasonal mean with standard deviation, if input is 3d variable, horizontal mean will be calculated

3) *args: not used for now

4) **kwargs: the optional input contains:

    season: selected season for seasonal mena contour

5) returns: res_mean is climatological mean and res_std is corresponding standard deviation
'''


def mon2clim(var, opt=0, *args, **kwargs):
    # check dimension
    var = np.ma.array(var)
    if len(var.shape) != 1 and len(var.shape) != 3:
        print('mon2clim: Error! The input variable is expected to be 1D or 3D, current dimension is '+str(len(var.shape))+'!')
        return -1, -1

    # check if the monthly data covers all months
    if var.shape[0] % 12 != 0:
        print('mon2clim: Error! The input time does not cover all months!')
        return -1, -1

    opt_list = [0, 1, 2, 3, 4]
    if opt not in opt_list:
        print('mon2clim: Error! Invalid option!')
        return -1, -1

    # calculate seasonality
    if opt == 0:
        # calculate horizontal mean if needed
        if len(var.shape) == 3:
            print('mon2clim: Warning! Input variable is 3D, horizoncal mean is calculated')
            # print(var[0,:,:])
            var = np.nanmean(var.reshape(var.shape[0], -1), axis=1)
            # print(var)
            # print(var)

        res_mean = []
        res_std = []

        for imon in range(12):
            temp = var[imon::12]
            # print(temp)
            res_mean.append(np.ma.mean(temp[~np.isnan(temp)]))
            res_std.append(np.ma.std(temp[~np.isnan(temp)]))

        res_mean = np.array(res_mean)
        res_std = np.array(res_std)

        return res_mean, res_std

    # calculate annual mean time series
    if opt == 1:
        # calculate horizontal mean if needed
        if len(var.shape) == 3:
            print('mon2clim: Warning! Input variable is 3D, horizoncal mean is calculated')
            var = np.nanmean(var.reshape(var.shape[0], -1), axis=1)

        res_mean = []
        res_std = []

        nyear = len(var)/12
        for iyear in range(nyear):
            res_mean.append(np.mean(var[iyear*12:iyear*12+12]))
            res_std.append(np.std(var[iyear*12:iyear*12+12]))

        res_mean = np.array(res_mean)
        res_std = np.array(res_std)

        return res_mean, res_std

    # calculate monthly (0-11) + annual (12) mean contour
    if opt == 2:
        # check if the input variable is 3D
        if len(var.shape) != 3:
            print('mon2clim: Error! Input variable is not 3D, unable to calculate mean contour')
            return -1, -1

        res_mean = []
        res_std = []

        for imon in range(12):
            res_mean.append(np.ma.mean(var[imon::12, :, :], axis=0))
            res_std.append(np.ma.std(var[imon::12, :, :], axis=0))

        temp_list = []
        nyears = var.shape[0]//12
        for iyear in range(nyears):
            temp_list.append(np.ma.mean(var[iyear*12:iyear*12+12, :, :], axis=0))
        temp_list = np.ma.array(temp_list)
        # print(temp_list.shape)

        res_mean.append(np.ma.mean(temp_list, axis=0))
        res_std.append(np.ma.std(temp_list, axis=0))

        res_mean = np.ma.array(res_mean)
        res_std = np.ma.array(res_std)

        return res_mean, res_std

    # calculate seasonal mean contour
    if opt == 3:
        # check if the input variable is 3D
        if len(var.shape) != 3:
            print('mon2clim: Error! Input variable is not 3D, unable to calculate mean contour')
            return -1, -1

        # check if the season has been selected
        if 'season' not in kwargs:
            print('mon2clim: Error! Selected season is needed!')
            return -1, -1

        # allowed all seasons
        seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8],
                   'JJAS': [6, 7, 8, 9], 'AMJ': [4, 5, 6], 'SON': [9, 10, 11]}

        # check if input season is allowed
        iseason = kwargs['season']
        if iseason not in seasons:
            print('mon2clim: Error! The input season can not be interpreted!')
            return -1, -1

        var_list = np.array([])

        iseason_val = seasons[iseason]
        for idx, imon in enumerate(iseason_val):
            imon = imon - 1
            # print(imon)
            if idx == 0:
                var_list = var[imon::12, :, :]
            else:
                var_list = np.concatenate((var_list, var[imon::12, :, :]), axis=0)

        # print(var_list.shape)

        res_mean = np.mean(var_list, axis=0)
        res_std = np.std(var_list, axis=0)

        return res_mean, res_std

    # calculate seasonal mean contour
    if opt == 4:
        # calculate horizontal mean if needed
        if len(var.shape) == 3:
            print('mon2clim: Warning! Input variable is 3D, horizoncal mean is calculated')
            var = np.nanmean(var.reshape(var.shape[0], -1), axis=1)

        res_mean = []
        res_std = []

        # check if the season has been selected
        if 'season' not in kwargs:
            print('mon2clim: Error! Selected season is needed!')
            return -1, -1

        # allowed all seasons
        seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8],
                   'JJAS': [6, 7, 8, 9], 'AMJ': [4, 5, 6], 'SON': [9, 10, 11]}

        # check if input season is allowed
        iseason = kwargs['season']
        if iseason not in seasons:
            print('mon2clim: Error! The input season can not be interpreted!')
            return -1, -1

        iseason_val = seasons[iseason]
        for idx, imon in enumerate(iseason_val):
            imon = imon - 1
            res_mean.extend(var[imon::12])

        # print(len(res_mean))
        res_std = np.std(res_mean)
        res_mean = np.mean(res_mean)

        return res_mean, res_std


################################################################################################
# calculate significance for difference between 2D variables
################################################################################################
def getstats_2Dsig(var1mean, var2mean, var1std, var2std, n1, n2):
    var1mean = np.ma.array(var1mean)
    var2mean = np.ma.array(var2mean)
    var1std = np.ma.array(var1std)
    var2std = np.ma.array(var2std)
    vardiff = var1mean - var2mean
    varttest = vardiff/np.sqrt(var1std**2/n1+var2std**2/n2)

    varttest = abs(varttest)
    vardiff = np.ma.array(vardiff)
    varttest = np.ma.array(varttest)

    return vardiff, varttest


def getstats_2D_ttest(var1mean, var2mean, var1std, var2std, n1, n2):
    var1mean = np.ma.array(var1mean)
    var2mean = np.ma.array(var2mean)
    var1std = np.ma.array(var1std)
    var2std = np.ma.array(var2std)
    vardiff = var1mean - var2mean
    varttest = vardiff/np.sqrt(var1std**2/n1+var2std**2/n2)

    varttest = abs(varttest)
    vardiff = np.ma.array(vardiff)
    varttest = np.ma.array(varttest)

    return vardiff, varttest


def getstats_2D_ftest(var1std, var2std):
    var1std = np.ma.array(var1std)
    var2std = np.ma.array(var2std)

    vardiff = var1std - var2std
    varftest = (var1std**2)/(var2std**2)

    varftest = varftest.flatten()
    temp = varftest[varftest < 1.]
    varftest[varftest < 1.] = 1./temp
    varftest = varftest.reshape(vardiff.shape)

    vardiff = np.ma.array(vardiff)
    varftest = np.ma.array(varftest)

    return vardiff, varftest
