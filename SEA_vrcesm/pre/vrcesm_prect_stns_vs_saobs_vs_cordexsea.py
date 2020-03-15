#This script is used to compare precip extreme difference bewteen vrcesm and CORDEX-SEA
#Several steps are inplemented:
#S1-read precip data from vrcesm and CORDEX-SEA
#S2-plot basic analysis
#S3-calculate and plot extreme
#
#Written by Harry Li

#import libraries
import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import pandas as pd
import math as math
import datetime as datetime

#import modules
from mod_dataread_cordex_sea import readcordex
from mod_dataread_vrcesm     import readvrcesm
from mod_dataread_obs_pre    import readobs_pre_day,read_SAOBS_pre
from mod_plt_findstns        import data_findstns
from mod_plt_lines           import plot_lines
from mod_plt_bars            import plot_bars
from mod_stats_clustering    import kmeans_cluster
from mod_stats_clim          import mon2clim

############################################################################
#setup directory
############################################################################
outdir = '/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/pre/VRseasia_vs_CORDEX_SEA/SA-OBS/'

############################################################################
#set parameters
############################################################################
#variable info
varname  = 'Total Precip'
varstr   = 'prect'
var_unit = 'mm/day'

#time bounds
iniyear   = 1980
endyear   = 2005
yearts    = np.arange(iniyear,endyear+1)
#yearts    = np.delete(yearts,9,None)
print(yearts)

#define regions
latbounds = [ -15 , 25 ]
lonbounds = [ 90 , 145 ]

#mainland Southeast Asia
reg_lats = [ 10 , 25 ]
reg_lons = [ 100 , 110 ]

#set data frequency
frequency = 'day'

#create months for plot
months = np.arange(1,13,1)
monnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

############################################################################
#read data
############################################################################

print('Reading CORDEX-SEA data...')

#read cordex
project       = 'SEA-22'
varname       = 'pr'
cordex_models = ['ICHEC-EC-EARTH','IPSL-IPSL-CM5A-LR','MPI-M-MPI-ESM-MR','MOHC-HadGEM2-ES']

modelname = 'ICHEC-EC-EARTH'
cordex_var1,cordex_time1,cordex_lats1,cordex_lons1 = readcordex(varname,iniyear,endyear,project,modelname,frequency,latbounds,lonbounds,oceanmask=0)
modelname = 'IPSL-IPSL-CM5A-LR'
cordex_var2,cordex_time2,cordex_lats2,cordex_lons2 = readcordex(varname,iniyear,endyear,project,modelname,frequency,latbounds,lonbounds,oceanmask=0)
modelname = 'MPI-M-MPI-ESM-MR'
cordex_var3,cordex_time3,cordex_lats3,cordex_lons3 = readcordex(varname,iniyear,endyear,project,modelname,frequency,latbounds,lonbounds,oceanmask=0)
modelname = 'MOHC-HadGEM2-ES'
cordex_var4,cordex_time4,cordex_lats4,cordex_lons4 = readcordex(varname,iniyear,endyear,project,modelname,frequency,latbounds,lonbounds,oceanmask=0)

#print(cordex_var4.shape)
#print(cordex_time4)
#read vrcesm

print('Reading VRCESM data...')

varname   = 'PRECT'

resolution= 'fv02'
varfname  = 'prec'
case      = 'vrseasia_AMIP_1979_to_2005'
model_var1,model_time1,model_lats1, model_lons1 = readvrcesm(varname,iniyear,endyear,resolution,varfname,case,frequency,latbounds,lonbounds,oceanmask=0)

resolution= 'fv09'
varfname  = 'PREC'
case      = 'ne30_ne30_AMIP_1979_to_2005'
model_var2,model_time2,model_lats2, model_lons2 = readvrcesm(varname,iniyear,endyear,resolution,varfname,case,frequency,latbounds,lonbounds,oceanmask=0)

resolution= 'fv09'
varfname  = 'PREC'
case      = 'f09_f09_AMIP_1979_to_2005'
model_var3,model_time3,model_lats3, model_lons3 = readvrcesm(varname,iniyear,endyear,resolution,varfname,case,frequency,latbounds,lonbounds,oceanmask=0)

resolution= 'fv19'
varfname  = 'PREC'
case      = 'f19_f19_AMIP_1979_to_2005'
model_var4,model_time4,model_lats4, model_lons4 = readvrcesm(varname,iniyear,endyear,resolution,varfname,case,frequency,latbounds,lonbounds,oceanmask=0)

#print(model_var1.shape)

#read Observations

print('Reading SA-OBS data...')

#read SA-OBS
version   = 'countries'
countries = ['Thailand','Vietnam','Myanmar','Cambodia']
dataset1,obs_var1,stnids,stnnames,countrynames,stnlats,stnlons,stnhgts,stnmiss = read_SAOBS_pre(version,iniyear,endyear,countries,missing_ratio=10)

#find stations in gridded data
cordex_var1 = data_findstns(cordex_var1,cordex_time1,cordex_lats1,cordex_lons1,obs_var1,stnlats,stnlons,stnnames)
cordex_var2 = data_findstns(cordex_var2,cordex_time2,cordex_lats2,cordex_lons2,obs_var1,stnlats,stnlons,stnnames)
cordex_var3 = data_findstns(cordex_var3,cordex_time3,cordex_lats3,cordex_lons3,obs_var1,stnlats,stnlons,stnnames)
cordex_var4 = data_findstns(cordex_var4,cordex_time4,cordex_lats4,cordex_lons4,obs_var1,stnlats,stnlons,stnnames)

model_var1 = data_findstns(model_var1,model_time1,model_lats1,model_lons1,obs_var1,stnlats,stnlons,stnnames)
model_var2 = data_findstns(model_var2,model_time2,model_lats2,model_lons2,obs_var1,stnlats,stnlons,stnnames)
model_var3 = data_findstns(model_var3,model_time3,model_lats3,model_lons3,obs_var1,stnlats,stnlons,stnnames)
model_var4 = data_findstns(model_var4,model_time4,model_lats4,model_lons4,obs_var1,stnlats,stnlons,stnnames)
#print(cordex_var1.shape)
#print(model_var4)

############################################################################
#plot for annual max ts for each station
############################################################################
#plot annual maximum precip for each station
cordex_annmax1 = cordex_var1.resample('A').max()
cordex_annmax2 = cordex_var2.resample('A').max()
cordex_annmax3 = cordex_var3.resample('A').max()
cordex_annmax4 = cordex_var4.resample('A').max()

model_annmax1 = model_var1.resample('A').max()
model_annmax2 = model_var2.resample('A').max()
model_annmax3 = model_var3.resample('A').max()
model_annmax4 = model_var4.resample('A').max()

dataset_annmax1 = dataset1.resample('A').max()
print(dataset_annmax1)

#calculate all station average
cordex_annmax1['avg'] = cordex_annmax1.mean(axis=1)
cordex_annmax2['avg'] = cordex_annmax2.mean(axis=1)
cordex_annmax3['avg'] = cordex_annmax3.mean(axis=1)
cordex_annmax4['avg'] = cordex_annmax4.mean(axis=1)

model_annmax1['avg'] = model_annmax1.mean(axis=1)
model_annmax2['avg'] = model_annmax2.mean(axis=1)
model_annmax3['avg'] = model_annmax3.mean(axis=1)
model_annmax4['avg'] = model_annmax4.mean(axis=1)

dataset_annmax1['avg'] = dataset_annmax1.mean(axis=1)
print(dataset_annmax1)


legends = ['CESM-vrseasia','CESM-ne30','CESM-fv0.9x1.25','CESM-fv1.9x2.5','CORDEX-ICHEC-EC-EARTH','CORDEX-IPSL-IPSL-CM5A-LR','CORDEX-MPI-M-MPI-ESM-MR','CORDEX-MOHC-HadGEM2-ES','SA-OBS']

colors = ['red','yellow','green','blue','tomato','goldenrod','darkcyan','darkmagenta','black']
line_types = ['dashed','dashed','dashed','dashed','-.','-.','-.','-.','-']

cesm_legends = ['CESM-vrseasia','CESM-ne30','CESM-fv0.9x1.25','CESM-fv1.9x2.5','SA-OBS']
cesm_colors = ['red','orange','green','blue','black']
cesm_line_types = ['dashed','dashed','dashed','dashed','-']


xlabel = 'Year'
ylabel = 'Precip (mm/day)'

print('Plot the annaul max time series for each station')
for idx,istnname in enumerate(stnnames):
    plot_data = [model_annmax1[istnname].values,model_annmax2[istnname].values,model_annmax3[istnname].values,model_annmax4[istnname].values,cordex_annmax1[istnname].values,cordex_annmax2[istnname].values,cordex_annmax3[istnname].values,cordex_annmax4[istnname].values,dataset_annmax1[istnname].values]
    
    title = str(iniyear)+' to '+str(endyear)+'Annual Maximum precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_annual_max_line_vs_cordex_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_lines(yearts,plot_data,colors,line_types,legends,xlabel,ylabel,title,outdir+fname)
 
    plot_data = [model_annmax1[istnname].values,model_annmax2[istnname].values,model_annmax3[istnname].values,model_annmax4[istnname].values,dataset_annmax1[istnname].values]

    title = str(iniyear)+' to '+str(endyear)+'Annual Maximum precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_annual_max_line_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_lines(yearts,plot_data,cesm_colors,cesm_line_types,cesm_legends,xlabel,ylabel,title,outdir+fname)


############################################################################
#plot for mean annual max bar for each station
############################################################################
#bar plot

print(cordex_annmax1.values[~np.isnan(cordex_annmax1.values)])
cordex_annmax_mean1 = cordex_annmax1.mean(axis=0)
cordex_annmax_mean2 = cordex_annmax2.mean(axis=0)
cordex_annmax_mean3 = cordex_annmax3.mean(axis=0)
cordex_annmax_mean4 = cordex_annmax4.mean(axis=0)

model_annmax_mean1 = model_annmax1.mean(axis=0)
model_annmax_mean2 = model_annmax2.mean(axis=0)
model_annmax_mean3 = model_annmax3.mean(axis=0)
model_annmax_mean4 = model_annmax4.mean(axis=0)

dataset_annmax_mean1 = dataset_annmax1.mean(axis=0)
print(dataset_annmax_mean1)
print(dataset_annmax_mean1['avg'])

cordex_annmax_std1 = cordex_annmax1.std(axis=0)
cordex_annmax_std2 = cordex_annmax2.std(axis=0)
cordex_annmax_std3 = cordex_annmax3.std(axis=0)
cordex_annmax_std4 = cordex_annmax4.std(axis=0)

model_annmax_std1 = model_annmax1.std(axis=0)
model_annmax_std2 = model_annmax2.std(axis=0)
model_annmax_std3 = model_annmax3.std(axis=0)
model_annmax_std4 = model_annmax4.std(axis=0)

dataset_annmax_std1 = dataset_annmax1.std(axis=0)

index = np.arange(9)
bar_width = 0.8
opacity = 0.8
shape_type = ['','','','','..','..','..','..','//']

print('Plot the mean annaul max for each station')
for idx,istnname in enumerate(stnnames):
    plot_data = [model_annmax_mean1[istnname],model_annmax_mean2[istnname],model_annmax_mean3[istnname],model_annmax_mean4[istnname],cordex_annmax_mean1[istnname],cordex_annmax_mean2[istnname],cordex_annmax_mean3[istnname],cordex_annmax_mean4[istnname],dataset_annmax_mean1[istnname]]
    plot_err  = [model_annmax_std1[istnname],model_annmax_std2[istnname],model_annmax_std3[istnname],model_annmax_std4[istnname],cordex_annmax_std1[istnname],cordex_annmax_std2[istnname],cordex_annmax_std3[istnname],cordex_annmax_std4[istnname],dataset_annmax_std1[istnname]]

    xlabel = 'Models and SA-OBS'
    ylabel = 'Precip (mm/day)'
    
    title = str(iniyear)+' to '+str(endyear)+' mean annual maximum precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_annual_max_bar_vs_cordex_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_bars(plot_data,plot_err,colors,legends,xlabel,ylabel,title,outdir+fname)

    plot_data = [model_annmax_mean1[istnname],model_annmax_mean2[istnname],model_annmax_mean3[istnname],model_annmax_mean4[istnname],dataset_annmax_mean1[istnname]]
    plot_err  = [model_annmax_std1[istnname],model_annmax_std2[istnname],model_annmax_std3[istnname],model_annmax_std4[istnname],dataset_annmax_std1[istnname]]

    xlabel = 'Models and SA-OBS'
    ylabel = 'Precip (mm/day)'

    title = str(iniyear)+' to '+str(endyear)+' mean annual maximum precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_annual_max_bar_vs_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_bars(plot_data,plot_err,cesm_colors,cesm_legends,xlabel,ylabel,title,outdir+fname)


############################################################################
#plot annual mean ts for each station
############################################################################
#plot annual mean precip for each station
cordex_annmean1 = cordex_var1.resample('A').mean()
cordex_annmean2 = cordex_var2.resample('A').mean()
cordex_annmean3 = cordex_var3.resample('A').mean()
cordex_annmean4 = cordex_var4.resample('A').mean()

model_annmean1 = model_var1.resample('A').mean()
model_annmean2 = model_var2.resample('A').mean()
model_annmean3 = model_var3.resample('A').mean()
model_annmean4 = model_var4.resample('A').mean()

dataset_annmean1 = dataset1.resample('A').mean()

#calculate all station average
cordex_annmean1['avg'] = cordex_annmean1.mean(axis=1)
cordex_annmean2['avg'] = cordex_annmean2.mean(axis=1)
cordex_annmean3['avg'] = cordex_annmean3.mean(axis=1)
cordex_annmean4['avg'] = cordex_annmean4.mean(axis=1)

model_annmean1['avg'] = model_annmean1.mean(axis=1)
model_annmean2['avg'] = model_annmean2.mean(axis=1)
model_annmean3['avg'] = model_annmean3.mean(axis=1)
model_annmean4['avg'] = model_annmean4.mean(axis=1)

dataset_annmean1['avg'] = dataset_annmean1.mean(axis=1)

xlabel = 'Year'
ylabel = 'Precip (mm/day)'

print('Plot the annaul mean time series for each station')
for idx,istnname in enumerate(stnnames):
    plot_data = [model_annmean1[istnname].values,model_annmean2[istnname].values,model_annmean3[istnname].values,model_annmean4[istnname].values,cordex_annmean1[istnname].values,cordex_annmean2[istnname].values,cordex_annmean3[istnname].values,cordex_annmean4[istnname].values,dataset_annmean1[istnname].values]

    title = str(iniyear)+' to '+str(endyear)+'Annual mean precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_annual_mean_line_vs_cordex_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_lines(yearts,plot_data,colors,line_types,legends,xlabel,ylabel,title,outdir+fname)

    plot_data = [model_annmean1[istnname].values,model_annmean2[istnname].values,model_annmean3[istnname].values,model_annmean4[istnname].values,dataset_annmean1[istnname].values]

    title = str(iniyear)+' to '+str(endyear)+'Annual mean precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_annual_mean_line_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_lines(yearts,plot_data,cesm_colors,cesm_line_types,cesm_legends,xlabel,ylabel,title,outdir+fname)


############################################################################
#plot for mean annual mean bar for each station
############################################################################
#bar plot
cordex_annmean_mean1 = cordex_annmean1.mean(axis=0)
cordex_annmean_mean2 = cordex_annmean2.mean(axis=0)
cordex_annmean_mean3 = cordex_annmean3.mean(axis=0)
cordex_annmean_mean4 = cordex_annmean4.mean(axis=0)

model_annmean_mean1 = model_annmean1.mean(axis=0)
model_annmean_mean2 = model_annmean2.mean(axis=0)
model_annmean_mean3 = model_annmean3.mean(axis=0)
model_annmean_mean4 = model_annmean4.mean(axis=0)

dataset_annmean_mean1 = dataset_annmean1.mean(axis=0)
print(cordex_annmean_mean1)

cordex_annmean_std1 = cordex_annmean1.std(axis=0)
cordex_annmean_std2 = cordex_annmean2.std(axis=0)
cordex_annmean_std3 = cordex_annmean3.std(axis=0)
cordex_annmean_std4 = cordex_annmean4.std(axis=0)

model_annmean_std1 = model_annmean1.std(axis=0)
model_annmean_std2 = model_annmean2.std(axis=0)
model_annmean_std3 = model_annmean3.std(axis=0)
model_annmean_std4 = model_annmean4.std(axis=0)

dataset_annmean_std1 = dataset_annmean1.std(axis=0)

index = np.arange(9)
bar_width = 0.8
opacity = 0.8
shape_type = ['','','','','..','..','..','..','//']

print('Plot the mean annaul mean for each station')
for idx,istnname in enumerate(stnnames):

    plot_data = [model_annmean_mean1[istnname],model_annmean_mean2[istnname],model_annmean_mean3[istnname],model_annmean_mean4[istnname],cordex_annmean_mean1[istnname],cordex_annmean_mean2[istnname],cordex_annmean_mean3[istnname],cordex_annmean_mean4[istnname],dataset_annmean_mean1[istnname]]
    plot_err  = [model_annmean_std1[istnname],model_annmean_std2[istnname],model_annmean_std3[istnname],model_annmean_std4[istnname],cordex_annmean_std1[istnname],cordex_annmean_std2[istnname],cordex_annmean_std3[istnname],cordex_annmean_std4[istnname],dataset_annmean_std1[istnname]]

    xlabel = 'Models and SA-OBS'
    ylabel = 'Precip (mm/day)'

    title = str(iniyear)+' to '+str(endyear)+' mean annual mean precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_annual_mean_bar_vs_cordex_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_bars(plot_data,plot_err,colors,legends,xlabel,ylabel,title,outdir+fname)

    plot_data = [model_annmean_mean1[istnname],model_annmean_mean2[istnname],model_annmean_mean3[istnname],model_annmean_mean4[istnname],dataset_annmean_mean1[istnname]]
    plot_err  = [model_annmean_std1[istnname],model_annmean_std2[istnname],model_annmean_std3[istnname],model_annmean_std4[istnname],dataset_annmean_std1[istnname]]

    xlabel = 'Models and SA-OBS'
    ylabel = 'Precip (mm/day)'

    title = str(iniyear)+' to '+str(endyear)+' mean annual mean precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_annual_mean_bar_vs_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_bars(plot_data,plot_err,cesm_colors,cesm_legends,xlabel,ylabel,title,outdir+fname)



############################################################################
#plot monthly mean ts for each station
############################################################################
#plot monthly mean precip for each station
cordex_monmean1 = cordex_var1.resample('M').mean()
cordex_monmean2 = cordex_var2.resample('M').mean()
cordex_monmean3 = cordex_var3.resample('M').mean()
cordex_monmean4 = cordex_var4.resample('M').mean()

model_monmean1 = model_var1.resample('M').mean()
model_monmean2 = model_var2.resample('M').mean()
model_monmean3 = model_var3.resample('M').mean()
model_monmean4 = model_var4.resample('M').mean()

dataset_monmean1 = dataset1.resample('M').mean()
#print(dataset_monmean1.iloc[95:120,:])

cordex_monmean1['avg'] = cordex_monmean1.mean(axis=1)
cordex_monmean2['avg'] = cordex_monmean2.mean(axis=1)
cordex_monmean3['avg'] = cordex_monmean3.mean(axis=1)
cordex_monmean4['avg'] = cordex_monmean4.mean(axis=1)

model_monmean1['avg'] = model_monmean1.mean(axis=1)
model_monmean2['avg'] = model_monmean2.mean(axis=1)
model_monmean3['avg'] = model_monmean3.mean(axis=1)
model_monmean4['avg'] = model_monmean4.mean(axis=1)

dataset_monmean1['avg'] = dataset_monmean1.mean(axis=1)

monthts = np.arange((endyear-iniyear+1)*12) + 1
#monthts = np.delete(monthts,[108,109,110,111,112,113,114,115,116,117,118,119],None)
#print(monthts)
xlabel = 'Month'
ylabel = 'Precip (mm/day)'
xticks = np.arange(6,(endyear-iniyear+1)*12,12)
#xticks = np.delete(xticks,9,None)
#print(xticks)
xticknames = [str(iyear) for iyear in yearts]

print('Plot the monthly mean time series for each station')
for idx,istnname in enumerate(stnnames):
    plot_data = [model_monmean1[istnname].values,model_monmean2[istnname].values,model_monmean3[istnname].values,model_monmean4[istnname].values,cordex_monmean1[istnname].values,cordex_monmean2[istnname].values,cordex_monmean3[istnname].values,cordex_monmean4[istnname].values,dataset_monmean1[istnname].values]

    title = str(iniyear)+' to '+str(endyear)+'Monthly mean precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_monthly_mean_line_vs_cordex_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_lines(monthts,plot_data,colors,line_types,legends,xlabel,ylabel,title,outdir+fname,xticks=xticks,xticknames=xticknames)

    plot_data = [model_monmean1[istnname].values,model_monmean2[istnname].values,model_monmean3[istnname].values,model_monmean4[istnname].values,dataset_monmean1[istnname].values]

    title = str(iniyear)+' to '+str(endyear)+'Monthly mean precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_monthly_mean_line_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_lines(monthts,plot_data,cesm_colors,cesm_line_types,cesm_legends,xlabel,ylabel,title,outdir+fname,xticks=xticks,xticknames=xticknames)


############################################################################
#plot climatological mean for each station
############################################################################

xlabel = 'Month'
ylabel = 'Precip (mm/day)'

print('Plot the seasonalities for each station')
for idx,istnname in enumerate(stnnames):
    cordex_mean1, codex_std1 = mon2clim(cordex_monmean1[istnname].values)
    cordex_mean2, codex_std2 = mon2clim(cordex_monmean2[istnname].values)
    cordex_mean3, codex_std3 = mon2clim(cordex_monmean3[istnname].values)
    cordex_mean4, codex_std4 = mon2clim(cordex_monmean4[istnname].values)

    model_mean1, model_std1 = mon2clim(model_monmean1[istnname].values)
    model_mean2, model_std2 = mon2clim(model_monmean2[istnname].values)
    model_mean3, model_std3 = mon2clim(model_monmean3[istnname].values)
    model_mean4, model_std4 = mon2clim(model_monmean4[istnname].values)

    dataset_mean1, dataset_std1 = mon2clim(dataset_monmean1[istnname].values)
    #print(dataset_mean1)

    plot_data = [model_mean1,model_mean2,model_mean3,model_mean4,cordex_mean1,cordex_mean2,cordex_mean3,cordex_mean4,dataset_mean1]
    plot_err  = [model_std1,model_std2,model_std3,model_std4,codex_std1,codex_std2,codex_std3,codex_std4,dataset_std1]

    title = str(iniyear)+' to '+str(endyear)+'Seasonal cycle of precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_clim_mean_line_vs_cordex_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_lines(months,plot_data,colors,line_types,legends,xlabel,ylabel,title,outdir+fname,yerr=plot_err)

    plot_data = [model_mean1,model_mean2,model_mean3,model_mean4,dataset_mean1]
    plot_err  = [model_std1,model_std2,model_std3,model_std4,dataset_std1]

    title = str(iniyear)+' to '+str(endyear)+'Seasonal cycle of precip in the station '+str(stnids[idx]).zfill(6)+': '+istnname+' in '+countrynames[idx]
    fname = 'vrcesm_prect_clim_mean_line_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'
    plot_lines(months,plot_data,cesm_colors,cesm_line_types,cesm_legends,xlabel,ylabel,title,outdir+fname,yerr=plot_err)


############################################################################
#plot histogram for each station
############################################################################

print('Plot the precip histogram for each station')
for idx,istnname in enumerate(stnnames):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    tempdata = dataset1[istnname].values
    #print(tempdata)
    #print(len(tempdata))
    binmax  = np.amax(tempdata[~np.isnan(tempdata)])*1./2.
    #print(binmax)
    binarray = np.arange(0,binmax,binmax/30)

    plot_data =[model_var1[istnname].values,model_var2[istnname].values,model_var3[istnname].values,model_var4[istnname].values,cordex_var1[istnname].values,cordex_var2[istnname].values,cordex_var3[istnname].values,cordex_var4[istnname].values,dataset1[istnname].values]

    for ii in range(9):
        tempdata = plot_data[ii]
        y,binEdges=np.histogram(tempdata[~np.isnan(tempdata)],bins=binarray)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        #print(bincenters)
        plt.plot(bincenters,y,c=colors[ii],linestyle=line_types[ii],linewidth=1.5, label = legends[ii])

    plt.yscale('log')
    plt.legend(handlelength=4,fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("Days")
    plt.xlabel("Precip(mm/day)")

    title = str(iniyear)+' to '+str(endyear)+' Total precip distribution in the station '+str(stnids[idx]).zfill(6)+': '+stnnames[idx]+' in '+countrynames[idx]
    fname = 'vrcesm_prect_hist_vs_cordex_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'

    plt.suptitle(title,fontsize=9,y=0.95)
    plt.savefig(outdir+fname,bbox_inches='tight')
    plt.close(fig)

    
    #plot for only cesm
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_data =[model_var1[istnname].values,model_var2[istnname].values,model_var3[istnname].values,model_var4[istnname].values,dataset1[istnname].values]

    for ii in range(5):
        tempdata = plot_data[ii]
        y,binEdges=np.histogram(tempdata[~np.isnan(tempdata)],bins=binarray)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        #print(bincenters)
        plt.plot(bincenters,y,c=cesm_colors[ii],linestyle=cesm_line_types[ii],linewidth=1.5, label = cesm_legends[ii])

    plt.yscale('log')
    plt.legend(handlelength=4,fontsize=5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("Days")
    plt.xlabel("Precip(mm/day)")

    title = str(iniyear)+' to '+str(endyear)+' Total precip distribution in the station '+str(stnids[idx]).zfill(6)+': '+stnnames[idx]+' in '+countrynames[idx]
    fname = 'vrcesm_prect_hist_refSAOBS_station_'+str(idx+1)+'_'+countrynames[idx]+'.pdf'

    plt.suptitle(title,fontsize=9,y=0.95)
    plt.savefig(outdir+fname,bbox_inches='tight')
    plt.close(fig)


############################################################################
#plot for all stations
############################################################################

############################################################################
#plot for annual max ts for all station 
############################################################################

xlabel = 'Year'
ylabel = 'Precip (mm/day)'

print('Plot the annaul max time series for all stations')
plot_data = [model_annmax1['avg'].values,model_annmax2['avg'].values,model_annmax3['avg'].values,model_annmax4['avg'].values,cordex_annmax1['avg'].values,cordex_annmax2['avg'].values,cordex_annmax3['avg'].values,cordex_annmax4['avg'].values,dataset_annmax1['avg'].values]

title = str(iniyear)+' to '+str(endyear)+'Annual Maximum precip averaged over all stations'
fname = 'vrcesm_prect_annual_max_line_vs_cordex_refSAOBS_all_stations.pdf'
plot_lines(yearts,plot_data,colors,line_types,legends,xlabel,ylabel,title,outdir+fname)

plot_data = [model_annmax1['avg'].values,model_annmax2['avg'].values,model_annmax3['avg'].values,model_annmax4['avg'].values,dataset_annmax1['avg'].values]

title = str(iniyear)+' to '+str(endyear)+'Annual Maximum precip averaged over all station'
fname = 'vrcesm_prect_annual_max_line_refSAOBS_all_stations.pdf'
plot_lines(yearts,plot_data,cesm_colors,cesm_line_types,cesm_legends,xlabel,ylabel,title,outdir+fname)


############################################################################
#plot for mean annual max bar for all stations
############################################################################

index = np.arange(9)
bar_width = 0.8
opacity = 0.8
shape_type = ['','','','','..','..','..','..','//']

print('Plot the mean annaul max for all stations')
plot_data = [model_annmax_mean1['avg'],model_annmax_mean2['avg'],model_annmax_mean3['avg'],model_annmax_mean4['avg'],cordex_annmax_mean1['avg'],cordex_annmax_mean2['avg'],cordex_annmax_mean3['avg'],cordex_annmax_mean4['avg'],dataset_annmax_mean1['avg']]
plot_err  = [model_annmax_std1['avg'],model_annmax_std2['avg'],model_annmax_std3['avg'],model_annmax_std4['avg'],cordex_annmax_std1['avg'],cordex_annmax_std2['avg'],cordex_annmax_std3['avg'],cordex_annmax_std4['avg'],dataset_annmax_std1['avg']]

xlabel = 'Models and SA-OBS'
ylabel = 'Precip (mm/day)'

title = str(iniyear)+' to '+str(endyear)+' mean annual maximum precip averaged over all stations'
fname = 'vrcesm_prect_annual_max_bar_vs_cordex_refSAOBS_all_stations.pdf'
plot_bars(plot_data,plot_err,colors,legends,xlabel,ylabel,title,outdir+fname)

plot_data = [model_annmax_mean1['avg'],model_annmax_mean2['avg'],model_annmax_mean3['avg'],model_annmax_mean4['avg'],dataset_annmax_mean1['avg']]
plot_err  = [model_annmax_std1['avg'],model_annmax_std2['avg'],model_annmax_std3['avg'],model_annmax_std4['avg'],dataset_annmax_std1['avg']]

xlabel = 'Models and SA-OBS'
ylabel = 'Precip (mm/day)'

title = str(iniyear)+' to '+str(endyear)+' mean annual maximum precip averaged over all station'
fname = 'vrcesm_prect_annual_max_bar_vs_refSAOBS_all_stations.pdf'
plot_bars(plot_data,plot_err,cesm_colors,cesm_legends,xlabel,ylabel,title,outdir+fname)



############################################################################
#plot annual mean ts for all stations
############################################################################

xlabel = 'Year'
ylabel = 'Precip (mm/day)'

print('Plot the annaul mean time series for all stations')
plot_data = [model_annmean1['avg'].values,model_annmean2['avg'].values,model_annmean3['avg'].values,model_annmean4['avg'].values,cordex_annmean1['avg'].values,cordex_annmean2['avg'].values,cordex_annmean3['avg'].values,cordex_annmean4['avg'].values,dataset_annmean1['avg'].values]

title = str(iniyear)+' to '+str(endyear)+'Annual mean precip averaged over all stations' 
fname = 'vrcesm_prect_annual_mean_line_vs_cordex_refSAOBS_all_stations.pdf'
plot_lines(yearts,plot_data,colors,line_types,legends,xlabel,ylabel,title,outdir+fname)

plot_data = [model_annmean1['avg'].values,model_annmean2['avg'].values,model_annmean3['avg'].values,model_annmean4['avg'].values,dataset_annmean1['avg'].values]

title = str(iniyear)+' to '+str(endyear)+'Annual mean precip averaged over all stations'
fname = 'vrcesm_prect_annual_mean_line_refSAOBS_all_stations.pdf'
plot_lines(yearts,plot_data,cesm_colors,cesm_line_types,cesm_legends,xlabel,ylabel,title,outdir+fname)


############################################################################
#plot for mean annual mean bar for all stations
############################################################################

index = np.arange(9)
bar_width = 0.8
opacity = 0.8
shape_type = ['','','','','..','..','..','..','//']

print('Plot the mean annaul mean for all stations')

plot_data = [model_annmean_mean1['avg'],model_annmean_mean2['avg'],model_annmean_mean3['avg'],model_annmean_mean4['avg'],cordex_annmean_mean1['avg'],cordex_annmean_mean2['avg'],cordex_annmean_mean3['avg'],cordex_annmean_mean4['avg'],dataset_annmean_mean1['avg']]
plot_err  = [model_annmean_std1['avg'],model_annmean_std2['avg'],model_annmean_std3['avg'],model_annmean_std4['avg'],cordex_annmean_std1['avg'],cordex_annmean_std2['avg'],cordex_annmean_std3['avg'],cordex_annmean_std4['avg'],dataset_annmean_std1['avg']]

xlabel = 'Models and SA-OBS'
ylabel = 'Precip (mm/day)'

title = str(iniyear)+' to '+str(endyear)+' mean annual mean precip averaged over all stations'
fname = 'vrcesm_prect_annual_mean_bar_vs_cordex_refSAOBS_all_stations.pdf'
plot_bars(plot_data,plot_err,colors,legends,xlabel,ylabel,title,outdir+fname)

plot_data = [model_annmean_mean1['avg'],model_annmean_mean2['avg'],model_annmean_mean3['avg'],model_annmean_mean4['avg'],dataset_annmean_mean1['avg']]
plot_err  = [model_annmean_std1['avg'],model_annmean_std2['avg'],model_annmean_std3['avg'],model_annmean_std4['avg'],dataset_annmean_std1['avg']]

xlabel = 'Models and SA-OBS'
ylabel = 'Precip (mm/day)'

title = str(iniyear)+' to '+str(endyear)+' mean annual mean precip averaged over all stations'
fname = 'vrcesm_prect_annual_mean_bar_vs_refSAOBS_all_stations.pdf'
plot_bars(plot_data,plot_err,cesm_colors,cesm_legends,xlabel,ylabel,title,outdir+fname)


############################################################################
#plot month mean ts for all stations
############################################################################

monthts = np.arange((endyear-iniyear+1)*12) + 1
#monthts = np.delete(monthts,[108,109,110,111,112,113,114,115,116,117,118,119],None)
#print(monthts)
xlabel = 'Month'
ylabel = 'Precip (mm/day)'
xticks = np.arange(6,(endyear-iniyear+1)*12,12)
#xticks = np.delete(xticks,9,None)
#print(xticks)
xticknames = [str(iyear) for iyear in yearts]

print('Plot the monthly mean time series for all stations')
plot_data = [model_monmean1['avg'].values,model_monmean2['avg'].values,model_monmean3['avg'].values,model_monmean4['avg'].values,cordex_monmean1['avg'].values,cordex_monmean2['avg'].values,cordex_monmean3['avg'].values,cordex_monmean4['avg'].values,dataset_monmean1['avg'].values]

title = str(iniyear)+' to '+str(endyear)+'Monthly mean precip averaged over all stations'
fname = 'vrcesm_prect_monthly_mean_line_vs_cordex_refSAOBS_all_stations.pdf'
plot_lines(monthts,plot_data,colors,line_types,legends,xlabel,ylabel,title,outdir+fname,xticks=xticks,xticknames=xticknames)

plot_data = [model_monmean1['avg'].values,model_monmean2['avg'].values,model_monmean3['avg'].values,model_monmean4['avg'].values,dataset_monmean1['avg'].values]

title = str(iniyear)+' to '+str(endyear)+'Monthly mean precip averaged over all stations'
fname = 'vrcesm_prect_monthly_mean_line_refSAOBS_all_stations.pdf'
plot_lines(monthts,plot_data,cesm_colors,cesm_line_types,cesm_legends,xlabel,ylabel,title,outdir+fname,xticks=xticks,xticknames=xticknames)


############################################################################
#plot climatological mean for all stations
############################################################################

xlabel = 'Month'
ylabel = 'Precip (mm/day)'


print('Plot the seasonalities for all stations')
cordex_mean1, codex_std1 = mon2clim(cordex_monmean1['avg'].values)
cordex_mean2, codex_std2 = mon2clim(cordex_monmean2['avg'].values)
cordex_mean3, codex_std3 = mon2clim(cordex_monmean3['avg'].values)
cordex_mean4, codex_std4 = mon2clim(cordex_monmean4['avg'].values)

model_mean1, model_std1 = mon2clim(model_monmean1['avg'].values)
model_mean2, model_std2 = mon2clim(model_monmean2['avg'].values)
model_mean3, model_std3 = mon2clim(model_monmean3['avg'].values)
model_mean4, model_std4 = mon2clim(model_monmean4['avg'].values)

dataset_mean1, dataset_std1 = mon2clim(dataset_monmean1['avg'].values)


plot_data = [model_mean1,model_mean2,model_mean3,model_mean4,cordex_mean1,cordex_mean2,cordex_mean3,cordex_mean4,dataset_mean1]
plot_err  = [model_std1,model_std2,model_std3,model_std4,codex_std1,codex_std2,codex_std3,codex_std4,dataset_std1]

title = str(iniyear)+' to '+str(endyear)+'Seasonal cycle of precip averaged over all stations'
fname = 'vrcesm_prect_clim_mean_line_vs_cordex_refSAOBS_all_stations.pdf'
plot_lines(months,plot_data,colors,line_types,legends,xlabel,ylabel,title,outdir+fname,yerr=plot_err)

plot_data = [model_mean1,model_mean2,model_mean3,model_mean4,dataset_mean1]
plot_err  = [model_std1,model_std2,model_std3,model_std4,dataset_std1]

title = str(iniyear)+' to '+str(endyear)+'Seasonal cycle of precip averaged over all stations'
fname = 'vrcesm_prect_clim_mean_line_refSAOBS_all_stations.pdf'
plot_lines(months,plot_data,cesm_colors,cesm_line_types,cesm_legends,xlabel,ylabel,title,outdir+fname,yerr=plot_err)


############################################################################
#plot histogram for all stations
############################################################################

print('Plot the precip histogram for all stations')
fig = plt.figure()
ax = fig.add_subplot(111)

tempdata = dataset1.values
#print(tempdata)
#print(len(tempdata))
binmax  = np.amax(tempdata[~np.isnan(tempdata)])*1./2.
#print(binmax)
binarray = np.arange(0,binmax,binmax/30)

plot_data =[model_var1.values,model_var2.values,model_var3.values,model_var4.values,cordex_var1.values,cordex_var2.values,cordex_var3.values,cordex_var4.values,dataset1.values]

for ii in range(9):
    tempdata = plot_data[ii]
    y,binEdges=np.histogram(tempdata[~np.isnan(tempdata)],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    #print(bincenters)
    plt.plot(bincenters,y,c=colors[ii],linestyle=line_types[ii],linewidth=1.5, label = legends[ii])

plt.yscale('log')
plt.legend(handlelength=4,fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel("Days")
plt.xlabel("Precip(mm/day)")

title = str(iniyear)+' to '+str(endyear)+' Total precip distribution over all stations'
fname = 'vrcesm_prect_hist_vs_cordex_refSAOBS_all_stations.pdf'

plt.suptitle(title,fontsize=9,y=0.95)
plt.savefig(outdir+fname,bbox_inches='tight')
plt.close(fig)


#plot for only cesm
fig = plt.figure()
ax = fig.add_subplot(111)

plot_data =[model_var1.values,model_var2.values,model_var3.values,model_var4.values,dataset1.values]

for ii in range(5):
    tempdata = plot_data[ii]
    y,binEdges=np.histogram(tempdata[~np.isnan(tempdata)],bins=binarray)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    #print(bincenters)
    plt.plot(bincenters,y,c=cesm_colors[ii],linestyle=cesm_line_types[ii],linewidth=1.5, label = cesm_legends[ii])

plt.yscale('log')
plt.legend(handlelength=4,fontsize=5)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylabel("Days")
plt.xlabel("Precip(mm/day)")

title = str(iniyear)+' to '+str(endyear)+' Total precip distribution over all stations'
fname = 'vrcesm_prect_hist_refSAOBS_all_stations.pdf'

plt.suptitle(title,fontsize=9,y=0.95)
plt.savefig(outdir+fname,bbox_inches='tight')
plt.close(fig)




