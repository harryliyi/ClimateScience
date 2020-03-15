#this script is used to compare vrcesm against fv and se
#here seasonal climatology is compared
#by Harry Li

#import libraries
import numpy as np
from scipy.stats.stats import pearsonr
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import interp
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm

#set up cesm data directories and filenames
dir1 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/vrseasia_AMIP_1979_to_2005/atm/hist/"
dir2 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/ne30_ne30_AMIP_1979_to_2005/atm/hist/"
dir3 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f09_f09_AMIP_1979_to_2005/atm/hist/"
dir4 = "/scratch/d/dylan/harryli/cesm1/vrcesm/archive/f19_f19_AMIP_1979_to_2005/atm/hist/"
fname1 = "fv02_TGCLDWP_vrseasia_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
fname2 = "fv09_TGCLDWP_ne30_ne30_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
fname3 = "TGCLDWP_f09_f09_AMIP_1979_to_2005.cam.h0.1979-2005.nc"
fname4 = "TGCLDWP_f19_f19_AMIP_1979_to_2005.cam.h0.1979-2005.nc"

mod1_in = Dataset(dir1+fname1)
mod2_in = Dataset(dir2+fname2)
mod3_in = Dataset(dir3+fname3)
mod4_in = Dataset(dir4+fname4)

model_list = ['vrseasia','ne30','fv0.9x1.25','fv1.9x2.5']

#set up output directory and output log
outdir = "/scratch/d/dylan/harryli/cesm1/vrcesm/Analysis/vrseasia_AMIP_1979_to_2005/radflx/"
outlog = open(outdir+"vrcesm_tgcldwp_modres_line_plot_output.log", "w")

#define inital year and end year
iniyear = 1980
endyear = 2005
yearts  = np.arange(iniyear,endyear+1)
print(yearts)

#set up variable name
var1name = 'TGCLDLWP'
var1str = 'Cloud liquid water path'
var2name = 'TGCLDIWP'
var2str = 'Cloud ice water path'

#define mainland Southeast Asia region
reg_lats = [10,25]
reg_lons = [100,110]

#define days of each month in the leap year and normal year
leapmonthday = np.array([31,29,31,30,31,30,31,31,30,31,30,31],dtype=np.float)
monthday = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.float)
month = np.arange(1,13,1)
mname = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def interp3d(var,xin,yin,xout,yout,opt):
    
    newvar = np.zeros([var.shape[0],xout.shape[0],xout.shape[1]])
    #print(var.shape[0])
    #print(xout.shape[0])
    #print(xout.shape[1])

    for idx in range(var.shape[0]):
        newvar[idx,:,:] = interp(var[idx,:,:],xin,yin,xout,yout,order=opt)

    return newvar;


###########################################################
#read data
###########################################################

print("Reading the data from model...")
#model  
#vrseasia
mod1_lats = mod1_in.variables['lat'][:]
mod1_lons = mod1_in.variables['lon'][:]
mod1_var1 = mod1_in.variables[var1name][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, :,:]
mod1_var1_zon = 1000. * np.mean(mod1_var1[:,:,:],axis=2)
mod1_var2 = mod1_in.variables[var2name][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, :,:]
mod1_var2_zon = 1000. * np.mean(mod1_var2[:,:,:],axis=2)

#ne30
mod2_lats = mod2_in.variables['lat'][:]
mod2_lons = mod2_in.variables['lon'][:]
mod2_var1 = mod2_in.variables[var1name][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, :,:]
mod2_var1_zon = 1000. * np.mean(mod2_var1[:,:,:],axis=2)
mod2_var2 = mod2_in.variables[var2name][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, :,:]
mod2_var2_zon = 1000. * np.mean(mod2_var2[:,:,:],axis=2)


#fv0.9x1.25
mod3_lats = mod3_in.variables['lat'][:]
mod3_lons = mod3_in.variables['lon'][:]
mod3_var1 = mod3_in.variables[var1name][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, :,:]
mod3_var1_zon = 1000. * np.mean(mod3_var1[:,:,:],axis=2)
mod3_var2 = mod3_in.variables[var2name][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, :,:]
mod3_var2_zon = 1000. * np.mean(mod3_var2[:,:,:],axis=2)

#fv1.9x2.5
mod4_lats = mod4_in.variables['lat'][:]
mod4_lons = mod4_in.variables['lon'][:]
mod4_var1 = mod4_in.variables[var1name][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, :,:]
mod4_var1_zon = 1000. * np.mean(mod4_var1[:,:,:],axis=2)
mod4_var2 = mod4_in.variables[var2name][(iniyear-1979)*12 : (endyear-1979 + 1) * 12, :,:]
mod4_var2_zon = 1000. * np.mean(mod4_var2[:,:,:],axis=2)

#regrid data
lons_reg, lats_reg = np.meshgrid(mod4_lons, mod4_lats)

mod1_var1_reg = interp3d(mod1_var1,mod1_lons,mod1_lats, lons_reg, lats_reg,1)
mod1_var2_reg = interp3d(mod1_var2,mod1_lons,mod1_lats, lons_reg, lats_reg,1)
mod2_var1_reg = interp3d(mod2_var1,mod2_lons,mod2_lats, lons_reg, lats_reg,1)
mod2_var2_reg = interp3d(mod2_var2,mod2_lons,mod2_lats, lons_reg, lats_reg,1)
mod3_var1_reg = interp3d(mod3_var1,mod3_lons,mod3_lats, lons_reg, lats_reg,1)
mod3_var2_reg = interp3d(mod3_var2,mod3_lons,mod3_lats, lons_reg, lats_reg,1)

#calculate zonal average and differences
mod1_var1_reg_zon = 1000. * np.mean(mod1_var1_reg[:,:,:],axis=2) - mod4_var1_zon
mod1_var2_reg_zon = 1000. * np.mean(mod1_var2_reg[:,:,:],axis=2) - mod4_var2_zon
mod2_var1_reg_zon = 1000. * np.mean(mod2_var1_reg[:,:,:],axis=2) - mod4_var1_zon
mod2_var2_reg_zon = 1000. * np.mean(mod2_var2_reg[:,:,:],axis=2) - mod4_var2_zon 
mod3_var1_reg_zon = 1000. * np.mean(mod3_var1_reg[:,:,:],axis=2) - mod4_var1_zon
mod3_var2_reg_zon = 1000. * np.mean(mod3_var2_reg[:,:,:],axis=2) - mod4_var2_zon

print("-------------------")

###########################################################
#calculate seasonal climatology
###########################################################
season_index = ['DJF','MAM','JJA','SON','ANN']

mod1_var1_seas = np.zeros([5,len(mod1_lats)])
mod2_var1_seas = np.zeros([5,len(mod2_lats)])
mod3_var1_seas = np.zeros([5,len(mod3_lats)])
mod4_var1_seas = np.zeros([5,len(mod4_lats)])

mod1_var2_seas = np.zeros([5,len(mod1_lats)])
mod2_var2_seas = np.zeros([5,len(mod2_lats)])
mod3_var2_seas = np.zeros([5,len(mod3_lats)])
mod4_var2_seas = np.zeros([5,len(mod4_lats)])

mod1_var1_reg_seas = np.zeros([5,len(mod4_lats)])
mod2_var1_reg_seas = np.zeros([5,len(mod4_lats)])
mod3_var1_reg_seas = np.zeros([5,len(mod4_lats)])

mod1_var2_reg_seas = np.zeros([5,len(mod4_lats)])
mod2_var2_reg_seas = np.zeros([5,len(mod4_lats)])
mod3_var2_reg_seas = np.zeros([5,len(mod4_lats)])

for idx,imonth in enumerate(month):
    #print(imonth/3.)
    #print(np.arange(idx,(endyear-iniyear+1)*12+idx,12))
    if (imonth/3.>=0.)&(imonth/3.<1.):
        mod1_var1_seas[0,:] = mod1_var1_seas[0,:] + np.mean(mod1_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_seas[0,:] = mod2_var1_seas[0,:] + np.mean(mod2_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_seas[0,:] = mod3_var1_seas[0,:] + np.mean(mod3_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var1_seas[0,:] = mod4_var1_seas[0,:] + np.mean(mod4_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_seas[0,:] = mod1_var2_seas[0,:] + np.mean(mod1_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_seas[0,:] = mod2_var2_seas[0,:] + np.mean(mod2_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_seas[0,:] = mod3_var2_seas[0,:] + np.mean(mod3_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var2_seas[0,:] = mod4_var2_seas[0,:] + np.mean(mod4_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var1_reg_seas[0,:] = mod1_var1_reg_seas[0,:] + np.mean(mod1_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_reg_seas[0,:] = mod2_var1_reg_seas[0,:] + np.mean(mod2_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_reg_seas[0,:] = mod3_var1_reg_seas[0,:] + np.mean(mod3_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_reg_seas[0,:] = mod1_var2_reg_seas[0,:] + np.mean(mod1_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_reg_seas[0,:] = mod2_var2_reg_seas[0,:] + np.mean(mod2_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_reg_seas[0,:] = mod3_var2_reg_seas[0,:] + np.mean(mod3_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

    if (imonth/3.>=1.)&(imonth/3.<2.):
        mod1_var1_seas[1,:] = mod1_var1_seas[1,:] + np.mean(mod1_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_seas[1,:] = mod2_var1_seas[1,:] + np.mean(mod2_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_seas[1,:] = mod3_var1_seas[1,:] + np.mean(mod3_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var1_seas[1,:] = mod4_var1_seas[1,:] + np.mean(mod4_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_seas[1,:] = mod1_var2_seas[1,:] + np.mean(mod1_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_seas[1,:] = mod2_var2_seas[1,:] + np.mean(mod2_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_seas[1,:] = mod3_var2_seas[1,:] + np.mean(mod3_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var2_seas[1,:] = mod4_var2_seas[1,:] + np.mean(mod4_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var1_reg_seas[1,:] = mod1_var1_reg_seas[1,:] + np.mean(mod1_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_reg_seas[1,:] = mod2_var1_reg_seas[1,:] + np.mean(mod2_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_reg_seas[1,:] = mod3_var1_reg_seas[1,:] + np.mean(mod3_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_reg_seas[1,:] = mod1_var2_reg_seas[1,:] + np.mean(mod1_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_reg_seas[1,:] = mod2_var2_reg_seas[1,:] + np.mean(mod2_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_reg_seas[1,:] = mod3_var2_reg_seas[1,:] + np.mean(mod3_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

    if (imonth/3.>=2.)&(imonth/3.<3.):
        mod1_var1_seas[2,:] = mod1_var1_seas[2,:] + np.mean(mod1_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_seas[2,:] = mod2_var1_seas[2,:] + np.mean(mod2_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_seas[2,:] = mod3_var1_seas[2,:] + np.mean(mod3_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var1_seas[2,:] = mod4_var1_seas[2,:] + np.mean(mod4_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_seas[2,:] = mod1_var2_seas[2,:] + np.mean(mod1_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_seas[2,:] = mod2_var2_seas[2,:] + np.mean(mod2_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_seas[2,:] = mod3_var2_seas[2,:] + np.mean(mod3_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var2_seas[2,:] = mod4_var2_seas[2,:] + np.mean(mod4_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var1_reg_seas[2,:] = mod1_var1_reg_seas[2,:] + np.mean(mod1_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_reg_seas[2,:] = mod2_var1_reg_seas[2,:] + np.mean(mod2_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_reg_seas[2,:] = mod3_var1_reg_seas[2,:] + np.mean(mod3_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_reg_seas[2,:] = mod1_var2_reg_seas[2,:] + np.mean(mod1_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_reg_seas[2,:] = mod2_var2_reg_seas[2,:] + np.mean(mod2_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_reg_seas[2,:] = mod3_var2_reg_seas[2,:] + np.mean(mod3_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

    if (imonth/3.>=3.)&(imonth/3.<4.):
        mod1_var1_seas[3,:] = mod1_var1_seas[3,:] + np.mean(mod1_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_seas[3,:] = mod2_var1_seas[3,:] + np.mean(mod2_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_seas[3,:] = mod3_var1_seas[3,:] + np.mean(mod3_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var1_seas[3,:] = mod4_var1_seas[3,:] + np.mean(mod4_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_seas[3,:] = mod1_var2_seas[3,:] + np.mean(mod1_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_seas[3,:] = mod2_var2_seas[3,:] + np.mean(mod2_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_seas[3,:] = mod3_var2_seas[3,:] + np.mean(mod3_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var2_seas[3,:] = mod4_var2_seas[3,:] + np.mean(mod4_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var1_reg_seas[3,:] = mod1_var1_reg_seas[3,:] + np.mean(mod1_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_reg_seas[3,:] = mod2_var1_reg_seas[3,:] + np.mean(mod2_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_reg_seas[3,:] = mod3_var1_reg_seas[3,:] + np.mean(mod3_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_reg_seas[3,:] = mod1_var2_reg_seas[3,:] + np.mean(mod1_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_reg_seas[3,:] = mod2_var2_reg_seas[3,:] + np.mean(mod2_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_reg_seas[3,:] = mod3_var2_reg_seas[3,:] + np.mean(mod3_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

    if (imonth/3.>=4.):
        mod1_var1_seas[0,:] = mod1_var1_seas[0,:] + np.mean(mod1_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_seas[0,:] = mod2_var1_seas[0,:] + np.mean(mod2_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_seas[0,:] = mod3_var1_seas[0,:] + np.mean(mod3_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var1_seas[0,:] = mod4_var1_seas[0,:] + np.mean(mod4_var1_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_seas[0,:] = mod1_var2_seas[0,:] + np.mean(mod1_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_seas[0,:] = mod2_var2_seas[0,:] + np.mean(mod2_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_seas[0,:] = mod3_var2_seas[0,:] + np.mean(mod3_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod4_var2_seas[0,:] = mod4_var2_seas[0,:] + np.mean(mod4_var2_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var1_reg_seas[0,:] = mod1_var1_reg_seas[0,:] + np.mean(mod1_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var1_reg_seas[0,:] = mod2_var1_reg_seas[0,:] + np.mean(mod2_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var1_reg_seas[0,:] = mod3_var1_reg_seas[0,:] + np.mean(mod3_var1_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

        mod1_var2_reg_seas[0,:] = mod1_var2_reg_seas[0,:] + np.mean(mod1_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod2_var2_reg_seas[0,:] = mod2_var2_reg_seas[0,:] + np.mean(mod2_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)
        mod3_var2_reg_seas[0,:] = mod3_var2_reg_seas[0,:] + np.mean(mod3_var2_reg_zon[idx:(endyear-iniyear+1)*12+idx:12,:],axis=0)

mod1_var1_seas = mod1_var1_seas/4.
mod2_var1_seas = mod2_var1_seas/4.
mod3_var1_seas = mod3_var1_seas/4.
mod4_var1_seas = mod4_var1_seas/4.

mod1_var2_seas = mod1_var2_seas/4.
mod2_var2_seas = mod2_var2_seas/4.
mod3_var2_seas = mod3_var2_seas/4.
mod4_var2_seas = mod4_var2_seas/4.

mod1_var1_reg_seas = mod1_var1_reg_seas/4.
mod2_var1_reg_seas = mod2_var1_reg_seas/4.
mod3_var1_reg_seas = mod3_var1_reg_seas/4.

mod1_var2_reg_seas = mod1_var2_reg_seas/4.
mod2_var2_reg_seas = mod2_var2_reg_seas/4.
mod3_var2_reg_seas = mod3_var2_reg_seas/4.

mod1_var1_seas[4,:] = np.mean(mod1_var1_zon[:,:],axis=0)
mod2_var1_seas[4,:] = np.mean(mod2_var1_zon[:,:],axis=0)
mod3_var1_seas[4,:] = np.mean(mod3_var1_zon[:,:],axis=0)
mod4_var1_seas[4,:] = np.mean(mod4_var1_zon[:,:],axis=0)

mod1_var2_seas[4,:] = np.mean(mod1_var2_zon[:,:],axis=0)
mod2_var2_seas[4,:] = np.mean(mod2_var2_zon[:,:],axis=0)
mod3_var2_seas[4,:] = np.mean(mod3_var2_zon[:,:],axis=0)
mod4_var2_seas[4,:] = np.mean(mod4_var2_zon[:,:],axis=0)

mod1_var1_reg_seas[4,:] = np.mean(mod1_var1_reg_zon[:,:],axis=0)
mod2_var1_reg_seas[4,:] = np.mean(mod2_var1_reg_zon[:,:],axis=0)
mod3_var1_reg_seas[4,:] = np.mean(mod3_var1_reg_zon[:,:],axis=0)

mod1_var2_reg_seas[4,:] = np.mean(mod1_var2_reg_zon[:,:],axis=0)
mod2_var2_reg_seas[4,:] = np.mean(mod2_var2_reg_zon[:,:],axis=0)
mod3_var2_reg_seas[4,:] = np.mean(mod3_var2_reg_zon[:,:],axis=0)

###########################################################
#draw graph for each season
########################################################### 

for idx in np.arange(5):
    plt.figure(idx)
    plt.clf()

    plt.plot(mod1_lats, mod1_var1_seas[idx,:] , color='red', linestyle='dashed',linewidth=1, label = 'vrseasia')
    plt.plot(mod2_lats, mod2_var1_seas[idx,:] , color='yellow', linestyle='dashed',linewidth=1, label = 'ne30')
    plt.plot(mod3_lats, mod3_var1_seas[idx,:] , color='green', linestyle='dashed',linewidth=1, label = 'fv0.9x1.25')
    plt.plot(mod4_lats, mod4_var1_seas[idx,:] , color='blue', linestyle='dashed',linewidth=1, label = 'fv1.9x2.5')

    plt.legend(fontsize=7)
    plt.xlabel('Latitude', fontsize=10)
    plt.ylabel('Water path (g/m^2)', fontsize=10)
    plt.title(str(iniyear)+'-'+str(endyear)+' '+season_index[idx]+' '+var1str+' Resolustion Comparison')

    plt.savefig(outdir+'vrseasia_'+var1name+'_modres_line_season_'+str(idx+1)+'.pdf')

plt.clf()
for idx in np.arange(5):
    plt.figure(idx)
    plt.clf()

    plt.plot(mod1_lats, mod1_var2_seas[idx,:] , color='red', linestyle='solid',linewidth=1, label = 'vrseasia')
    plt.plot(mod2_lats, mod2_var2_seas[idx,:] , color='yellow', linestyle='solid',linewidth=1, label = 'ne30')
    plt.plot(mod3_lats, mod3_var2_seas[idx,:] , color='green', linestyle='solid',linewidth=1, label = 'fv0.9x1.25')
    plt.plot(mod4_lats, mod4_var2_seas[idx,:] , color='blue', linestyle='solid',linewidth=1, label = 'fv1.9x2.5')

    plt.legend(fontsize=7)
    plt.xlabel('Latitude', fontsize=10)
    plt.ylabel('Water path (g/m^2)', fontsize=10)
    plt.title(str(iniyear)+'-'+str(endyear)+' '+season_index[idx]+' '+var2str+' Resolustion Comparison')

    plt.savefig(outdir+'vrseasia_'+var2name+'_modres_line_season_'+str(idx+1)+'.pdf')

plt.clf()
for idx in np.arange(5):
    plt.figure(idx)
    plt.clf()

    plt.plot(mod1_lats, mod1_var1_seas[idx,:] , color='red', linestyle='dashed',linewidth=1, label = 'vrseasia '+var1name)
    plt.plot(mod2_lats, mod2_var1_seas[idx,:] , color='yellow', linestyle='dashed',linewidth=1, label = 'ne30 '+var1name)
    plt.plot(mod3_lats, mod3_var1_seas[idx,:] , color='green', linestyle='dashed',linewidth=1, label = 'fv0.9x1.25 '+var1name)
    plt.plot(mod4_lats, mod4_var1_seas[idx,:] , color='blue', linestyle='dashed',linewidth=1, label = 'fv1.9x2.5 '+var1name)

    plt.plot(mod1_lats, mod1_var2_seas[idx,:] , color='red', linestyle='solid',linewidth=1, label = 'vrseasia '+var2name)
    plt.plot(mod2_lats, mod2_var2_seas[idx,:] , color='yellow', linestyle='solid',linewidth=1, label = 'ne30 '+var2name)
    plt.plot(mod3_lats, mod3_var2_seas[idx,:] , color='green', linestyle='solid',linewidth=1, label = 'fv0.9x1.25 '+var2name)
    plt.plot(mod4_lats, mod4_var2_seas[idx,:] , color='blue', linestyle='solid',linewidth=1, label = 'fv1.9x2.5 '+var2name)

    plt.legend(fontsize=7)
    plt.xlabel('Latitude', fontsize=10)
    plt.ylabel('Water path (g/m^2)', fontsize=10)
    plt.title(str(iniyear)+'-'+str(endyear)+' '+season_index[idx]+' LWP&IWP Resolustion Comparison')

    plt.savefig(outdir+'vrseasia_LWPandIWP_modres_line_season_'+str(idx+1)+'.pdf')

plt.clf()
for idx in np.arange(5):
    plt.figure(idx)
    plt.clf()

    plt.plot(mod4_lats, mod1_var1_reg_seas[idx,:] , color='red', linestyle='dashed',linewidth=1, label = 'vrseasia')
    plt.plot(mod4_lats, mod2_var1_reg_seas[idx,:] , color='yellow', linestyle='dashed',linewidth=1, label = 'ne30')
    plt.plot(mod4_lats, mod3_var1_reg_seas[idx,:] , color='green', linestyle='dashed',linewidth=1, label = 'fv0.9x1.25')

    plt.legend(fontsize=7)
    plt.xlabel('Latitude', fontsize=10)
    plt.ylabel('Water path (g/m^2)', fontsize=10)
    plt.title(str(iniyear)+'-'+str(endyear)+' '+season_index[idx]+' '+var1str+' difference vs fv1.9x2.5')

    plt.savefig(outdir+'vrseasia_'+var1name+'_modres_diff_line_season_Reffv2x2_'+str(idx+1)+'.pdf')

plt.clf()
for idx in np.arange(5):
    plt.figure(idx)
    plt.clf()

    plt.plot(mod4_lats, mod1_var2_reg_seas[idx,:] , color='red', linestyle='solid',linewidth=1, label = 'vrseasia')
    plt.plot(mod4_lats, mod2_var2_reg_seas[idx,:] , color='yellow', linestyle='solid',linewidth=1, label = 'ne30')
    plt.plot(mod4_lats, mod3_var2_reg_seas[idx,:] , color='green', linestyle='solid',linewidth=1, label = 'fv0.9x1.25')

    plt.legend(fontsize=7)
    plt.xlabel('Latitude', fontsize=10)
    plt.ylabel('Water path (g/m^2)', fontsize=10)
    plt.title(str(iniyear)+'-'+str(endyear)+' '+season_index[idx]+' '+var2str+' difference vs fv1.9x2.5')

    plt.savefig(outdir+'vrseasia_'+var2name+'_modres_diff_line_season_Reffv2x2_'+str(idx+1)+'.pdf')

plt.clf()
for idx in np.arange(5):
    plt.figure(idx)
    plt.clf()

    plt.plot(mod4_lats, mod1_var1_reg_seas[idx,:] , color='red', linestyle='dashed',linewidth=1, label = 'vrseasia '+var1name)
    plt.plot(mod4_lats, mod2_var1_reg_seas[idx,:] , color='yellow', linestyle='dashed',linewidth=1, label = 'ne30 '+var1name)
    plt.plot(mod4_lats, mod3_var1_reg_seas[idx,:] , color='green', linestyle='dashed',linewidth=1, label = 'fv0.9x1.25 '+var1name)

    plt.plot(mod4_lats, mod1_var2_reg_seas[idx,:] , color='red', linestyle='solid',linewidth=1, label = 'vrseasia '+var2name)
    plt.plot(mod4_lats, mod2_var2_reg_seas[idx,:] , color='yellow', linestyle='solid',linewidth=1, label = 'ne30 '+var2name)
    plt.plot(mod4_lats, mod3_var2_reg_seas[idx,:] , color='green', linestyle='solid',linewidth=1, label = 'fv0.9x1.25 '+var2name)

    plt.legend(fontsize=7)
    plt.xlabel('Latitude', fontsize=10)
    plt.ylabel('Water path (g/m^2)', fontsize=10)
    plt.title(str(iniyear)+'-'+str(endyear)+' '+season_index[idx]+' LWP&IWP difference vs fv1.9x2.5')

    plt.savefig(outdir+'vrseasia_LWPandIWP_modres_diff_line_season_REFfv2x2_'+str(idx+1)+'.pdf')

