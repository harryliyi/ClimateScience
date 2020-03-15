# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_cesm import readcesm, readcesm_3D
from modules.plot.mod_plt_contour import plot_2Dcontour


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')


# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = "/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/"


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/climatology/mf_atBC/'

# set up variable names and file name
varname = 'mf'
var_longname = 'Moisture Fluxes'
varstr = "MF_atBC"
var_unit = r'$g/kg\times m/s$'


# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [0, 30]
lonbounds = [90, 120]

# define Southeast region
reg_lats = [10, 20]
reg_lons = [100, 110]
reg_name = 'mainSEA'

# set data frequency
frequency = 'mon'

# define pressure level
ptop = 200
plevel = 500
plevs = [200, 300, 400, 500, 700, 850, 925]

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# contants
oro_water = 997  # kg/m^3
g = 9.8  # m/s^2
r_earth = 6371000  # m

############################################################################
# define functions
############################################################################


def plot_contour(plot_data, levs, months, colormap, clevs, varname, var_unit, title, fname):

    plt.clf()
    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('(+ flux into region)                                                                                                                       ', fontsize=8)
    cs = ax.contourf(months, levs, plot_data, levels=clevs, cmap=cm.RdBu_r, extend='both')

    # set x/y tick label size
    ax.set_xticks(months)
    ax.set_xticklabels(monnames)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.set_yticks(plevs)
    ax.set_yticklabels(plevs)
    ax.yaxis.set_tick_params(labelsize=7)
    plt.gca().invert_yaxis()

    # ax.set_xlabel('Months', fontsize=5)
    ax.set_ylabel('Pressure(hPa)', fontsize=8, labelpad=0.7)

    fig.subplots_adjust(bottom=0.23, wspace=0.2, hspace=0.2)
    cbar_ax = fig.add_axes([0.15, 0.17, 0.7, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(varname+' ['+var_unit+']', fontsize=8, labelpad=-0.7)

    # add title
    plt.savefig(fname+'.png', bbox_inches='tight', dpi=600)
    plt.suptitle(title, fontsize=8, y=0.95)

    # save figure
    plt.savefig(fname+'.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)


############################################################################
# read data
############################################################################

# read vrcesm
print('Reading CESM data...')

# read PS
varname = 'PS'

resolution = 'fv19'
varfname = 'PS'
model_ps, model_time,  model_lats, model_lons = readcesm(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)


model_ps = model_ps/100

varname = 'U'
varfname = 'U'
model_u, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

varname = 'V'
varfname = 'V'
model_v, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

varname = 'Q'
varfname = 'Q'
model_q, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

print(model_time)
print(model_q.shape)
print(model_levs)
nlevs = len(model_levs)

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))
levtop = np.abs(model_levs - ptop).argmin()
print(model_lonl)
print(model_lonr)

select_time = (model_time.month == 10)
print(model_time[select_time])
print(model_levs[20])
print(model_v[select_time, 20, model_latu, model_lonl: model_lonr + 1])


############################################################################
# calculate the lev vs months
############################################################################

############################################################################
# convert q from kg/kg to g/kg and calculate the zonal moisture flux
model_var = model_q * model_u * 1000

# print(model_var[0, :, :])

# at left boundary
model_var_bc = np.mean(model_var[:, :, model_latl:model_latu+1, model_lonl-1], axis=2)
model_var_bc_mons = np.zeros((12, nlevs))
model_ps_bc = np.mean(model_ps[:, model_latl:model_latu+1, model_lonl-1], axis=1)
model_ps_bc_mons = np.zeros((12))
for idx in range(12):
    select_time = (model_time.month == months[idx])
    time_temp = model_time[select_time]
    # print(time_temp)
    temp_var = np.mean(model_var_bc[select_time, :], axis=0)
    model_var_bc_mons[idx, :] = temp_var
    model_ps_bc_mons[idx] = np.mean(model_ps_bc[select_time], axis=0)

# calculate the net flux
model_var_bc_mons_totnet = model_var_bc_mons.copy()
model_var_bc_mons_zonalnet = model_var_bc_mons.copy()

# mask the lev below the ps
model_var_bc_mons[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
model_var_bc_mons_totnet[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
model_var_bc_mons_zonalnet[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan

plot_data = np.transpose(model_var_bc_mons)
# print(model_var_bc_mons[0, :])

clevs = np.arange(-85, 90, 5)
colormap = cm.RdBu_r

title = ' icam monthly averaged '+var_longname+' at left boundary'
fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_left'

plot_contour(plot_data[levtop:, :], model_levs[levtop:], months,
             colormap, clevs, var_longname, var_unit, title, outdir+fname)

# at right boundary
model_var_bc = np.mean(model_var[:, :, model_latl:model_latu+1, model_lonr+1], axis=2)
model_var_bc = model_var_bc * -1
model_var_bc_mons = np.zeros((12, nlevs))
model_ps_bc = np.mean(model_ps[:, model_latl:model_latu+1, model_lonr+1], axis=1)
model_ps_bc_mons = np.zeros((12))
for idx in range(12):
    select_time = (model_time.month == months[idx])
    time_temp = model_time[select_time]
    # print(time_temp)
    temp_var = np.mean(model_var_bc[select_time, :], axis=0)
    model_var_bc_mons[idx, :] = temp_var
    model_ps_bc_mons[idx] = np.mean(model_ps_bc[select_time], axis=0)

# calculate the net flux
model_var_bc_mons_totnet = model_var_bc_mons_totnet + model_var_bc_mons
model_var_bc_mons_zonalnet = model_var_bc_mons_zonalnet + model_var_bc_mons

# mask the lev below the ps
model_var_bc_mons[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
model_var_bc_mons_totnet[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
model_var_bc_mons_zonalnet[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan

plot_data = np.transpose(model_var_bc_mons)
# print(model_var_bc_mons[0, :])

clevs = np.arange(-85, 90, 5)
colormap = cm.RdBu_r

title = ' icam monthly averaged '+var_longname+' at right boundary'
fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_right'

plot_contour(plot_data[levtop:, :], model_levs[levtop:], months,
             colormap, clevs, var_longname, var_unit, title, outdir+fname)


############################################################################
# convert q from kg/kg to g/kg and calculate the meridional moisture flux
model_var = model_q * model_v * 1000

# print(model_var.shape)
# print(model_latl)
# print(model_lonl)
# print(model_lonr)


# at bottom boundary
model_var_bc = np.mean(model_var[:, :, model_latl-1, model_lonl: model_lonr + 1], axis=2)
# select_time = (model_time.month == 10)
# print(model_var[select_time, 20, model_latl, model_lonl: model_lonr + 1])

model_var_bc_mons = np.zeros((12, nlevs))
model_ps_bc = np.mean(model_ps[:, model_latl-1, model_lonl: model_lonr + 1], axis=1)
model_ps_bc_mons = np.zeros((12))
for idx in range(12):
    select_time = (model_time.month == months[idx])
    time_temp = model_time[select_time]
    # print(time_temp)
    temp_var = np.mean(model_var_bc[select_time, :], axis=0)
    model_var_bc_mons[idx, :] = temp_var
    model_ps_bc_mons[idx] = np.mean(model_ps_bc[select_time], axis=0)

# calculate the net flux
model_var_bc_mons_totnet = model_var_bc_mons_totnet + model_var_bc_mons
model_var_bc_mons_meridnet = model_var_bc_mons

# mask the lev below the ps
model_var_bc_mons[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
model_var_bc_mons_totnet[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
model_var_bc_mons_meridnet[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan


plot_data = np.transpose(model_var_bc_mons)
# print(model_var_bc_mons[0, :])

clevs = np.arange(-85, 90, 5)
colormap = cm.RdBu_r

title = ' icam monthly averaged '+var_longname+' at bottom boundary'
fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_bottom'

plot_contour(plot_data[levtop:, :], model_levs[levtop:], months,
             colormap, clevs, var_longname, var_unit, title, outdir+fname)

# at top boundary
model_var_bc = np.mean(model_var[:, :, model_latu+1, model_lonl: model_lonr + 1], axis=2)
model_var_bc = model_var_bc * -1
model_var_bc_mons = np.zeros((12, nlevs))
model_ps_bc = np.mean(model_ps[:, model_latu+1, model_lonl: model_lonr + 1], axis=1)
model_ps_bc_mons = np.zeros((12))
for idx in range(12):
    select_time = (model_time.month == months[idx])
    time_temp = model_time[select_time]
    # print(time_temp)
    temp_var = np.mean(model_var_bc[select_time, :], axis=0)
    model_var_bc_mons[idx, :] = temp_var
    model_ps_bc_mons[idx] = np.mean(model_ps_bc[select_time], axis=0)

# calculate the net flux
model_var_bc_mons_totnet = model_var_bc_mons_totnet + model_var_bc_mons
model_var_bc_mons_meridnet = model_var_bc_mons_meridnet + model_var_bc_mons

# mask the lev below the ps
model_var_bc_mons[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
model_var_bc_mons_totnet[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
model_var_bc_mons_meridnet[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan

plot_data = np.transpose(model_var_bc_mons)
# print(model_var_bc_mons[0, :])

clevs = np.arange(-85, 90, 5)
colormap = cm.RdBu_r

title = ' icam monthly averaged '+var_longname+' at top boundary'
fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_top'

plot_contour(plot_data[levtop:, :], model_levs[levtop:], months,
             colormap, clevs, var_longname, var_unit, title, outdir+fname)

############################################################################
# calculate the convergence
############################################################################

# convergence is calculated by:
# zonal: q * du * delta p / dx / g / oro_water
# meridional: q * dv * delta_p / dy / g / oro_water
dx = (model_lons[model_lonr] - model_lons[model_lonl])*np.pi/180*r_earth
dy = (model_lats[model_latu] - model_lats[model_latl])*np.pi/180*r_earth

delta_p = np.zeros(len(model_levs))
for idx in range(len(model_levs)):
    if idx == 0:
        delta_p[idx] = (model_levs[idx + 1] - model_levs[idx])/2
    elif (idx == len(model_levs)-1):
        delta_p[idx] = (model_levs[idx] - model_levs[idx - 1])/2
    else:
        delta_p[idx] = (model_levs[idx + 1] - model_levs[idx - 1])/2

print(delta_p)

model_var_conv_mons_totnet = np.zeros((12, nlevs))
model_var_conv_mons_zonalnet = np.zeros((12, nlevs))
model_var_conv_mons_meridnet = np.zeros((12, nlevs))
for idx in range(12):
    # # actual convergence m/s
    # model_var_conv_mons_zonalnet[idx, :] = model_var_bc_mons_zonalnet[idx, :] * (100 * delta_p)/g/dx/oro_water/1000
    # model_var_conv_mons_meridnet[idx, :] = model_var_bc_mons_meridnet[idx, :] * (100 * delta_p)/g/dy/oro_water/1000
    # convergence into the basin g/ms
    model_var_conv_mons_zonalnet[idx, :] = model_var_bc_mons_zonalnet[idx, :] * (100 * delta_p)/g
    model_var_conv_mons_meridnet[idx, :] = model_var_bc_mons_meridnet[idx, :] * (100 * delta_p)/g

model_var_conv_mons_totnet = model_var_conv_mons_zonalnet + model_var_conv_mons_meridnet

# # convert from m/s to mm/day
# model_var_conv_mons_totnet = model_var_conv_mons_totnet * 86400 * 1000
# model_var_conv_mons_zonalnet = model_var_conv_mons_zonalnet * 86400 * 1000
# model_var_conv_mons_meridnet = model_var_conv_mons_meridnet * 86400 * 1000

# convert from g/ms to 10^4 g/ms
model_var_conv_mons_totnet = model_var_conv_mons_totnet / 1000
model_var_conv_mons_zonalnet = model_var_conv_mons_zonalnet / 1000
model_var_conv_mons_meridnet = model_var_conv_mons_meridnet / 1000


# plot total convergence
plot_data = np.transpose(model_var_conv_mons_totnet)
print(np.nansum(model_var_conv_mons_totnet[:, :], axis=1))

clevs = np.arange(-20., 22., 2.)
# clevs = np.arange(-2, 2.2, 0.2)
# colormap = cm.RdBu_r

title = ' icam monthly averaged total moisture flux convergence'
fname = 'icam5_mfc_atBC_SEA_monthly_mean_levvsmons_total'

plot_contour(plot_data[levtop:, :], model_levs[levtop:], months, colormap,
             clevs, 'Moisture Flux Convergence', 'g/ms', title, outdir+fname)

# plot zonal convergence
plot_data = np.transpose(model_var_conv_mons_zonalnet)
print(np.nansum(model_var_conv_mons_zonalnet[:, :], axis=1))


# clevs = np.arange(-2, 2.2, 0.2)
# colormap = cm.RdBu_r

title = ' icam monthly averaged zonal moisture flux convergence'
fname = 'icam5_mfc_atBC_SEA_monthly_mean_levvsmons_zonal'

plot_contour(plot_data[levtop:, :], model_levs[levtop:], months, colormap,
             clevs, 'Moisture Flux Convergence', 'g/ms', title, outdir+fname)

# plot meridional convergence
plot_data = np.transpose(model_var_conv_mons_meridnet)
print(np.nansum(model_var_conv_mons_meridnet[:, :], axis=1))

# clevs = np.arange(-2, 2.2, 0.2)
# colormap = cm.RdBu_r

title = ' icam monthly averaged meridional moisture flux convergence'
fname = 'icam5_mfc_atBC_SEA_monthly_mean_levvsmons_meridional'

plot_contour(plot_data[levtop:, :], model_levs[levtop:], months, colormap,
             clevs, 'Moisture Flux Convergence', 'g/ms', title, outdir+fname)
