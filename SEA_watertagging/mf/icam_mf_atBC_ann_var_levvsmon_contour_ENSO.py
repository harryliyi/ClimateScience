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
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/mf_atBC/'

# set up variable names and file name
varname = 'mf'
var_longname = 'Moisture Fluxes'
varstr = "MF_atBC"
var_unit = r'$g/kg\times m/s$'


# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [10, 20]
lonbounds = [100, 110]

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

# define ENSO yyears
years_elweak = [1980, 1983, 1987, 1988, 1992, 1995, 1998, 2003, 2005]
years_elmod = [1983, 1987, 1988, 1992, 1998, 2003]
years_laweak = [1984, 1985, 1989, 1996, 1999, 2000, 2001]
years_lamod = [1989, 1999, 2000]

years_elweakpre = [iyear-1 for iyear in years_elweak]
years_laweakpre = [iyear-1 for iyear in years_laweak]

# contants
oro_water = 997  # kg/m^3
g = 9.8  # m/s^2
r_earth = 6371000  # m

############################################################################
# define functions
############################################################################


def plot_contour(plot_data, levs, months, colormap, clevs, varname, var_unit, legends, title, fname):

    plt.clf()
    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)

    for idx in range(3):
        ax = fig.add_subplot(3, 1, idx+1)
        ax.set_title(legends[idx], fontsize=6)
        cs = ax.contourf(months, levs, plot_data[idx], levels=clevs[idx], cmap=colormap[idx], extend='both')

        # set x/y tick label size
        if idx == 2:
            ax.set_xticks(months)
            ax.set_xticklabels(monnames)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel('Months', fontsize=5)
        else:
            ax.set_xticks([])

        ax.set_yticks(plevs)
        ax.set_yticklabels(plevs)
        ax.yaxis.set_tick_params(labelsize=5)
        plt.gca().invert_yaxis()

        ax.set_ylabel('Pressure(hPa)', fontsize=5, labelpad=0.5)

        cbar = fig.colorbar(cs, shrink=0.95, pad=0.01, orientation='vertical', ax=ax)
        cbar.set_label(varname+' ['+var_unit+']', fontsize=5, labelpad=0.6)
        cbar.set_ticks(clevs[idx])
        cbar.set_ticklabels(clevs[idx])
        cbar.ax.tick_params(labelsize=4)

    # add title
    plt.savefig(fname+'.png', bbox_inches='tight', dpi=600)
    plt.suptitle(title, fontsize=7, y=0.95)

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


############################################################################
# calculate the lev vs months
############################################################################

############################################################################
# convert q to g/kg and calculate the zonal moisture flux
model_var = model_q * model_u * 1000
legends = ['a) Under El Nino events', 'b) Under La Nina events', 'c) Differences (El Nino - La Nina)']

print(model_var[0, :, :])

# at left boundary
model_var_bc = np.mean(model_var[:, :, model_latl:model_latu+1, model_lonl], axis=2)
model_var_bc_mons_el = np.zeros((12, nlevs))
model_var_bc_mons_la = np.zeros((12, nlevs))
model_ps_bc = np.mean(model_ps[:, model_latl:model_latu+1, model_lonl], axis=1)
model_ps_bc_mons_el = np.zeros((12))
model_ps_bc_mons_la = np.zeros((12))
for idx in range(12):
    select_el = np.in1d(model_time.year, years_elweak) & (model_time.month == months[idx])
    select_la = np.in1d(model_time.year, years_laweak) & (model_time.month == months[idx])
    time_temp = model_time[select_el]
    # print(time_temp)
    model_var_bc_mons_el[idx, :] = np.mean(model_var_bc[select_el, :], axis=0)
    model_ps_bc_mons_el[idx] = np.mean(model_ps_bc[select_el], axis=0)

    model_var_bc_mons_la[idx, :] = np.mean(model_var_bc[select_la, :], axis=0)
    model_ps_bc_mons_la[idx] = np.mean(model_ps_bc[select_la], axis=0)

model_var_bc_mons_el[:, model_levs > np.amin(model_ps_bc_mons_el)] = np.nan
model_var_bc_mons_la[:, model_levs > np.amin(model_ps_bc_mons_la)] = np.nan


model_var_bc_mons_el = model_var_bc_mons_el[:, levtop:]
model_var_bc_mons_la = model_var_bc_mons_la[:, levtop:]
model_var_bc_mons_diff = model_var_bc_mons_el - model_var_bc_mons_la

plot_data = [np.transpose(model_var_bc_mons_el), np.transpose(
    model_var_bc_mons_la), np.transpose(model_var_bc_mons_diff)]
print(model_var_bc_mons_el[0, :])

clevs = [np.arange(-80, 90, 10), np.arange(-80, 90, 10), np.arange(-12, 14, 2)]
colormap = [cm.BrBG, cm.BrBG, cm.RdBu_r]

title = ' icam monthly averaged '+var_longname+' at left boundary in ENSO events'
fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_ENSOdiff_left'

plot_contour(plot_data, model_levs[levtop:], months, colormap, clevs,
             var_longname, var_unit, legends, title, outdir+fname)

# at right boundary
model_var_bc = np.mean(model_var[:, :, model_latl:model_latu+1, model_lonr], axis=2)
model_var_bc = model_var_bc * -1
model_var_bc_mons_el = np.zeros((12, nlevs))
model_var_bc_mons_la = np.zeros((12, nlevs))
model_ps_bc = np.mean(model_ps[:, model_latl:model_latu+1, model_lonl], axis=1)
model_ps_bc_mons_el = np.zeros((12))
model_ps_bc_mons_la = np.zeros((12))
for idx in range(12):
    select_el = np.in1d(model_time.year, years_elweak) & (model_time.month == months[idx])
    select_la = np.in1d(model_time.year, years_laweak) & (model_time.month == months[idx])
    time_temp = model_time[select_el]
    # print(time_temp)
    model_var_bc_mons_el[idx, :] = np.mean(model_var_bc[select_el, :], axis=0)
    model_ps_bc_mons_el[idx] = np.mean(model_ps_bc[select_el], axis=0)

    model_var_bc_mons_la[idx, :] = np.mean(model_var_bc[select_la, :], axis=0)
    model_ps_bc_mons_la[idx] = np.mean(model_ps_bc[select_la], axis=0)

model_var_bc_mons_el[:, model_levs > np.amin(model_ps_bc_mons_el)] = np.nan
model_var_bc_mons_la[:, model_levs > np.amin(model_ps_bc_mons_la)] = np.nan


model_var_bc_mons_el = model_var_bc_mons_el[:, levtop:]
model_var_bc_mons_la = model_var_bc_mons_la[:, levtop:]
model_var_bc_mons_diff = model_var_bc_mons_el - model_var_bc_mons_la

plot_data = [np.transpose(model_var_bc_mons_el), np.transpose(
    model_var_bc_mons_la), np.transpose(model_var_bc_mons_diff)]
print(model_var_bc_mons_el[0, :])

clevs = [np.arange(-80, 90, 10), np.arange(-80, 90, 10), np.arange(-12, 14, 2)]
colormap = [cm.BrBG, cm.BrBG, cm.RdBu_r]

title = ' icam monthly averaged '+var_longname+' at right boundary in ENSO events'
fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_ENSOdiff_right'

plot_contour(plot_data, model_levs[levtop:], months, colormap, clevs,
             var_longname, var_unit, legends, title, outdir+fname)


############################################################################
# convert q to g/kg and calculate the meridional moisture flux
model_var = model_q * model_v * 1000

# at bottom boundary
model_var_bc = np.mean(model_var[:, :, model_latl, model_lonl: model_lonr + 1], axis=2)
model_var_bc_mons_el = np.zeros((12, nlevs))
model_var_bc_mons_la = np.zeros((12, nlevs))
model_ps_bc = np.mean(model_ps[:, model_latl, model_lonl: model_lonr + 1], axis=1)
model_ps_bc_mons_el = np.zeros((12))
model_ps_bc_mons_la = np.zeros((12))
for idx in range(12):
    select_el = np.in1d(model_time.year, years_elweak) & (model_time.month == months[idx])
    select_la = np.in1d(model_time.year, years_laweak) & (model_time.month == months[idx])
    time_temp = model_time[select_el]
    # print(time_temp)
    model_var_bc_mons_el[idx, :] = np.mean(model_var_bc[select_el, :], axis=0)
    model_ps_bc_mons_el[idx] = np.mean(model_ps_bc[select_el], axis=0)

    model_var_bc_mons_la[idx, :] = np.mean(model_var_bc[select_la, :], axis=0)
    model_ps_bc_mons_la[idx] = np.mean(model_ps_bc[select_la], axis=0)

model_var_bc_mons_el[:, model_levs > np.amin(model_ps_bc_mons_el)] = np.nan
model_var_bc_mons_la[:, model_levs > np.amin(model_ps_bc_mons_la)] = np.nan


model_var_bc_mons_el = model_var_bc_mons_el[:, levtop:]
model_var_bc_mons_la = model_var_bc_mons_la[:, levtop:]
model_var_bc_mons_diff = model_var_bc_mons_el - model_var_bc_mons_la

plot_data = [np.transpose(model_var_bc_mons_el), np.transpose(
    model_var_bc_mons_la), np.transpose(model_var_bc_mons_diff)]
print(model_var_bc_mons_el[0, :])

clevs = [np.arange(-80, 90, 10), np.arange(-80, 90, 10), np.arange(-12, 14, 2)]
colormap = [cm.BrBG, cm.BrBG, cm.RdBu_r]

title = ' icam monthly averaged '+var_longname+' at bottom boundary in ENSO events'
fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_ENSOdiff_bottom'

plot_contour(plot_data, model_levs[levtop:], months, colormap, clevs,
             var_longname, var_unit, legends, title, outdir+fname)

# at top boundary
model_var_bc = np.mean(model_var[:, :, model_latu, model_lonl: model_lonr + 1], axis=2)
model_var_bc_mons_el = np.zeros((12, nlevs))
model_var_bc_mons_la = np.zeros((12, nlevs))
model_ps_bc = np.mean(model_ps[:, model_latu, model_lonl: model_lonr + 1], axis=1)
model_ps_bc_mons_el = np.zeros((12))
model_ps_bc_mons_la = np.zeros((12))
for idx in range(12):
    select_el = np.in1d(model_time.year, years_elweak) & (model_time.month == months[idx])
    select_la = np.in1d(model_time.year, years_laweak) & (model_time.month == months[idx])
    time_temp = model_time[select_el]
    # print(time_temp)
    model_var_bc_mons_el[idx, :] = np.mean(model_var_bc[select_el, :], axis=0)
    model_ps_bc_mons_el[idx] = np.mean(model_ps_bc[select_el], axis=0)

    model_var_bc_mons_la[idx, :] = np.mean(model_var_bc[select_la, :], axis=0)
    model_ps_bc_mons_la[idx] = np.mean(model_ps_bc[select_la], axis=0)

model_var_bc_mons_el[:, model_levs > np.amin(model_ps_bc_mons_el)] = np.nan
model_var_bc_mons_la[:, model_levs > np.amin(model_ps_bc_mons_la)] = np.nan


model_var_bc_mons_el = model_var_bc_mons_el[:, levtop:]
model_var_bc_mons_la = model_var_bc_mons_la[:, levtop:]
model_var_bc_mons_diff = model_var_bc_mons_el - model_var_bc_mons_la

plot_data = [np.transpose(model_var_bc_mons_el), np.transpose(
    model_var_bc_mons_la), np.transpose(model_var_bc_mons_diff)]
print(model_var_bc_mons_el[0, :])

clevs = [np.arange(-80, 90, 10), np.arange(-80, 90, 10), np.arange(-12, 14, 2)]
colormap = [cm.BrBG, cm.BrBG, cm.RdBu_r]

title = ' icam monthly averaged '+var_longname+' at top boundary in ENSO events'
fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_ENSOdiff_top'

plot_contour(plot_data, model_levs[levtop:], months, colormap, clevs,
             var_longname, var_unit, legends, title, outdir+fname)


#
# print(model_var.shape)
# print(model_latl)
# print(model_lonl)
# print(model_lonr)
#
#
# # at bottom boundary
# model_var_bc = np.mean(model_var[:, :, model_latl, model_lonl: model_lonr + 1], axis=2)
# model_var_bc_mons = np.zeros((12, nlevs))
# model_ps_bc = np.mean(model_ps[:, model_latl, model_lonl: model_lonr + 1], axis=1)
# model_ps_bc_mons = np.zeros((12))
# for idx in range(12):
#     select_time = (model_time.month == months[idx])
#     time_temp = model_time[select_time]
#     print(time_temp)
#     temp_var = np.mean(model_var_bc[select_time, :], axis=0)
#     model_var_bc_mons[idx, :] = temp_var
#     model_ps_bc_mons[idx] = np.mean(model_ps_bc[select_time], axis=0)
#
# # calculate the net flux
# model_var_bc_mons_totnet = model_var_bc_mons_totnet + model_var_bc_mons
# model_var_bc_mons_meridnet = model_var_bc_mons
#
# model_var_bc_mons[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
#
# plot_data = np.transpose(model_var_bc_mons)
# print(model_var_bc_mons[0, :])
#
# clevs = np.arange(-85, 90, 5)
# colormap = cm.RdBu_r
#
# title = ' icam monthly averaged '+var_longname+' at bottom boundary'
# fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_bottom'
#
# plot_contour(plot_data[levtop:, :], model_levs[levtop:], months, colormap, clevs, var_longname, var_unit, title, outdir+fname)
#
# # at top boundary
# model_var_bc = np.mean(model_var[:, :, model_latu, model_lonl: model_lonr + 1], axis=2)
# model_var_bc = model_var_bc * -1
# model_var_bc_mons = np.zeros((12, nlevs))
# model_ps_bc = np.mean(model_ps[:, model_latu, model_lonl: model_lonr + 1], axis=1)
# model_ps_bc_mons = np.zeros((12))
# for idx in range(12):
#     select_time = (model_time.month == months[idx])
#     time_temp = model_time[select_time]
#     print(time_temp)
#     temp_var = np.mean(model_var_bc[select_time, :], axis=0)
#     model_var_bc_mons[idx, :] = temp_var
#     model_ps_bc_mons[idx] = np.mean(model_ps_bc[select_time], axis=0)
#
# # calculate the net flux
# model_var_bc_mons_totnet = model_var_bc_mons_totnet + model_var_bc_mons
# model_var_bc_mons_meridnet = model_var_bc_mons_meridnet + model_var_bc_mons
#
# model_var_bc_mons[:, model_levs > np.amin(model_ps_bc_mons)] = np.nan
#
# plot_data = np.transpose(model_var_bc_mons)
# print(model_var_bc_mons[0, :])
#
# clevs = np.arange(-85, 90, 5)
# colormap = cm.RdBu_r
#
# title = ' icam monthly averaged '+var_longname+' at top boundary'
# fname = 'icam5_' + varstr + '_SEA_monthly_mean_contour_top'
#
# plot_contour(plot_data[levtop:, :], model_levs[levtop:], months, colormap, clevs, var_longname, var_unit, title, outdir+fname)
