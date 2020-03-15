# This script is used to read climatological data from Aerosol AMIP runs and investigate the aerosol impact

# S1-plot climatological data
# S2-plot contours
#
# by Harry Li

# import libraries
import pathmagic  # noqa: F401
from modules.datareader.mod_dataread_cesm import readcesm, readcesm_3D

import matplotlib.cm as cm
import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = "/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/"


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/mf_atBC/'

# set up variable names and file name
varname = 'MFC'
varfname = "mfc"
var_longname = "Moisture Flux Convergence"
var_res = "fv19"
var_unit = 'mm/day'

# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-40, 60]
lonbounds = [40, 180]

# latbounds = [ -40 , 40 ]
# lonbounds = [ 10 , 160 ]

# define top layer
ptop = 200
plevs = [200, 300, 400, 500, 700, 850, 925]

# contants
oro_water = 997
g = 9.8
r_earth = 6371000

# define Southeast region
reg_lats = [10, 20]
reg_lons = [100, 110]

# set data frequency
frequency = 'mon'

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

################################################################################################
# S0-Define functions
################################################################################################


# calculate layer thickness
def dpres_plevel(levs, ps, ptop):
    # calculate thickness
    dlevs = np.gradient(levs)
    dlevs[0] = dlevs[0]/2
    dlevs[-1] = dlevs[-1]/2

    # get dimensions
    ntime = ps.shape[0]
    nlev = len(levs)
    nlat = ps.shape[1]
    nlon = ps.shape[2]
    levli = np.abs(levs - ptop).argmin()

    layer_thickness = np.zeros((ntime, nlev, nlat, nlon))
    for ilev in range(levli, nlev, 1):
        temp = np.zeros((ntime, nlat, nlon))
        temp[ps > levs[ilev]] = dlevs[ilev]
        layer_thickness[:, ilev, :, :] = temp

    layer_thickness = layer_thickness * 100  # convert from hPa to Pa
    layer_thickness[layer_thickness == 0] = np.nan

    return layer_thickness

# calculat divergence


def getdiv_tot_test(lats, lons, u, v):
    dlats = lats
    dlons = lons
#    print(dlats)

    diverge = np.gradient(u, dlons, axis=1)/np.pi*180/r_earth + np.gradient(v, dlats, axis=0)/np.pi*180/r_earth

    return diverge


def getdiv_tot(lats, lons, u, v):
    dlats = lats
    dlons = lons
    dtemp = np.zeros((u.shape[0], u.shape[1], u.shape[2], u.shape[3]))

    for ilat in range(len(dlats)):
        dtemp[:, :, ilat, :] = v[:, :, ilat, :]/r_earth * np.tan(np.deg2rad(dlats[ilat]))

    diverge = np.gradient(u, dlons, axis=3)/np.pi*180/r_earth + np.gradient(v, dlats, axis=2)/np.pi*180/r_earth - dtemp

    return diverge


def getdiv_u(lats, lons, u, v):
    dlons = lons

    diverge = np.gradient(u, dlons, axis=3)/np.pi*180/r_earth

    return diverge


def getdiv_v(lats, lons, u, v):
    dlats = lats

    dtemp = np.zeros((u.shape[0], u.shape[1], u.shape[2], u.shape[3]))
    for ilat in range(len(dlats)):
        dtemp[:, :, ilat, :] = v[:, :, ilat, :]/r_earth * np.tan(np.deg2rad(dlats[ilat]))

    diverge = np.gradient(v, dlats, axis=2)/np.pi*180/r_earth - dtemp

    return diverge


# calculate moisture flux convergence


def getmfc(lats, lons, levs, u, v, q, ps, ptop):

    qu = q * u
    qv = q * v

#    print('calculating thickness...')
    layer_thickness = dpres_plevel(levs, ps, ptop)
    print(ps[5, lattest, lontest])
    print(levs)
    print(layer_thickness[5, :, lattest, lontest])

#    print(qu[5,:,lattest,lontest])
#    print(qu_int[5,lattest,lontest])
#    print(sum(qu[5,:,lattest,lontest]*layer_thickness[5,:,lattest,lontest]))

    ntime = qu.shape[0]
    nlev = qu.shape[1]
    nlat = qu.shape[2]
    nlon = qu.shape[3]

    div_tot = np.zeros((ntime, nlev, nlat, nlon))
    div_u = np.zeros((ntime, nlev, nlat, nlon))
    div_v = np.zeros((ntime, nlev, nlat, nlon))

    # for itime in range(ntime):
    #     for ilev in range(nlev):
    #         div_tot_test[itime, ilev, :, :] = getdiv_tot_test(lats, lons, qu[itime, ilev, :, :], qv[itime, ilev, :, :])
    #         div_u[itime, ilev, :, :] = getdiv_u(lats, lons, qu[itime, ilev, :, :], qv[itime, ilev, :, :])
    #         div_v[itime, ilev, :, :] = getdiv_v(lats, lons, qu[itime, ilev, :, :], qv[itime, ilev, :, :])

    div_tot = getdiv_tot(lats, lons, qu, qv)
    div_u = getdiv_u(lats, lons, qu, qv)
    div_v = getdiv_v(lats, lons, qu, qv)
    # print(np.sum(div_tot_test-div_tot))

    div_tot = div_tot*layer_thickness/oro_water/g
    div_u = div_u*layer_thickness/oro_water/g
    div_v = div_v*layer_thickness/oro_water/g

    return -div_tot, -div_u, -div_v


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
        ticks = clevs[idx]
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=4)

    # add title
    plt.savefig(fname+'.png', bbox_inches='tight', dpi=600)
    plt.suptitle(title, fontsize=7, y=0.95)

    # save figure
    plt.savefig(fname+'.pdf', bbox_inches='tight', dpi=600)
    plt.close(fig)


################################################################################################
# S1-read climatological data
################################################################################################
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


print('finished reading...')

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))
lattest = np.argmin(np.abs(model_lats - 5))
lontest = np.argmin(np.abs(model_lons - 98))

levtop = np.abs(model_levs - ptop).argmin()
nlevs = len(model_levs)
print(model_lonl)
print(model_lonr)

print('calculating mfc...')
# calculate mfc
mfc_tot, mfc_u, mfc_v = getmfc(model_lats, model_lons, model_levs, model_u, model_v, model_q, model_ps, ptop)

# convert to mm/day
mfc_tot = mfc_tot * 86400 * 1000
mfc_u = mfc_u * 86400 * 1000
mfc_v = mfc_v * 86400 * 1000

print(np.sum(mfc_tot - (mfc_u+mfc_v)))

#################################################################################################
# plot climatology
#################################################################################################
print('plotting climatology...')

legends = ['a) Under El Nino events', 'b) Under La Nina events', 'c) Differences (El Nino - La Nina)']

mfc_tot_mons_el = np.zeros((12, nlevs))
mfc_u_mons_el = np.zeros((12, nlevs))
mfc_v_mons_el = np.zeros((12, nlevs))

mfc_tot_mons_la = np.zeros((12, nlevs))
mfc_u_mons_la = np.zeros((12, nlevs))
mfc_v_mons_la = np.zeros((12, nlevs))

for idx in range(12):
    select_el = np.in1d(model_time.year, years_elweak) & (model_time.month == months[idx])
    select_la = np.in1d(model_time.year, years_laweak) & (model_time.month == months[idx])
    time_temp = model_time[select_el]
    # print(time_temp)

    temp = np.nanmean(np.nanmean(mfc_tot[:, :, model_latl:model_latu+1, model_lonl:model_lonr+1], axis=3), axis=2)
    mfc_tot_mons_el[idx, :] = np.nanmean(temp[select_el, :], axis=0)
    mfc_tot_mons_la[idx, :] = np.nanmean(temp[select_la, :], axis=0)

    temp = np.nanmean(np.nanmean(mfc_u[:, :, model_latl:model_latu+1, model_lonl:model_lonr+1], axis=3), axis=2)
    mfc_u_mons_el[idx, :] = np.nanmean(temp[select_el, :], axis=0)
    mfc_u_mons_la[idx, :] = np.nanmean(temp[select_la, :], axis=0)

    temp = np.nanmean(np.nanmean(mfc_v[:, :, model_latl:model_latu+1, model_lonl:model_lonr+1], axis=3), axis=2)
    mfc_v_mons_el[idx, :] = np.nanmean(temp[select_el, :], axis=0)
    mfc_v_mons_la[idx, :] = np.nanmean(temp[select_la, :], axis=0)


# plot total convergence
mfc_tot_mons_el = mfc_tot_mons_el[:, levtop:]
mfc_tot_mons_la = mfc_tot_mons_la[:, levtop:]
mfc_tot_mons_diff = mfc_tot_mons_el - mfc_tot_mons_la
plot_data = [np.transpose(mfc_tot_mons_el), np.transpose(mfc_tot_mons_la), np.transpose(mfc_tot_mons_diff)]
# print(mfc_tot_mons[6, :])
# print(np.nansum(mfc_tot_mons[6, :]))
print(np.nansum(mfc_tot_mons_el[:, :], axis=1))

clevs = [np.arange(-2, 2.4, 0.4), np.arange(-2, 2.4, 0.4), np.arange(-0.6, 0.7, 0.1)]
colormap = [cm.BrBG, cm.BrBG, cm.RdBu_r]

title = ' icam monthly averaged total moisture flux convergence in ENSO events'
fname = 'icam5_mfc_SEA_monthly_mean_levvsmons_ENSOdiff_total'
plot_contour(plot_data, model_levs[levtop:], months, colormap, clevs,
             var_longname, var_unit, legends, title, outdir+fname)

# plot zonal convergence
mfc_u_mons_el = mfc_u_mons_el[:, levtop:]
mfc_u_mons_la = mfc_u_mons_la[:, levtop:]
mfc_u_mons_diff = mfc_u_mons_el - mfc_u_mons_la
plot_data = [np.transpose(mfc_u_mons_el), np.transpose(mfc_u_mons_la), np.transpose(mfc_u_mons_diff)]
# print(mfc_tot_mons[6, :])
# print(np.nansum(mfc_tot_mons[6, :]))
print(np.nansum(mfc_u_mons_el[:, :], axis=1))

clevs = [np.arange(-2, 2.4, 0.4), np.arange(-2, 2.4, 0.4), np.arange(-0.6, 0.7, 0.1)]
colormap = [cm.BrBG, cm.BrBG, cm.RdBu_r]

title = ' icam monthly averaged zonal moisture flux convergence in ENSO events'
fname = 'icam5_mfc_SEA_monthly_mean_levvsmons_ENSOdiff_zonal'
plot_contour(plot_data, model_levs[levtop:], months, colormap, clevs,
             var_longname, var_unit, legends, title, outdir+fname)

# plot meridional convergence
mfc_v_mons_el = mfc_v_mons_el[:, levtop:]
mfc_v_mons_la = mfc_v_mons_la[:, levtop:]
mfc_v_mons_diff = mfc_v_mons_el - mfc_v_mons_la
plot_data = [np.transpose(mfc_v_mons_el), np.transpose(mfc_v_mons_la), np.transpose(mfc_v_mons_diff)]
# print(mfc_tot_mons[6, :])
# print(np.nansum(mfc_tot_mons[6, :]))
print(np.nansum(mfc_v_mons_el[:, :], axis=1))

clevs = [np.arange(-2, 2.4, 0.4), np.arange(-2, 2.4, 0.4), np.arange(-0.6, 0.7, 0.1)]
colormap = [cm.BrBG, cm.BrBG, cm.RdBu_r]

title = ' icam monthly averaged meridional moisture flux convergence in ENSO events'
fname = 'icam5_mfc_SEA_monthly_mean_levvsmons_ENSOdiff_meridional'
plot_contour(plot_data, model_levs[levtop:], months, colormap, clevs,
             var_longname, var_unit, legends, title, outdir+fname)


# # plot zonal convergence
# plot_data = np.transpose(mfc_u_mons)
# # print(mfc_u_mons[6, :])
# # print(np.nansum(mfc_u_mons[6, :]))
# print(np.nansum(mfc_u_mons[:, :], axis=1))
#
# clevs = np.arange(-2, 2.2, 0.2)
# colormap = cm.RdBu_r
#
# title = ' icam monthly averaged zonal moisture flux convergence'
# fname = 'icam5_mfc_SEA_monthly_mean_levvsmons_zonal'
# plot_contour(plot_data[levtop:, :], model_levs[levtop:], months, colormap, clevs, 'Moisture Flux Convergence', 'mm/day', title, outdir+fname)
#
# # plot meridional convergence
# plot_data = np.transpose(mfc_v_mons)
# # print(mfc_v_mons[6, :])
# # print(np.nansum(mfc_v_mons[6, :]))
# print(np.nansum(mfc_v_mons[:, :], axis=1))
#
# clevs = np.arange(-2, 2.2, 0.2)
# colormap = cm.RdBu_r
#
# title = ' icam monthly averaged meridional moisture flux convergence'
# fname = 'icam5_mfc_SEA_monthly_mean_levvsmons_meridional'
# plot_contour(plot_data[levtop:, :], model_levs[levtop:], months, colormap, clevs, 'Moisture Flux Convergence', 'mm/day', title, outdir+fname)
