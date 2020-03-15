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
import pandas as pd

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.1
plt.switch_backend('agg')

# set up data directories and filenames
case = "SEA_wt_1920today"

expdir1 = '/scratch/d/dylan/harryli/gpcdata/fields_archive/icam5/"+case+"/atm/hist/'


# set up output directory and output log
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/omega/drywet/'

# set up variable names and file name
varname = 'OMEGA'
var_longname = r"$\omega$"
varstr = "omega"
var_unit = r'$\times 10^{-3} Pa/s$'

# define inital year and end year
iniyear = 1980
endyear = 2005
yearts = np.arange(iniyear, endyear+1)
nyears = len(yearts)

# define the contour plot region
latbounds = [-20, 40]
lonbounds = [80, 140]

vc_latbounds = [10, 22]
vc_lonbounds = [100, 110]

# latbounds = [-20, 40]
# lonbounds = [60, 150]

# define top layer
ptop = 200
plevel = 500
plevs = [200, 300, 400, 500, 700, 850, 925]

# contants
oro_water = 997
g = 9.8
r_earth = 6371000

# define Southeast region
reg_lats = [10, 20]
reg_lons = [100, 110]

vc_reg_lats = [16, 20]
vc_reg_lons = [106, 110]

# define Wet-Dry yyears
monsoon_period_str = 'JAS'
if monsoon_period_str == 'JJA':
    monsoon_period = [6, 7, 8]
    years_wet = [1987, 1993, 2002]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'JJAS':
    monsoon_period = [6, 7, 8, 9]
    years_wet = [1987, 1993, 2002]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'MJJAS':
    monsoon_period = [5, 6, 7, 8, 9]
    years_wet = [1987, 1993, 2000]
    years_dry = [1988, 1992, 1995, 1998, 1999, 2005]
if monsoon_period_str == 'JAS':
    monsoon_period = [7, 8, 9]
    years_wet = [1987, 1991, 1993, 2000]
    years_dry = [1988, 1990, 1995, 1998, 1999]
if monsoon_period_str == 'AS':
    monsoon_period = [8, 9]
    years_wet = [1987, 1991, 1996, 2000]
    years_dry = [1980, 1990, 1995, 1998, 2005]
if monsoon_period_str == 'Sep':
    monsoon_period = [9]
    years_wet = [1981, 1983, 1987, 1996, 2000]
    years_dry = [1980, 1990, 1995, 2005]
if monsoon_period_str == 'Oct':
    monsoon_period = [10]
    years_wet = [1981, 1986, 1989, 2001]
    years_dry = [1990, 1992, 1993, 1995, 2004]
if monsoon_period_str == 'May':
    monsoon_period = [5]
    years_wet = [1981, 1982, 1985, 1990, 2000]
    years_dry = [1983, 1987, 1988, 1992, 1998]
if monsoon_period_str == 'AM':
    monsoon_period = [4, 5]
    years_wet = [1981, 1985, 1996, 2000]
    years_dry = [1983, 1987, 1992, 1998, 2005]
if monsoon_period_str == 'MJ':
    monsoon_period = [5, 6]
    years_wet = [1981, 1985, 1997, 2001]
    years_dry = [1987, 1988, 1992, 2005]
outdir = outdir + monsoon_period_str+'/'

nyears_wet = len(years_wet)
nyears_dry = len(years_dry)

# set data frequency
frequency = 'day'

# month series
months = np.arange(1, 13, 1)
monnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

################################################################################################
# S0-Define functions
################################################################################################


def getvc(lats, lons, u, v):
    dlats = lats
    dlons = lons
    # dtemp = np.zeros((u.shape[0], u.shape[1], u.shape[2], u.shape[3]))
    #
    # for ilat in range(len(dlats)):
    #     dtemp[:, :, ilat, :] = v[:, :, ilat, :]/r_earth * np.tan(np.deg2rad(dlats[ilat]))

    vorticity = -1*np.gradient(u, dlats, axis=1)/np.pi*180/r_earth + np.gradient(v, dlons, axis=2)/np.pi*180/r_earth

    return vorticity


def cal_diff(var1, var2, std1, std2, n1, n2):
    res = var1-var2
    SE = np.sqrt((std1**2/n1) + (std2**2/n2))
    res_sig = res/SE
    res_sig = np.abs(res_sig)

    return res, res_sig


################################################################################################
# S1-read climatological data
################################################################################################
# read vrcesm
print('Reading CESM data...')
resolution = 'fv19'
# # read PS
# varname = 'PS'
#
# resolution = 'fv19'
# varfname = 'PS'
# model_ps, model_time,  model_lats, model_lons = readcesm(
#     varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)
# model_ps = model_ps/100

varname = 'U'
varfname = 'U'
model_u, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

print(model_u.shape)
# print(model_u[0, :, :])


varname = 'V'
varfname = 'V'
model_v, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)


varname = 'OMEGA'
varfname = 'OMEGA'
model_var, model_time, model_levs, model_lats, model_lons = readcesm_3D(
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
print(model_lats[model_latl])
print(model_lats[model_latu])
print(model_lons[model_lonl])
print(model_lons[model_lonr])

wet_index = np.in1d(model_time.year, years_wet) & np.in1d(model_time.month, monsoon_period)
dry_index = np.in1d(model_time.year, years_dry) & np.in1d(model_time.month, monsoon_period)
wet_time = model_time[wet_index]
dry_time = model_time[dry_index]

print(model_time[wet_index])


################################################################################################
# calculate Hovmöller diagram of vorticity at 700 hPa based on Takahashi_etal_2006
model_lev = np.argmin(np.abs(model_levs - 700))


# calculate vorticity
model_u_lev = model_u[:, model_lev, :, :]
model_v_lev = model_v[:, model_lev, :, :]

model_u_wet = model_u_lev[wet_index, :, :]
model_u_dry = model_u_lev[dry_index, :, :]
model_v_wet = model_v_lev[wet_index, :, :]
model_v_dry = model_v_lev[dry_index, :, :]

vr_wet = getvc(model_lats, model_lons, model_u_wet, model_v_wet)
vr_dry = getvc(model_lats, model_lons, model_u_dry, model_v_dry)


# calculate mean omega
model_lev = np.argmin(np.abs(model_levs - plevel))
model_var = model_var[:, model_lev, :, :] * 1000

model_var_wet = model_var[wet_index, :, :]
model_var_dry = model_var[dry_index, :, :]


model_latl = np.argmin(np.abs(model_lats - vc_latbounds[0]))
model_latu = np.argmin(np.abs(model_lats - vc_latbounds[1]))

print(model_lats[model_latl])
print(model_lats[model_latu])

# wet_corr = np.corrcoef(vr_wet.flatten(), model_var_wet.flatten())[0][1]
# dry_corr = np.corrcoef(vr_dry.flatten(), model_var_dry.flatten())[0][1]
#
# print('Correlation for Wet/Dry and Diff:')
# print(wet_corr)
# print(dry_corr)

vr_wet = np.mean(vr_wet[:, model_latl:model_latu+1, :], axis=1)
vr_dry = np.mean(vr_dry[:, model_latl:model_latu+1, :], axis=1)

vc_wet = np.mean(model_var_wet[:, model_latl:model_latu+1, :], axis=1)
vc_dry = np.mean(model_var_dry[:, model_latl:model_latu+1, :], axis=1)


# print(vc_wet[25, :])

ndays = int(len(wet_time)/len(years_wet))

vc_wet_mean = np.zeros((ndays, len(model_lons)))
vc_dry_mean = np.zeros((ndays, len(model_lons)))
vc_wet_std = np.zeros((ndays, len(model_lons)))
vc_dry_std = np.zeros((ndays, len(model_lons)))

vr_wet_mean = np.zeros((ndays, len(model_lons)))
vr_dry_mean = np.zeros((ndays, len(model_lons)))
vr_wet_std = np.zeros((ndays, len(model_lons)))
vr_dry_std = np.zeros((ndays, len(model_lons)))

for idx in range(ndays):
    vc_wet_mean[idx, :] = np.mean(vc_wet[idx::ndays, :], axis=0)
    vc_wet_std[idx, :] = np.std(vc_wet[idx::ndays, :], axis=0)
    vc_dry_mean[idx, :] = np.mean(vc_dry[idx::ndays, :], axis=0)
    vc_dry_std[idx, :] = np.std(vc_dry[idx::ndays, :], axis=0)

    vr_wet_mean[idx, :] = np.mean(vr_wet[idx::ndays, :], axis=0)
    vr_wet_std[idx, :] = np.std(vr_wet[idx::ndays, :], axis=0)
    vr_dry_mean[idx, :] = np.mean(vr_dry[idx::ndays, :], axis=0)
    vr_dry_std[idx, :] = np.std(vr_dry[idx::ndays, :], axis=0)


res_diff, res_sig = cal_diff(vc_wet_mean, vc_dry_mean, vc_wet_std, vc_dry_std, len(years_wet), len(years_dry))
vrres_diff, vrres_sig = cal_diff(vr_wet_mean, vr_dry_mean, vr_wet_std, vr_dry_std, len(years_wet), len(years_dry))

# vr_wet_mean[vr_wet_mean < 0] = 0.
# vr_dry_mean[vr_dry_mean < 0] = 0.
# vrres_diff[vrres_diff < 0] = 0.

wet_corr = np.corrcoef(vr_wet_mean.flatten(), vc_wet_mean.flatten())[0][1]
dry_corr = np.corrcoef(vr_dry_mean.flatten(), vc_dry_mean.flatten())[0][1]
res_corr = np.corrcoef(vrres_diff.flatten(), res_diff.flatten())[0][1]

print('Correlation for Wet/Dry and Diff:')
print(wet_corr)
print(dry_corr)
print(res_corr)

# vc_wet_mean = vc_wet_mean
# vc_dry_mean = vc_dry_mean
# res_diff = res_diff

# print(vc_wet_mean[25, :])
# print(res_diff[25, :])

days = range(ndays)
model_sig = np.zeros((ndays, len(model_lons)))
model_sig[:, :] = 0.5

plot_data = [vc_wet_mean, vc_dry_mean, res_diff]
plot_days = [days, days, days]
plot_lons = [model_lons, model_lons, model_lons]
plot_test = [model_sig, model_sig, res_sig]
plot_corr = [wet_corr, dry_corr, res_corr]

df = {}
df['index'] = ['Wet', 'Dry', 'Wet-Dry']
df['corr with 700hPa Vorticity'] = plot_corr
df = pd.DataFrame(df)
df = df.set_index('index')
df.to_csv(outdir+str(plevel)+'hPa_omega_corr_with_700hPa_vorticity.csv', sep=',')

legends = ['a) Wet', 'b) Dry', 'c) Wet-Dry']

title = ' CESM differences in '+str(plevel)+'hPa zonal averaged omega between Wet and Dry years'
fname = 'icam5_'+str(plevel)+'hPa_omega_SEA_daily_contour_diff_drywet_'+monsoon_period_str

print('plotting for season/month: '+monsoon_period_str+'...')
# plot for wet-dry contours
plt.clf()
fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ss in range(3):
    axes[ss].set_title(legends[ss]+', Correlation with 700hPa Vorticity: '+str(np.round(plot_corr[ss], 3)), fontsize=7, pad=0.3)

    x, y = np.meshgrid(model_lons, days)
    if ss < 2:
        clevs = np.arange(-120, 120.1, 30)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.PuOr_r, alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_ylabel('Days', fontsize=7, labelpad=1.2)
        axes[ss].set_xticks([])
        axes[ss].yaxis.set_tick_params(labelsize=6)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-120, 120.1, 30)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_xlabel('Longitude', fontsize=7, labelpad=1.2)
        axes[ss].set_ylabel('Days', fontsize=7, labelpad=1.2)
        axes[ss].xaxis.set_tick_params(labelsize=6)
        axes[ss].yaxis.set_tick_params(labelsize=6)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=4)
    cbar.ax.set_title(r'$\times 10^{-3} Pa/s$', fontsize=5, loc='left')

plt.savefig(outdir+fname+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=7, y=0.95)
plt.savefig(outdir+fname+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)

# plot for wet-dry contours with sig
plt.clf()
fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ss in range(3):
    axes[ss].set_title(legends[ss], fontsize=7, pad=0.3)

    x, y = np.meshgrid(model_lons, days)
    if ss < 2:
        clevs = np.arange(-120, 120.1, 30)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.PuOr_r, alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_ylabel('Days', fontsize=7, labelpad=1.2)
        axes[ss].set_xticks([])
        axes[ss].yaxis.set_tick_params(labelsize=6)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-120, 120.1, 30)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.RdBu, alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_xlabel('Longitude', fontsize=7, labelpad=1.2)
        axes[ss].set_ylabel('Days', fontsize=7, labelpad=1.2)
        axes[ss].xaxis.set_tick_params(labelsize=6)
        axes[ss].yaxis.set_tick_params(labelsize=6)
        temptest = plot_test[ss]
        levels = [0., 2.01, temptest.max()]
        axes[ss].contourf(x, y, plot_test[ss], levels=levels, colors='none', hatches=['', '//////'], alpha=0)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=4)
    cbar.ax.set_title(r'$\times 10^{-3} Pa/s$', fontsize=5, loc='left')

plt.savefig(outdir+fname+'_wtsig.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=7, y=0.95)
plt.savefig(outdir+fname+'_wtsig.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)


# plot for wet and dry contours
plt.clf()
fig, axes = plt.subplots(2, 1)
axes = axes.flatten()

for ss in range(2):
    axes[ss].set_title(legends[ss], fontsize=7, pad=0.3)

    x, y = np.meshgrid(model_lons, days)
    if ss < 1:
        clevs = np.arange(-120, 120.1, 30)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.PuOr_r, alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_ylabel('Days', fontsize=8, labelpad=1.2)
        axes[ss].set_xticks([])
        axes[ss].yaxis.set_tick_params(labelsize=7)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    else:
        clevs = np.arange(-120, 120.1, 30)
        cs = axes[ss].contourf(x, y, plot_data[ss], clevs, cmap=cm.PuOr_r, alpha=0.9, extend="both")
        axes[ss].axvline(x=105, linewidth=1.5, color='black')
        axes[ss].set_xlabel('Longitude', fontsize=8, labelpad=1.2)
        axes[ss].set_ylabel('Days', fontsize=8, labelpad=1.2)
        axes[ss].xaxis.set_tick_params(labelsize=7)
        axes[ss].yaxis.set_tick_params(labelsize=7)
        cbar = fig.colorbar(cs, shrink=0.8, pad=0.03, orientation='vertical', ax=axes[ss])
        ticks = clevs
        ticks = np.round(ticks, 2)
        ticklabels = [str(itick) for itick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.set_title(r'$\times 10^{-3} Pa/s$', fontsize=6, loc='left')

plt.savefig(outdir+fname+'_nodiff.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=8, y=0.95)
plt.savefig(outdir+fname+'_nodiff.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
