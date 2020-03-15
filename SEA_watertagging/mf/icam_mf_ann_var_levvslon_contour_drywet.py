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
outdir = '/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/annual_variability/mf_atBC/drywet/'

# set up variable names and file name
varname = 'mf'
var_longname = 'Moisture Fluxes'
varstr = "MF"
var_unit = r'$g/kg\times m/s$'


# define inital year and end year
iniyear = 1980
endyear = 2005

# define the contour plot region
latbounds = [-20, 40]
lonbounds = [90, 130]

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

# define Wet-Dry yyears
monsoon_period_str = 'JJA'
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

# contants
oro_water = 997  # kg/m^3
g = 9.8  # m/s^2
r_earth = 6371000  # m
rgas = 287.058

############################################################################
# define functions
############################################################################


def plot_contour(plot_udata, plot_vdata, plot_data, xx, yy, colormap, clevs, xticks, xlabel, varname, var_unit, legends, title, fname):

    plt.clf()
    figsize = (8, 6)
    fig = plt.figure(figsize=figsize)

    for idx in range(3):
        ax = fig.add_subplot(3, 1, idx+1)
        ax.set_title(legends[idx], fontsize=6)
        cs = ax.contourf(xx, yy, plot_data[idx], levels=clevs[idx], cmap=colormap[idx], extend='both')

        cq = ax.quiver(xx[::1], yy[::1], plot_udata[idx][::1, ::1],
                       plot_vdata[idx][::1, ::1], scale=2.5, scale_units='xy')
        if idx == 0:
            qk = ax.quiverkey(cq, 0.75, 0.9, 20, '20', labelpos='E',
                              coordinates='figure', fontproperties={'size': 4})

        # set x/y tick label size
        if idx == 2:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)
            ax.xaxis.set_tick_params(labelsize=5)
            ax.set_xlabel(xlabel, fontsize=5)
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
print(model_ps.shape)

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

varname = 'OMEGA'
varfname = 'OMEGA'
model_w, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)

varname = 'T'
varfname = 'T'
model_t, model_time, model_levs, model_lats, model_lons = readcesm_3D(
    varname, iniyear, endyear, resolution, varfname, case, frequency, latbounds, lonbounds)


ntime = len(model_time)
nlats = len(model_lats)
nlons = len(model_lons)
nlevs = len(model_levs)

# print(model_time)
print(model_q.shape)
print(model_levs)

# find regional lat/lon boundaries
model_latl = np.argmin(np.abs(model_lats - reg_lats[0]))
model_latu = np.argmin(np.abs(model_lats - reg_lats[1]))
model_lonl = np.argmin(np.abs(model_lons - reg_lons[0]))
model_lonr = np.argmin(np.abs(model_lons - reg_lons[1]))
levtop = np.abs(model_levs - ptop).argmin()
# print(model_lons[model_lonl])
# print(model_lons[model_lonr])

wet_index = np.in1d(model_time.year, years_wet) & np.in1d(model_time.month, monsoon_period)
dry_index = np.in1d(model_time.year, years_dry) & np.in1d(model_time.month, monsoon_period)
wet_time = model_time[wet_index]
dry_time = model_time[dry_index]
print(wet_time)

for ilev in range(nlevs):
    temp = model_u[:, ilev, :, :]
    temp[model_ps < model_levs[ilev]] = np.nan
    model_u[:, ilev, :, :] = temp

    temp = model_v[:, ilev, :, :]
    temp[model_ps < model_levs[ilev]] = np.nan
    model_v[:, ilev, :, :] = temp

    temp = model_w[:, ilev, :, :]
    rho = 100*model_levs[ilev]/(rgas*model_t[:, ilev, :, :])
    temp = -temp / (rho*g)  # convert Pa/s to m/s
    temp[model_ps < model_levs[ilev]] = np.nan
    model_w[:, ilev, :, :] = temp

    temp = model_q[:, ilev, :, :]
    temp[model_ps < model_levs[ilev]] = np.nan
    model_q[:, ilev, :, :] = temp

# print(rho)
# print(model_ps[0, 3, 3])
# print(model_u[0, :, 3, 3])


############################################################################
# calculate the lev vs lats
############################################################################

############################################################################
# convert q flux to kg/kg*m/s and calculate the zonal moisture flux
model_q = model_q
model_qu = model_q * model_u
model_qv = model_q * model_v
model_qw = model_q * model_w / g/oro_water

legends = ['Wet composites', 'Dry composites', 'Differences (Wet-Dry)']

model_qu_xx = np.mean(model_qu[:, :, model_latl:model_latu+1, :], axis=2)
model_qw_xx = np.mean(model_qw[:, :, model_latl:model_latu+1, :], axis=2)
model_q_xx = np.mean(model_q[:, :, model_latl:model_latu+1, :], axis=2)

model_qu_wet = np.mean(model_qu_xx[wet_index, :, :], axis=0)
model_qw_wet = np.mean(model_qw_xx[wet_index, :, :], axis=0)

model_qu_dry = np.mean(model_qu_xx[dry_index, :, :], axis=0)
model_qw_dry = np.mean(model_qw_xx[dry_index, :, :], axis=0)

model_q_wet = np.mean(model_q_xx[wet_index, :, :], axis=0)
model_q_dry = np.mean(model_q_xx[dry_index, :, :], axis=0)

model_qu_wet = model_qu_wet[levtop:, :]
model_qw_wet = model_qw_wet[levtop:, :]
model_qu_dry = model_qu_dry[levtop:, :]
model_qw_dry = model_qw_dry[levtop:, :]
model_q_wet = model_q_wet[levtop:, :]
model_q_dry = model_q_dry[levtop:, :]
print(model_q_wet.shape)

res_qu_diff = model_qu_wet - model_qu_dry
res_qw_diff = model_qw_wet - model_qw_dry
res_q_diff = model_q_wet - model_q_dry

print(model_qu_wet[:, 0])
print(model_qw_wet[:, 0])
print(model_q_wet[:, 0])

plot_data = [model_q_wet, model_q_dry, res_q_diff]
plot_udata = [model_qu_wet, model_qu_dry, res_qu_diff]
plot_vdata = [model_qw_wet, model_qw_dry, res_qw_diff]


clevs = [np.arange(0, 3.1, 0.5), np.arange(0, 3.1, 0.5), np.arange(-3, 3.1, 0.5)]
colormap = [cm.Blues, cm.Blues, cm.RdBu_r]


title = ' icam monthly averaged '+var_longname+' in '+monsoon_period_str
fname = 'icam5_' + varstr + '_SEA_monthly_mean_levvslon_contour_drywet_'+monsoon_period_str

xticks = np.arange(lonbounds[0], lonbounds[1], 20)
xlabel = [str(ii) for ii in xticks]

plot_contour(plot_udata, plot_vdata, plot_data, model_lons, model_levs[levtop:], colormap, clevs,
             xticks, xlabel, 'Specific Humidity', var_unit, legends, title, outdir+fname)
