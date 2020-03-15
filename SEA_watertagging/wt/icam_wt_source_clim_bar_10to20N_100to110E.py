# import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


odir = "/scratch/d/dylan/harryli/gpcdata/Analysis/icam5/SEA_wt_1920today/climatology/watertag/"

lats = 10
latn = 20
lonw = 100
lone = 110
iniyear = 1980
endyear = 2005


##############################################################################
# reading data
##############################################################################


month = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
prect = np.array([1.54, 1.37, 2.05, 3.51, 6.12, 6.28, 8.05, 7.72, 8.31, 7.66, 5.04, 2.53])

localfrac = np.array([0.03, 0.05, 0.08, 0.13, 0.13, 0.12, 0.11, 0.12, 0.12, 0.08, 0.05, 0.03])
localpre = prect*localfrac

indoceanfrac = np.array([0.02, 0.03, 0.06, 0.16, 0.50, 0.58, 0.55, 0.47, 0.34, 0.15, 0.04, 0.02])
indoceanpre = prect*indoceanfrac

pacificfrac = np.array([0.81, 0.79, 0.71, 0.54, 0.20, 0.11, 0.11, 0.18, 0.32, 0.58, 0.75, 0.81])
pacificpre = prect*pacificfrac

tagfrac = np.array([0.92, 0.92, 0.92, 0.92, 0.91, 0.89, 0.86, 0.85, 0.87, 0.91, 0.92, 0.93])

monnam = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

##############################################################################
# plot for Fractional moisture contribution
##############################################################################
legends = ['Total tags', 'Indian Ocean', 'Local Evaporation', 'Pacific Ocean']
filename = "icam_"+str(iniyear)+"_"+str(endyear)+"_SEA_watersource_frac_"+str(lats) + \
    "N_"+str(latn)+"N_"+str(lonw)+"E_"+str(lone)+"E_py"
title = "CESM "+str(iniyear)+" to "+str(endyear)+" Fractional Moisture Sources Contribution to SEA Precipitation"

plt.clf()
figsize = (8, 6)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1, 1, 1)

ax.bar(month, tagfrac, width=0.75, color='royalblue', edgecolor='black', linewidth=.5, label=legends[0])
ax.bar(month, indoceanfrac, width=0.75, color='lightskyblue', edgecolor='black', linewidth=.5, label=legends[1])
ax.bar(month, localfrac, width=0.75, bottom=indoceanfrac, color='limegreen',
       edgecolor='black', linewidth=.5, label=legends[2])
ax.bar(month, pacificfrac, width=0.75, bottom=indoceanfrac +
       localfrac, color='lightpink', edgecolor='black', linewidth=.5, label=legends[3])

ax.set_xticks(month)
ax.set_xticklabels(monnam, fontsize=9)
yticks = np.arange(0, 1.1, 0.1)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_yticklabels([str(np.round(yy, 2)) for yy in yticks], fontsize=9)
ax.set_ylabel('Relative Contribution (Fraction)', fontsize=9, labelpad=3.5)

# fig.subplots_adjust(bottom=0.4)
ax.legend(ncol=len(legends), bbox_to_anchor=(0.06, -.12, .85, 0.2),
          loc='lower center', fontsize=9, edgecolor='None', handlelength=2.5, handletextpad=1., columnspacing=2.)

plt.savefig(odir+filename+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(odir+filename+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)


##############################################################################
# plot for Fractional moisture contribution
##############################################################################
legends = ['SEA Precipitation', 'Indian Ocean', 'Local Evaporation', 'Pacific Ocean']
filename = "icam_"+str(iniyear)+"_"+str(endyear)+"_SEA_watersource_"+str(lats) + \
    "N_"+str(latn)+"N_"+str(lonw)+"E_"+str(lone)+"E_py"
title = "CESM "+str(iniyear)+" to "+str(endyear)+" Fractional Moisture Sources Contribution to SEA Precipitation"

plt.clf()
figsize = (8, 6)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1, 1, 1)

# ax.bar(month, prect, width=0.75, color='royalblue', edgecolor='black', linewidth=.5, label=legends[0])
# ax.bar(month, indoceanpre, width=0.75, color='lightskyblue', edgecolor='black', linewidth=.5, label=legends[1])
# ax.bar(month, localpre, width=0.75, bottom=indoceanpre, color='limegreen',
#        edgecolor='black', linewidth=.5, label=legends[2])
# ax.bar(month, pacificpre, width=0.75, bottom=indoceanpre +
#        localpre, color='lightpink', edgecolor='black', linewidth=.5, label=legends[3])

ax.plot(month, prect, color='royalblue', linestyle='solid', linewidth=2., label=legends[0])
ax.plot(month, indoceanpre, color='lightskyblue', linestyle='solid', linewidth=2., label=legends[1])
ax.plot(month, localpre,  color='limegreen', linestyle='solid', linewidth=2., label=legends[2])
ax.plot(month, pacificpre, color='lightpink', linestyle='solid', linewidth=2., label=legends[3])

ax.set_xticks(month)
ax.set_xticklabels(monnam, fontsize=9)
yticks = np.arange(0, 9.1, 1.)
ax.set_yticks(yticks)
ax.set_yticklabels([str(np.round(yy, 2)) for yy in yticks], fontsize=9)
ax.set_ylabel('Precipitation (mm/day)', fontsize=9, labelpad=3.5)

# fig.subplots_adjust(bottom=0.4)
ax.legend(ncol=len(legends), bbox_to_anchor=(0.06, -.12, .85, 0.2),
          loc='lower center', fontsize=9, edgecolor='None', handlelength=2.5, handletextpad=1., columnspacing=2.)

plt.savefig(odir+filename+'.png', bbox_inches='tight', dpi=600)
plt.suptitle(title, fontsize=9, y=0.95)
plt.savefig(odir+filename+'.pdf', bbox_inches='tight', dpi=600)
plt.close(fig)
