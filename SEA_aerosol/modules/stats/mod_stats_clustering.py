from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import matplotlib.cm as cm
'''
#This is a module that used to cluster data

#Written by Harry Li
'''

# modules that the function needs
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, scale
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.switch_backend('agg')


################################################################################################
# k-means
################################################################################################

def kmeans_cluster(var, ncluster, stnlats, stnlons, stnnames, fname, map_plot=True):
    for ii in range(var.shape[1]):
        for jj in range(var.shape[1]):
            var[np.isnan(var[:, jj]), ii] = np.NaN

    temp = []
    for ii in range(var.shape[1]):
        temp.append(var[~np.isnan(var[:, ii]), ii])

    temp = np.array(temp)
    print(temp.shape)
    nclusters_list = np.arange(2, 10)
    Jcost = []
    res = []

    # use pca to compress data
    # temp = scale(temp)
    # pca = PCA().fit(temp)
    # print(pca.explained_variance_ratio_)
    # print(np.sum(pca.explained_variance_ratio_))
    # sum = 0
    # k = 0
    # while sum < 0.99:
    #     sum += pca.explained_variance_ratio_[k]
    #     k += 1
    # print(k)

    for icluster in nclusters_list:
        print('run kMeans algorithm for '+str(icluster)+' clusters')

        # use pca as first guess
        pca = PCA(n_components=icluster).fit(temp)
        kmeans = KMeans(n_clusters=icluster, init=pca.components_, n_init=1, max_iter=300).fit(temp)

        # use pca to compress data
        # pca_data = PCA(n_components=k).fit_transform(temp)
        # print(pca_data.shape)
        # kmeans = KMeans(n_clusters=icluster, init='k-means++', n_init=10, max_iter=300).fit(pca_data)

        print(kmeans.labels_)
        Jcost.append(kmeans.inertia_)

        res.append(kmeans)
        # if icluster==ncluster:
        # res = kmeans

        # if map_plot=True,then plot the cluster
        if (map_plot is True):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            map = Basemap(llcrnrlon=91.0, llcrnrlat=6, urcrnrlon=110.0, urcrnrlat=22,
                          lat_0=14., lon_0=100.5, resolution='h', epsg=4326)
            # map = Basemap(projection='', lat_0=17, lon_0=100.5,resolution = 'l', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

            # map.drawcoastlines(linewidth=0.3)
            # map.drawcountries()
            map.arcgisimage(service='World_Physical_Map', xpixels=50000)

            latsmap = stnlats[0:]
            lonsmap = stnlons[0:]
            x, y = map(lonsmap, latsmap)

            cs = map.scatter(x, y, s=20, marker="o", c=kmeans.labels_,
                             cmap=plt.cm.get_cmap('viridis', icluster), alpha=0.7)

            labels = stnnames[0:]
            count = 0
            for label, xpt, ypt in zip(labels, x, y):
                plt.text(xpt+0.1, ypt+0.1, str(count+1)+':'+label, fontsize=5)
                count += 1

            fig.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
            cbar_ax = fig.add_axes([0.2, 0.17, 0.6, 0.02])
            cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal', ticks=range(icluster))
            cbar.ax.tick_params(labelsize=4)
            cbar.set_label('Cluster labels')
            plt.clim(-0.5, icluster-0.5)

            title = 'K-means clustering: '+str(icluster)+' groups'
            plt.suptitle(title, fontsize=9, y=0.95)
            plt.savefig(fname+'_'+str(icluster)+'.pdf', bbox_inches='tight', dpi=3000)
            plt.close(fig)

        # print(kmeans.cluster_centers_)
        # print(kmeans.cluster_centers_.shape)
        # print(icluster)
        # print(kmeans.labels_)
        # print(kmeans.inertia_)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(nclusters_list, Jcost, c='k', linewidth=1.5)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')

    title = 'K-means clustering'
    plt.suptitle(title, fontsize=9, y=0.95)
    plt.savefig(fname+'.pdf', bbox_inches='tight', dpi=3000)
    plt.close(fig)

    return res

################################################################################################
# SpectralClustering
################################################################################################


def spectral_cluster(var, ncluster, stnlats, stnlons, stnnames, fname):
    for ii in range(var.shape[1]):
        for jj in range(var.shape[1]):
            var[np.isnan(var[:, jj]), ii] = np.NaN

    temp = []
    for ii in range(var.shape[1]):
        temp.append(var[~np.isnan(var[:, ii]), ii])

    temp = np.array(temp)
    print(temp.shape)
    nclusters_list = np.arange(2, 10)
    Jcost = []
    res = []
    print(nclusters_list)

    for icluster in nclusters_list:
        print('run spectral clustering algorithm for '+str(icluster)+' clusters')
        model = SpectralClustering(n_clusters=icluster, affinity='nearest_neighbors', assign_labels='kmeans').fit(temp)
        print(model.labels_)
        res.append(model)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        map = Basemap(llcrnrlon=91.0, llcrnrlat=6, urcrnrlon=110.0, urcrnrlat=22,
                      lat_0=14., lon_0=100.5, resolution='h', epsg=4326)
        # map = Basemap(projection='', lat_0=17, lon_0=100.5,resolution = 'l', area_thresh = 0.1,llcrnrlon=91.0, llcrnrlat=6.5,urcrnrlon=110.0, urcrnrlat=27.5)

        # map.drawcoastlines(linewidth=0.3)
        # map.drawcountries()
        map.arcgisimage(service='World_Physical_Map', xpixels=50000)

        latsmap = stnlats[0:]
        lonsmap = stnlons[0:]
        x, y = map(lonsmap, latsmap)

        cs = map.scatter(x, y, s=20, marker="o", c=model.labels_, cmap=plt.cm.get_cmap('viridis', icluster), alpha=0.7)

        labels = stnnames[0:]
        count = 0
        for label, xpt, ypt in zip(labels, x, y):
            plt.text(xpt+0.1, ypt+0.1, str(count+1)+':'+label, fontsize=5)
            count += 1

        fig.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
        cbar_ax = fig.add_axes([0.2, 0.17, 0.6, 0.02])
        cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal', ticks=range(icluster))
        cbar.ax.tick_params(labelsize=4)
        cbar.set_label('Cluster labels')
        plt.clim(-0.5, icluster-0.5)

        title = 'Spectral clustering: '+str(icluster)+' groups'
        plt.suptitle(title, fontsize=9, y=0.95)
        plt.savefig(fname+'_'+str(icluster)+'.pdf', bbox_inches='tight', dpi=3000)
        plt.close(fig)

    return model
