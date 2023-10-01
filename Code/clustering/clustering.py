import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN,AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

def plot_clustering(clusterer, cluster_labels, n_clusters, df, clm):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(df, cluster_labels)
    print("Average silhouette score", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(df[:, 0], df[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    '''
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
    '''
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel(clm[0])
    ax2.set_ylabel(clm[1])

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = 3",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig("C:/Users/annin/PycharmProjects/Tesi/Data/ClusteringPlots/Plot - " + str(clusterer) + ".png")
    plt.show()

def remove_correlated_features(data, columns_names):
    cov = np.cov(data, rowvar=False)
    c = pd.DataFrame(np.abs(cov), columns=columns_names)
    # select upper triangle of correlation matrix
    upper = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    data_sc = pd.DataFrame(data, columns=columns_names)
    data_sc.drop(to_drop, axis=1, inplace=True)
    print("[INFO] Removed ", len(to_drop), "correlated features")
    return data_sc

def remove_correlated_features(data, columns_names):
    cov = np.cov(data, rowvar=False)
    c = pd.DataFrame(np.abs(cov), columns=columns_names)
    # select upper triangle of correlation matrix
    print(c.to_markdown())
    upper = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
    print(upper.to_markdown())
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    data_sc = pd.DataFrame(data, columns=columns_names)
    data_sc.drop(to_drop, axis=1, inplace=True)
    print("[INFO] Removed ", len(to_drop), "correlated features: ")
    print(to_drop)
    print("left ", data_sc.columns)
    return data_sc

def ClusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]
if __name__ == '__main__':
    n_clusters = 3
    df = pd.read_excel('C:/Users/annin/PycharmProjects/Tesi/Data/Dataset-normalized lengths.xlsx')
    col = pd.DataFrame()
    col['Patient'] = df['Patient']
    col['Task'] = df['Exercise']

    df.drop(['Unnamed: 0', 'Patient'], axis = 1, inplace = True)
    dp = pd.DataFrame()

    #dp['Normalized Stride Length'] = df['Normalized Stride Length']
    #dp['Normalized Step Length'] = df['Normalized Step Length']
    #dp['Normalized Left Stride Length'] = df['Normalized Left Stride Length']
    #dp['Normalized Right Stride Length'] = df['Normalized Right Stride Length']
    #dp['Left Step Length'] = df['Left Step Length']
    #dp['Right Step Length'] = df['Right Step Length']
    #dp['Normalized Left Step Length'] = df['Normalized Left Step Length']
    #dp['Normalized Right Step Length'] = df['Normalized Right Step Length']
    dp['Gait Speed'] = df['Gait Speed']
    '''
    dp['Left - Right Single Ratio']=df['Left - Right Single Ratio']
    dp['Right Angular Y Skewness'] = df['Right Angular Y Skewness']
    #df = remove_correlated_features(df, clm)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    #print(upper.to_markdown())
    #print(df)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    while len(to_drop) > 0:
        df.drop(to_drop[0], axis = 1, inplace = True)
        #print("Removed ", to_drop[0])
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.87)]
    #print(upper[to_drop].to_markdown())
    #print(corr_matrix.to_markdown())
    
    '''
    clm = dp.columns
    dp = dp.to_numpy()

    #pca = PCA(n_components=2)
    #df = pca.fit_transform(df)

    #variance = pca.explained_variance_ratio_ * 100

    #for v in variance:
    #    print(f"% Variance Ratio per PC ", v)

    clusterer = KMeans(n_clusters=n_clusters,random_state=12345,n_init=100,algorithm='elkan')
    #clusterer = DBSCAN(eps=15, min_samples=10)
    #clusterer = AgglomerativeClustering(n_clusters = n_clusters)

    cluster_labels = clusterer.fit_predict(dp)

    #plot_clustering(clusterer=clusterer, cluster_labels=cluster_labels, n_clusters=n_clusters, df=dp, clm = clm)
    dp = pd.DataFrame(dp, columns = clm)
    dp.insert(loc = 0, column = 'Patient', value =  col['Patient'],allow_duplicates=True)
    dp.insert(loc = 1 ,column = 'Task', value = col['Task'], allow_duplicates=True)
    get_0 =ClusterIndicesNumpy(0, clusterer.labels_)
    get_1 =ClusterIndicesNumpy(1, clusterer.labels_)
    get_2 = ClusterIndicesNumpy(2, clusterer.labels_)
    plt.figure(figsize = (32,12))
    plt.plot(dp.iloc[(get_0)]['Gait Speed'], dp.iloc[(get_0)]['Gait Speed'], 'o', label = 'cluster 0')
    plt.plot(dp.iloc[(get_1)]['Gait Speed'], dp.iloc[(get_1)]['Gait Speed'], 'o', label='cluster 1')
    plt.plot(dp.iloc[(get_2)]['Gait Speed'],dp.iloc[(get_2)]['Gait Speed'], 'o', label='cluster 2')
    plt.legend()
    plt.show()
    print("Cluster 0\n", dp.iloc[get_0].to_markdown())
    print("Cluster 1\n", dp.iloc[get_1].to_markdown())
    print("Cluster 2\n", dp.iloc[get_2].to_markdown())

    # save_estimator
    #save_estimator(clusterer, "KMeans.joblib")
