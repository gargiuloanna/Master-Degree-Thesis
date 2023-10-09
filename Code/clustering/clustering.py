import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from utilities import cluster_indices, print_samples
from Code.plotting.cluster_plots import plot_clustering, plot_single_feature

if __name__ == '__main__':
    n_clusters = 3

    df = pd.read_excel('C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/Dataset-only normalized lengths.xlsx')

    col = pd.DataFrame()
    col['Patient'] = df['Patient']
    col['Task'] = df['Exercise']

    df.drop(['Unnamed: 0', 'Patient', 'Exercise'], axis = 1, inplace = True)
    #dp = pd.DataFrame()

    #dp['Normalized Stride Length'] = df['Normalized Stride Length']
    #dp['Normalized Step Length'] = df['Normalized Step Length']
    #dp['Normalized Left Stride Length'] = df['Normalized Left Stride Length']
    #dp['Normalized Right Stride Length'] = df['Normalized Right Stride Length']
    #dp['Left Step Length'] = df['Left Step Length']
    #dp['Right Step Length'] = df['Right Step Length']
    #dp['Normalized Left Step Length'] = df['Normalized Left Step Length']
    #dp['Normalized Right Step Length'] = df['Normalized Right Step Length']
    #dp['Gait Speed'] = df['Gait Speed']

    clm = df.columns
    #dp = dp.to_numpy()
    dp = df.to_numpy()
    clusterer = KMeans(n_clusters=n_clusters,random_state=12345,n_init=100,algorithm='elkan')
    #clusterer = DBSCAN(eps=15, min_samples=10)
    #clusterer = AgglomerativeClustering(n_clusters = n_clusters)

    cluster_labels = clusterer.fit_predict(dp)

    plot_clustering(clusterer=clusterer, cluster_labels=cluster_labels, n_clusters=n_clusters, df=dp, clm = clm)

    dp = pd.DataFrame(dp, columns = clm)
    dp.insert(loc = 0, column = 'Patient', value =  col['Patient'],allow_duplicates=True)
    dp.insert(loc = 1 ,column = 'Task', value = col['Task'], allow_duplicates=True)

    samples_cluster = cluster_indices(clusterer, n_clusters=n_clusters)

    #plot_single_feature(samples_cluster, n_clusters)

    print_samples(samples_cluster, dp)

    # save_estimator
    #save_estimator(clusterer, "KMeans.joblib")
