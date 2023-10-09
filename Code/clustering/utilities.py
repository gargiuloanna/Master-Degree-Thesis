import numpy as np

def cluster_indices(clusterer, n_clusters):
    samples_clusters = {}
    print(clusterer.labels_)
    for i in range(n_clusters):
        print(i, clusterer.labels_[i])
        samples_clusters[i] = np.where(clusterer.labels_ == i)[0]
    return samples_clusters


def print_samples(samples_clusters, data):
    for i in samples_clusters.keys():
        print("Cluster " + str(i), data.iloc[samples_clusters[i]].to_markdown())