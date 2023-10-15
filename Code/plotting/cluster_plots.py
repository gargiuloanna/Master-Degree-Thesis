import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import random

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
        uno = random.randint(0,107)
        due = random.randint(0, 107)
        ax2.scatter(df[:, uno], df[:,due], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

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
        ax2.set_title("The visualization of the clustered Data.")
        ax2.set_xlabel(clm[uno])
        ax2.set_ylabel(clm[due])

        plt.suptitle(
                "Silhouette analysis on sample Data with n_clusters = " + str(n_clusters),
                fontsize=14,
                fontweight="bold",
        )
        plt.savefig("C:/Users/annin/PycharmProjects/Master-Degree-Thesis/Code/Data/ClusteringPlots/Plot - " + str(clusterer) + ".png")
        plt.show()


def plot_single_feature(samples_clusters, n_clusters):
    plt.figure(figsize=(32, 12))
    for i in range(n_clusters):
        plt.plot(samples_clusters[0], 'o', label='cluster ' + str(i))
    plt.legend()
    plt.show()