"""
Kmeans clustering demo. We create some blobs on a plot,
then run kmeans to show how centroids move over iterations.

All this can be done in 3 lines of code, but I broke up each iteration step
so that it's easier to visualize.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def kmeans_demo():
    n_clusters=3  #Number of clusters we want for kmeans
    n_steps=10    #Number of iterations to draw

    # Randomly generate some cluster data
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # Create some intial guesses to where the centroids are
    initial_centers = X[0:3, :]
    initial_centers = np.array([[-8, 0], [-2, 5], [2, -12]])

    # Run n_steps of iterations of kmeans
    for i in range(n_steps):
        km = KMeans(n_clusters=n_clusters, init=initial_centers, max_iter=1)
        y_pred = km.fit_predict(X)
        centers = km.cluster_centers_

        # Plot the prediction of what point belongs to what cluster at each iteration
        plot_data(X, y_pred, centers, title="Kmeans step number: %d" % i)

        initial_centers = centers


def plot_data(X, y, centers, title=None):
    """
    Plot a scatter plot of the points, labels and centers.
    :param X: 2D data
    :param y: array of cluster IDs, one for each row of X
    :param centers: 2D array of n_samples x n_features of centroids
    :param title: Title of the plot
    """

    plt.scatter(X[:, 0], X[:, 1], c=y)
    if title is None:
        title = "Scatter Plot"
    plt.scatter(centers[:, 0], centers[:, 1],
                marker='x', s=169, linewidths=3,
                color='b', zorder=10)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.show()


if __name__ == "__main__":
    kmeans_demo()
