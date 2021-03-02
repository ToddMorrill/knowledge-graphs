"""This module implements unsupervised clustering using KMeans.

Examples:
    $ python cluster.py
"""
import logging
import os
import pickle
import time

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn import datasets
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class Cluster:
    def __init__(self, data):
        self.data = np.array(data)

    def fit(self, k):
        self.k = k
        self.model = KMeans(n_clusters=k,
                            random_state=RANDOM_STATE).fit(self.data)

    def find_k_fit(self, k_max=80, k_step=4, var_explained=50):
        # Explain some fraction of the data's variance
        self.var_explained = var_explained

        models = []
        sse = {}
        centroids = []
        experimental_ks = range(1, k_max + 1, k_step)
        logger.info(f'Fitting {len(experimental_ks)} KMeans models.')
        model_start = time.time()
        for k in experimental_ks:
            start = time.time()
            model = KMeans(n_clusters=k,
                           random_state=RANDOM_STATE).fit(self.data)
            end = time.time()
            duration = round(end - start, 2)
            logger.info(
                f'Time to fit KMeans model with k={k} clusters: {duration} seconds'
            )
            models.append(model)

            # sum of squared errors, which we can't use to determine k because it isn't normalized
            # and would be hard define a reasonable threshold for
            sse[k] = model.inertia_

            # TODO: move these centroids to a ndarray instead of a python list
            # may get some speedup
            # cluster centroids
            centroids.append(model.cluster_centers_)

        model_end = time.time()
        model_duration = round(model_end - model_start, 2)
        logger.info('Total time to fit N={} KMeans models: {} seconds'.format(
            len(experimental_ks), model_duration))

        k, var_explained = self._distance_metrics(experimental_ks, centroids)

        self.k = k
        self.model = models[experimental_ks.index(k)]

    def _distance_metrics(self, experimental_ks, centroids):
        logger.info(
            'Computing distance metrics to search for optimal k in KMeans')
        distance_start = time.time()
        # computes the euclidean distance from every data point to every every cluster center
        # input has shapes:
        # shape(A)=(data_samples, dimensions)
        # shape(B)=(experimental_ks (i.e. k), dimensions)
        # shape(output)=(data_samples, experimental_ks)
        # shape(distance_to_centroids)=(number_of_experiments, data_samples, experimental_ks)
        distance_to_centroids = [
            cdist(self.data, cent, 'euclidean') for cent in centroids
        ]

        # computes the cluster centroid that is closest (not necessary here)
        # closest_cluster_idx = [np.argmin(d, axis=1) for d in distance_to_centroids]

        # computes the distance to the closest cluster center
        # shape(dist)=(number_of_experiments, data_samples)
        dist = [np.min(d, axis=1) for d in distance_to_centroids]

        # explained variance, total within-cluster sum of squares
        within_clus_ss = [np.sum(d**2) for d in dist]

        # TODO: find a way around this O(N^2) operation
        # calculate pairwise distance between all data points (this is the bottlneck in the whole program)
        # we divide by the number of data points because we've overcounted by a factor of N
        # since we're comparing this to the within_clus_ss, which only compares each data point to its
        # centroid once, while the pairwise calculation in tot_ss treats every other data point as a
        # centroid, hence the overcounting
        tot_ss = np.sum(pdist(self.data)**
                        2) / self.data.shape[0]  # total sum of squares

        # ratio of explained variance to total variance
        var_explained_by_clus = (1 - (within_clus_ss / tot_ss)) * 100

        # persist experimental_ks with explained_variance
        self.var_explained_by_clus = []
        for idx, k in enumerate(experimental_ks):
            self.var_explained_by_clus.append((k, var_explained_by_clus[idx]))

        # find cluster count, k, that satifies the self.var_explained threshold
        k_var_explained_tuple = tuple()
        for k, var_explained in self.var_explained_by_clus:
            if var_explained > self.var_explained:
                k_var_explained_tuple = (k, var_explained)
                break

        # if var_explained can't be met, use the max available number of clusters and warn the users
        if not k_var_explained_tuple:
            k_var_explained_tuple = self.var_explained_by_clus[-1]
            logger.warning(
                f'var_explained: {self.var_explained} not satisfied. Defaulting to cluster size k={k_var_explained_tuple[0]} with variance explained: {k_var_explained_tuple[1]}%'
            )

        # clus_count_idx = np.argmax(self.var_explained_by_clus>self.var_explained)

        distance_end = time.time()
        distance_duration = round(distance_end - distance_start, 2)
        logger.info(
            'Total time to compute distance metrics for optimal k in KMeans: {} seconds'
            .format(distance_duration))

        return k_var_explained_tuple

    def plot_elbows(self, directory='results'):
        # sum of squared errors plot
        # plt.figure()
        # plt.plot(sorted(list(self.sse.keys())), sorted(list(self.sse.values()),reverse=True))
        # plt.xlabel("Number of clusters")
        # plt.ylabel("SSE")
        # plt.show()

        # variance explained plot
        plt.figure()
        num_clusters, var_explained = zip(*self.var_explained_by_clus)
        plt.plot(num_clusters, var_explained)
        plt.xlabel("Number of clusters")
        plt.ylabel("Fraction of Variance Explained")
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, 'kmeans_variance_explained.png')
        plt.savefig(file_path, bbox_inches='tight')

    def save_model(self, directory='results'):
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, 'kmeans_model.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def labels(self):
        return self.model.labels_

    def get_centroid_data_points(self):
        pass


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data

    cluster = Cluster(X)
    cluster.find_k_fit(k_max=10, k_step=1, var_explained=80)

    assert cluster.k == 3

    cluster.plot_elbows()