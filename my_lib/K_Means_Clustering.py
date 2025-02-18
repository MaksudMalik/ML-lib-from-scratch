import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Kmeans:
    def __init__(self):
        self.centroids=None
        self.assigned_indexes=None

    def initialize_centroid(self,X,k):
        n_samples = X.shape[0]
        centroids = np.zeros((k, X.shape[1]))
        centroids[0] = X[np.random.randint(n_samples)]
        for i in range(1, k):
            dist = np.min(np.linalg.norm(X[:, np.newaxis] - centroids[:i], axis=2), axis=1)
            prob = dist / np.sum(dist)
            centroids[i] = X[np.random.choice(n_samples, p=prob)]
        return centroids
    
    def assign_centroid(self, X, centroids):
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(dists, axis=1)

    
    def compute_new_centroids(self,X,assigned_centroids,k):
        centroids = np.array([X[assigned_centroids == i].mean(axis=0) for i in range(k)])
        return centroids
    
    def plot_graph(self,X,centroids,assigned_centroids):
        dimension=X.shape[1]
        if dimension<4:
            fig=plt.figure(figsize=(10,6))
            if dimension==3:
                ax=fig.add_subplot(111, projection='3d')
                ax.scatter(X[:,0], X[:,1], X[:,2], c=assigned_centroids, edgecolors="k")
                ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='red', marker='X', s=200, label="Final Centroids")

            elif dimension==2:
                ax=fig.add_subplot(111)
                ax.scatter(X[:,0], X[:,1], c=assigned_centroids, edgecolors="k")
                ax.scatter(centroids[:,0], centroids[:,1], c='red', marker='X', s=200, label="Final Centroids")

            else:
                ax=fig.add_subplot(111)
                ax.scatter(X, np.zeros_like(X), c=assigned_centroids, edgecolors="k")
                ax.scatter(centroids, np.zeros_like(centroids), c='red', marker='X', s=200, label="Final Centroids")
            
            ax.set_title("Scatter Plot of Clusters")
            ax.legend()
            plt.show()
        else:
            print("Cannot plot multidimensional data (dimension >= 4)")
        return

    def fit_kmeans(self,X,k,max_iters=15,plot=False):
        initial_centroids=self.initialize_centroid(X,k)
        centroids=initial_centroids
        for i in range (max_iters):
            assigned_centroids=self.assign_centroid(X,centroids)
            new_centroids=self.compute_new_centroids(X,assigned_centroids,k)
            if np.all(centroids==new_centroids):
                break
            centroids=new_centroids
        if plot:
            self.plot_graph(X,centroids, assigned_centroids)
        self.centroids=centroids
        self.assigned_centroids=assigned_centroids
        return

