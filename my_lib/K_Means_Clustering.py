import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Kmeans:
    def __init__(self):
        self.centroids=None
        self.assigned_indexes=None

    def initialize_centroid(self,X,k):
        centroids=[]
        n_samples=X.shape[0]
        centroids.append(X[np.random.randint(n_samples)])
        for _ in range(1,k):
            dist=np.min([np.sum((X-centroid)**2,axis=1) for centroid in centroids], axis=0)
            prob=dist**2/np.sum(dist**2)
            centroids.append(X[np.random.choice(n_samples, p=prob)])
        centroids=np.array(centroids)
        return centroids
    
    def assign_centroid(self,X,centroids):
        assigned=np.zeros(X.shape[0])
        for i,point in enumerate(X):
            dists=np.sqrt(np.sum((point-centroids)**2,axis=1))
            assigned[i]=np.argmin(dists)
        return assigned
    
    def compute_new_centroids(self,X,assigned_centroid,k):
        centroids=np.zeros((k,X.shape[1]))
        for i in range (k):
            points = X[assigned_centroid==i]  
            centroids[i]=np.mean(points, axis=0)
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
            centroids=self.compute_new_centroids(X,assigned_centroids,k)
        if plot:
            self.plot_graph(X,centroids, assigned_centroids)
        self.centroids=centroids
        self.assigned_centroids=assigned_centroids
        return

