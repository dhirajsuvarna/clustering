# file to try out different clustering on sample data


# latent_point = latent_vector[1].detach().numpy()
# filename_map = dict()
# filename_map.update({'filename' : latent_point})
# torch.save(filename_map, 'test.pth')
# load_file_map = torch.load('test.pth')

from sklearn import cluster
import torch
import numpy as np
import timeit

latent_vector_file = 'saved_models/best_latent_vector_1.pth'

# Load the clusters
latent_vector = torch.load(latent_vector_file).detach()

###################################################################################################
# Train K-Means
n_clusters = 2
kmeans = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=10).fit(latent_vector)

print(f"#KMeans Clusters : {np.unique(kmeans.labels_)}")
# print(f"Clusters : {kmeans.labels_}")
# print(f"KMeans Cluster Centroids : {kmeans.cluster_centers_}")

###################################################################################################
# Train Mean Shift 
bandwidth = cluster.estimate_bandwidth(latent_vector)
meanshift = cluster.MeanShift(bandwidth=bandwidth).fit(latent_vector)
print(f"#MeanShift Clusters : {np.unique(meanshift.labels_)}")
# print(f"Clusters : {meanshift.labels_}")
# print(f"MeanShift Cluster Centroids : {meanshift.cluster_centers_}")

# to do - bin_seeding = True && estimate_bandwidth with quantile = 0.3

###################################################################################################
# Affinity Propogation 
affinity = cluster.AffinityPropagation(preference=-750).fit(latent_vector)
print(f"#Affinity Clusters : {np.unique(affinity.labels_)}")
# print(f"Clusters : {affinity.labels_}")
# print(f"Affinity Cluster Centroids : {affinity.cluster_centers_}")


###################################################################################################
# DBSCAN 
db = cluster.DBSCAN(eps=1.0, min_samples=10).fit(latent_vector)
print(f"#DBSCAN Clusters : {np.unique(db.labels_)}")
# print(f"Clusters : {db.labels_}")
# print(f"DBSCAN Cluster Centroids : {db.cluster_centers_}")
