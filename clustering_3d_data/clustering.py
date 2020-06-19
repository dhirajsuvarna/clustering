
# Perform clustering

from sklearn import cluster
from sklearn.externals import joblib
import os

# Achieving Abstract Classes and Methods
# https://stackoverflow.com/questions/44576167/force-child-class-to-override-parents-methods/44576235

def get_clusteringAlgo(iAlgo):
    clusteringAlgo = None 
    if iAlgo == "meanshift":
        clusteringAlgo = MeanShift()
    elif iAlgo == "dbscan":
        clusteringAlgo = DBScan()
    elif iAlgo == "kmeans":
        clusteringAlgo = KMeans()
    else:
        ValueError(iAlgo)
    
    return clusteringAlgo
        

DEFAULT_PATH = 'saved_models'

class BaseCluster:
    def __init__(self, name):
        self.name = name
        self.algo = None
    
    def performClustering(self, iData):
        raise NotImplementedError

    def save(self, iDir = None):
        iDir = DEFAULT_PATH if iDir is None else iDir
        filePath = os.path.join(iDir, self.name + '.pkl')
        joblib.dump(self.algo, filePath)

class MeanShift(BaseCluster):
    def __init__(self):
        super().__init__("MeanShift")

    def performClustering(self, iData):
        bandwidth = cluster.estimate_bandwidth(iData)
        self.algo = cluster.MeanShift(bandwidth=bandwidth).fit(iData)

class DBScan(BaseCluster):        
    def __init__(self):
        super().__init__("DBScan")

    def performClustering(self, iData):
        self.algo = cluster.DBSCAN(eps=1.0, min_samples=10).fit(iData)

class KMeans(BaseCluster):
    def __init__(self):
        super().__init__("KMeans")

    def performClustering(self, iData, iNumCluster):
        self.algo = cluster.KMeans(init='k-means++', n_clusters=iNumCluster, n_init=10).fit(iData)



#Test code 
import torch 
if __name__ == "__main__":
    algo = "dbscan"

    algo = get_clusteringAlgo(algo)

    data = torch.Tensor()
    algo.performClustering(data)
