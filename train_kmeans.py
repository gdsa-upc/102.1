# -*- coding: utf-8 -*-
from get_local_features import get_local_features
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq

def train_codebook(nclusters,normalized_descriptors):
    return kmeans(normalized_descriptors,nclusters)#obtenim els clusters de les imatges
    
des = get_local_features("/imagen_primerscript/tiger.jpg")
centroide,_ = train_codebook(2,des)

plt.scatter(des[:,0],des[:,1]),plt.scatter(centroide[:,0],centroide[:,1], color ='r'),plt.show()