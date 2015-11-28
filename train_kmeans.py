# -*- coding: utf-8 -*-
from get_local_features import get_local_features
import matplotlib.pyplot as plt
import os
from scipy.cluster.vq import kmeans,vq
import numpy as np

def train_codebook(nclusters,normalized_descriptors):
    return kmeans(normalized_descriptors,nclusters) #obtenim els centroides de les imatges
"""    
des = get_local_features("/imagen_primerscript/tiger.jpg")
centroide,_ = train_codebook(2,des)

plt.scatter(des[:,0],des[:,1]),plt.scatter(centroide[:,0],centroide[:,1], color ='r'),plt.show()
"""
ruta = os.path.dirname(os.path.abspath(__file__)) 
nfiles = os.listdir(ruta + "/TerrassaBuildings900/train/images")
descriptors = []
for file in nfiles:
    ds = get_local_features("/TerrassaBuildings900/train/images/"+file)
    descriptors.append(ds)
centroide,_ = train_codebook(13,descriptors)
plt.scatter(descriptors[:,0],descriptors[:,1]),plt.scatter(centroide[:,0],centroide[:,1], color='r'), plt.show()
    