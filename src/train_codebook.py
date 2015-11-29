# -*- coding: utf-8 -*-
from get_local_features import get_local_features
import matplotlib.pyplot as plt
import os
from scipy.cluster.vq import kmeans,vq
import numpy as np

def train_codebook(nclusters,normalized_descriptors):
    return kmeans(normalized_descriptors,nclusters) #obtenim els centroides de les imatges
'''
ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta del l'arxiu
nfiles = os.listdir(ruta + "/TerrassaBuildings900/train/images") #llistem tots els arxius de la carpeta 
descriptors = [] #inicialitzem el vector descriptors
for file in nfiles:
    ds = get_local_features("/TerrassaBuildings900/train/images/"+file) #obtenim els descriptors de cada imatge
    for feat in ds:
        descriptors.append(feat) #guardem tots els descriptors de totes les imatges
centroides,_ = train_codebook(12,descriptors) #calculem els centroides de les imatges
#plt.scatter(centroides[:,0],centroides[:,1]), plt.show()
'''