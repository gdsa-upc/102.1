import os
from src import * as SR
import numpy as np
import pickle
from scipy.cluster.vq import *

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta del l'arxiu

#Entrenem el codebook i fem crides a get_local_features

nfiles = os.listdir(ruta + '/TerrassaBuildings900/train/images') #llistem tots els arxius de la carpeta """/TerrassaBuildings900/train/images"""
descriptors = [] #inicialitzem el vector descriptors
for file in nfiles:
    ds = SR.get_local_features('/TerrassaBuildings900/train/images'+file) #obtenim els descriptors de cada imatge """/TerrassaBuildings900/train/images/"""
    for feat in ds:
        descriptors.append(feat) #guardem tots els descriptors de totes les imatges
centroides,_ = SR.train_codebook(100,descriptors) #calculem els centroides de les imatges
plt.scatter(centroides[:,0],centroides[:,1]), plt.show()