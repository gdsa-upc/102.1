# -*- coding: utf-8 -*-
import os
import get_local_features as GET
import train_codebook as TRA
import compute_assignments as COMP
import construct_bow_vector as CONS
import numpy as np
import pickle
from scipy.cluster.vq import *
import matplotlib.pyplot as plt

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta del l'arxiu

#Entrenem el codebook i fem crides a get_local_features
nfiles_t = os.listdir(ruta +"/.."+ '/TerrassaBuildings900/train/images') #llistem tots els arxius de la carpeta """/TerrassaBuildings900/train/images"""
descriptores_train = [] #inicialitzem el vector codebook
for file in nfiles_t:
    ds = GET.get_local_features('/TerrassaBuildings900/train/images/'+file) #obtenim els descriptors de cada imatge """/TerrassaBuildings900/train/images/"""
    for feat in ds:
        descriptores_train.append(feat) #guardem tots els descriptors de totes les imatges
centroides,_ = TRA.train_codebook(100,descriptores_train) #calculem els centroides de les imatges
plt.scatter(descriptores_train[0],descriptores_train[1]),plt.scatter(centroides[:,0],centroides[:,1],color = 'r'),plt.show()

#Compute Assigments
assig_train = [] #Declarem el vector d'assignacions de train
assig_train = COMP.compute_assignments(centroides,descriptores_train)
nfiles_v = os.listdir(ruta +"/.."+ '/TerrassaBuildings900/val/images') #"""/TerrassaBuildings900/val/images"""
descriptor_val = []
for file in nfiles_v:
    val = GET.get_local_features('/TerrassaBuildings900/val/images/'+file)
    for feat in val:
        descriptor_val.append(feat) #guardem tots els descriptors de totes les imatges
assig_val = [] #Declarem el vector d'assignacions
assig_val = COMP.compute_assignments(centroides,descriptor_val)

#Construïm els vectors BoW
BoW_train = CONS.construct_bow_vector(assig_train)
BoW_val = CONS.construct_bow_vector(assig_val)