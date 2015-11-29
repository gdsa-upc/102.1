    #-*- coding: utf-8 -*-    
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,whiten
from train_kmeans import train_codebook
from get_local_features import get_local_features

ruta = os.path.dirname(os.path.abspath(__file__)) # Definim la instrucció principal que busca la ruta absoluta del fitxer

def compute_assignments(codebook,desc):
    #Paràmetres de la funcio: el codebook amb les centroides trobats i els decriptors
    norm_desc = whiten(desc) # Normaliza descriptores
    return vq(norm_desc, codebook) # la funció vq elabora el vector d'assignacions i retorna el vector d'assignacions
    
nfiles_t = os.listdir(ruta+"/TerrassaBuildings900/train/images")
nfiles_v = os.listdir(ruta+"/TerrassaBuildings900/val/images")
descriptors = [] #Declarem el vector de descriptors
assig = [] #Declarem el vector d'assignacions

for file in nfiles_t: 
   dscrp = get_local_features("/TerrassaBuildings900/train/images"+file)
   descriptors.append(dscrp)
   centroide,_ = train_codebook(13,descriptors)
assig = compute_assignments(centroide,_,dscrp)

for file in nfiles_v:
   dscrp = get_local_features("/TerrassaBuildings900/val/images"+file)
   descriptors.append(dscrp)
   centroide,_ = train_codebook(13,descriptors)
assig = compute_assignments(centroide,_,dscrp)

#A continuació representarem la grafica amb els descriptors i els centroides mrcats amb color vermell
plt.scatter(descriptors[:,0],descriptors[:,1]),plt.scatter(centroide[:,0],centroide[:,1],color = 'r'),plt.show()
#Mes tard, mostrem per pantalla el vector d'assignacions separats amb comes
print", ".join(assig)
