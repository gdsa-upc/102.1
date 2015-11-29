    #-*- coding: utf-8 -*-    
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import predict
from train_kmeans import train_codebook
from get_local_features import get_local_features

ruta = os.path.dirname(os.path.abspath(__file__)) # Definim la instrucció principal que busca la ruta absoluta del fitxer

def compute_assignments(codebook,desc,id_image,n_cluster):
    #Paràmetres de la funcio: el codebook amb les centroides trobats, els decriptors, les ids de les imatges i si l'indicador de si la carpeta 
    #on estem es la de validacio o la d'entrenament.
    TrainImages = open(ruta+"/TerrassaBuildings900/train/images",'r')
    ValImages = open(ruta+"/TerrassaBuildings900/val/images",'r')
    assignments = [] #Declaració del vector d'assignacions que realitzarà
    for file in TrainImages: #En aquest for omplim la primera part del vector d'assignacions per les imatges d'entrenament 
        for desc in file:   #en funcio de en quin cluster es trobi el descriptor gràcies a 'predict'. 
            pred = predict(13,desc)
            assignments.append(pred)
    TrainImages.close()
    for file in ValImages: #En aquest for omplim la primera part del vector d'assignacions per les imatges de validacio
        for desc in file:  #en funcio de en quin cluster es trobi el descriptor gràcies a 'predict'. 
            pred = predict(13,desc)
            assignments.append(pred)
    ValImages.close()
    return assignments #Retorna el vector d'assignacions
    
nfiles_t = os.listdir(ruta+"/TerrassaBuildings900/train/images")
nfiles_v = os.listdir(ruta+"/TerrassaBuildings900/val/images")
descriptors = [] #Declarem el vector de descriptors
assig = [] #Declarem el vector d'assignacions

for file in nfiles_t: 
   dscrp = get_local_features("/TerrassaBuildings900/train/images"+file)
   descriptors.append(dscrp)
   centroide,_ = train_codebook(13,descriptors)
assig = compute_assignments(centroide,_,dscrp,file,"train")

for file in nfiles_v:
   dscrp = get_local_features("/TerrassaBuildings900/val/images"+file)
   descriptors.append(dscrp)
   centroide,_ = train_codebook(13,descriptors)
assig = compute_assignments(centroide,_,dscrp,file,"val")

#A continuació representarem la grafica amb els descriptors i els centroides mrcats amb color vermell
plt.scatter(descriptors[:,0],descriptors[:,1]),plt.scatter(centroide[:,0],centroide[:,1],color = 'r'),plt.show()
#Mes tard, mostrem per pantalla el vector d'assignacions separats amb comes
print", ".join(assig)
