#-*- coding: utf-8 -*-    import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
from sklearn.preprocessing import normalize
from train_kmeans import train_codebook
from get_local_features import get_local_features
from compute_assignments import get_assignments

def construct_bow_vector(assignments,train_codebook,id_image,val_or_train):
    #Paràmetres de la funcio: els assignments realitzadors sobre els descriptors, el codebook amb les centroides trobats, 
    # els decriptors, les ids de les imatges i si l'indicador de si la carpeta 
    #on estem es la de validacio o la d'entrenament. 
    return normalize(assignments,train_codebook,id_image,val_or_train)
ruta = os.path.dirname(os.path.abspath(__file__)) 
nfiles_t = os.listdir(ruta+"/TerrassaBuildings900/train/images")
nfiles_v = os.listdir(ruta+"/TerrassaBuildings900/val/images")
descriptors = []
assignments = []
BoW = []

for file in nfiles_t:
    dscrp = get_local_features("/TerrassaBuildings900/train/images"+file)
    descriptors.append(dscrp)
    centroide,_ = train_codebook(13,descriptors)
    assig = get_assignments(centroide,_,dscrp,file,"train")
    assignments.append(assig)
norm = construct_bow_vector(assig,centroide,_,dscrp,file,"train")
BoW.append(norm)

for file in nfiles_v:
    dscrp = get_local_features("/TerrassaBuildings900/val/images"+file)
    descriptors.append(dscrp)
    centroide,_ = train_codebook(13,descriptors)
    assig = get_assignments(centroide,_,dscrp,file,"val")
    assignments.append(assig)
norm = construct_bow_vector(assig,centroide,_,dscrp,file,"val")
BoW.append(norm)
