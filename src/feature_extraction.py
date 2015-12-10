# -*- coding: utf-8 -*-
from functions import *
import os
import pickle

def get_descriptors(path): #funció on obtenim tots els descriptors de les imatges de un directori
    nfiles = os.listdir(path) #llistem tots els arxius del directori
    dsc_all = [] #vector on dessem tots els descriptors de totes les imatges
    dsc_ind = {} #vector on dessem els descriptors de cada imatge associats amb la seva id
    for file in nfiles:
        filename = file[0:file.index(".")] #obtenim el nom de l'arxiu
        dsc_ind[filename] = get_local_features(path + "/" + file) #dessem els descriptors de la imatge a un vector amb la seva id
        for feat in dsc_ind[filename]:
            dsc_all.append(feat) #fiquem tots els descriptors en el vector que conté tots els descriptors de totes les imatges
    return dsc_all,dsc_ind #retornem tots els descriptors junts i els descriptors corresponents a cada imatge

def train(dsc,nclusters): #funció per entrenar el codebook
    centroides,_ = train_codebook(nclusters,dsc) 
    return centroides #retornem els centroides

def save_bow(centroides,dsc,val_or_train,nclusters):
    bow = dict() #inicialitzem el bow
    for l in dsc:
        assig = compute_assignments(centroides,dsc[l]) #obtenim els asignaments amb els descriptors d'entrada que li passem
        bow[l] = construct_bow_vector(assig,nclusters) #construim el bow corresponent a cada imatge
    bow_file = open("../files/bow_" + val_or_train + ".p",'w')
    pickle.dump(bow,bow_file) #escribim el bow a el diccionari
    bow_file.close()

if __name__ == "__main__":
    dsc_all_train, dsc_ind_train = get_descriptors("../TerrassaBuildings900/train/images") #obtenim els descriptors de les imatges d'entrenament
    nclusters = 700 #fixem el número de clusters
    centroides = train(dsc_all_train,nclusters) #calculem els centroides del codebook
    save_bow(centroides,dsc_ind_train,"train",nclusters) #calculem i dessem els bow de les imatges d'entrenament
    _, dsc_ind_val = get_descriptors("../TerrassaBuildings900/val/images") #obtenim els descriptors de validació
    save_bow(centroides,dsc_ind_val,"val",nclusters) #guardem els bow de les imatges de validació