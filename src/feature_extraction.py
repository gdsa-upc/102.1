# -*- coding: utf-8 -*-
from functions import *
import os
import pickle

def get_descriptors_rootSIFT(path): #funció on obtenim tots els descriptors de les imatges de un directori
    nfiles = os.listdir(path) #llistem tots els arxius del directori
    dsc_all = [] #vector on dessem tots els descriptors de totes les imatges
    dsc_ind = {} #vector on dessem els descriptors de cada imatge associats amb la seva id
    for file in nfiles:
        filename = file[0:file.index(".")] #obtenim el nom de l'arxiu
        dsc_ind[filename] = get_local_features_rootSIFT(path + "/" + file) #dessem els descriptors de la imatge a un vector amb la seva id
        for feat in dsc_ind[filename]:
            dsc_all.append(feat) #fiquem tots els descriptors en el vector que conté tots els descriptors de totes les imatges
    return dsc_all,dsc_ind #retornem tots els descriptors junts i els descriptors corresponents a cada imatge

def get_descriptors_SURF(path): #funció on obtenim tots els descriptors de les imatges de un directori
    nfiles = os.listdir(path) #llistem tots els arxius del directori
    dsc_all = [] #vector on dessem tots els descriptors de totes les imatges
    dsc_ind = {} #vector on dessem els descriptors de cada imatge associats amb la seva id
    for file in nfiles:
        filename = file[0:file.index(".")] #obtenim el nom de l'arxiu
        dsc_ind[filename] = get_local_features_SURF(path + "/" + file) #dessem els descriptors de la imatge a un vector amb la seva id
        for feat in dsc_ind[filename]:
            dsc_all.append(feat) #fiquem tots els descriptors en el vector que conté tots els descriptors de totes les imatges
    return dsc_all,dsc_ind #retornem tots els descriptors junts i els descriptors corresponents a cada imatge

def train(dsc,nclusters): #funció per entrenar el codebook
    centroides,_ = train_codebook(nclusters,dsc) 
    return centroides #retornem els centroides

def save_bow(centroides,dsc,nclusters):
    bow = dict() #inicialitzem el bow
    for l in dsc:
        assig = compute_assignments(centroides,dsc[l]) #obtenim els asignaments amb els descriptors d'entrada que li passem
        bow[l] = construct_bow_vector(assig,nclusters) #construim el bow corresponent a cada imatge
    return bow

if __name__ == "__main__":
    dsc_all_train_rootSIFT, dsc_ind_train_rootSIFT = get_descriptors_rootSIFT("../TerrassaBuildings900/train/images") #obtenim els descriptors de les imatges d'entrenament
    dsc_all_train_SURF, dsc_ind_train_SURF = get_descriptors_SURF("../TerrassaBuildings900/train/images") #obtenim els descriptors de les imatges d'entrenament
    nclusters = 512 #fixem el número de clusters
    centroides_rootSIFT = train(dsc_all_train_rootSIFT,nclusters) #calculem els centroides del codebook
    bow_rootSIFT=save_bow(centroides_rootSIFT,dsc_ind_train_rootSIFT,nclusters) #calculem i dessem els bow de les imatges d'entrenament
    centroides_SURF = train(dsc_all_train_SURF,nclusters) #calculem els centroides del codebook
    bow_SURF=save_bow(centroides_SURF,dsc_ind_train_SURF,nclusters) #calculem i dessem els bow de les imatges d'entrenament
    bows = [bow_rootSIFT,bow_SURF]
    bow_train = dict()
    for k in bow_rootSIFT.iterkeys():
        bow_train[k] = tuple(bow_train[k] for bow_train in bows)
    pickle.dump(bow_train,open("../files/bow_train.p",'wb'))
    _, dsc_ind_val_rootSIFT = get_descriptors_rootSIFT("../TerrassaBuildings900/val/images") #obtenim els descriptors de validació
    _,dsc_ind_val_SURF = get_descriptors_SURF("../TerrassaBuildings900/val/images")
    bow_SURF = save_bow(centroides_SURF,dsc_ind_val_SURF,nclusters) #guardem els bow de les imatges de validació"""
    bow_rootSIFT = save_bow(centroides_rootSIFT,dsc_ind_val_rootSIFT,nclusters) #guardem els bow de les imatges de validació"""
    bows = [bow_rootSIFT,bow_SURF]
    bow_val = dict()
    for k in bow_rootSIFT.iterkeys():
        bow_val[k] = tuple(bow_val[k] for bow_val in bows)
    pickle.dump(bow_train,open("../files/bow_val.p",'wb'))
