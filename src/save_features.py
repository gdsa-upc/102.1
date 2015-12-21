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


if __name__ == "__main__":
    dsc_all_train, dsc_ind_train = get_descriptors("../TerrassaBuildings900/train/images") #obtenim els descriptors de les imatges d'entrenament
    pickle.dump(dsc_all_train,open("../files/dsc_all_train.p",'wb'))
    pickle.dump(dsc_ind_train, open("../files/dsc_ind_train.p",'wb'))
    _, dsc_ind_val = get_descriptors("../TerrassaBuildings900/val/images") #obtenim els descriptors de validació
    pickle.dump(dsc_ind_val,open("../files/dsc_ind_val.p",'wb'))