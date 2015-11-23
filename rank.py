# -*- coding: utf-8 -*-
import os
import pickle #carreguem la llibreria pickle per poder treballar amb els diccionaris
import random #carreguem la llibreria que utilitzarem per fer el rànking aleatori de cada fitxer.
import numpy as np

def rank(features_path,save_path,features_train,val_or_test):
    out = []
    featuresfile = open(features_path+'/features_'+val_or_test+'.p','r') #obrim el diccionari de vectors de característiques de validació o de test
    train_featuresfile = open(features_train,'r') #obrim el diccionari de vectors de característiques d'entrenament
    rankfiles = pickle.load(featuresfile) #carreguem el diccionari validació o test
    train = pickle.load(train_featuresfile) #carreguem el diccionari entrenament
    for k in rankfiles.keys(): #per cada clau del diccionari dels vectors de caracteristiques de validació
                                #o test ens crearà un fitxer .txt guardat a la seva carpeta corresponent,
                                #el qual tindrà el rànking aleatori de les claus del diccionari d'entrenament
        outfile = open(save_path+'/ranking_'+val_or_test+'/'+k+'.txt','w')
        for k in train.keys():
            out.insert(np.random.randint(0,451),k)
        for item in out:
            outfile.write("%s\n" % item)
        out = []
        outfile.close()
    featuresfile.close()
    train_featuresfile.close()

ruta = os.path.dirname(os.path.abspath(__file__)) #ruta absoluta del projecte
rank(ruta+'/files',ruta+'/files',ruta+'/files/features_train.p',"val") #crida a la funció rank pel diccionari de validació