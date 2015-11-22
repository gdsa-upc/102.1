# -*- coding: utf-8 -*-
import os
import pickle
import random

def rank(features_path,save_path,features_train,val_or_test):
    featuresfile = open(features_path+'/features_'+val_or_test+'.p','r')
    train_featuresfile = open(features_train,'r')
    rankfiles = pickle.load(featuresfile)
    train = pickle.load(train_featuresfile)
    for k in rankfiles.keys(): #per cada clau del diccionari dels vectors de caracteristiques de validació
                                #o test ens crearà un fitxer .txt guardat a la seva carpeta corresponent,
                                #el qual tindrà el rànking aleatori de les claus del diccionari d'entrenament
        outfile = open(save_path+'/ranking_'+val_or_test+'/'+k+'.txt','w')
        for k in train.keys():
            outfile.write(random.choice(train.keys())+"\n")
        outfile.close()
    featuresfile.close()
    train_featuresfile.close()

ruta = os.path.dirname(os.path.abspath(__file__)) #ruta absoluta del projecte
rank(ruta+'/files',ruta+'/files',ruta+'/files/features_train.p',"val")