# -*- coding: utf-8 -*-
import os
import pickle #carreguem la llibreria pickle per poder treballar amb els diccionaris
import random #carreguem la llibreria que utilitzarem per fer el rànking aleatori de cada fitxer.
import numpy as np

def rank(features_path,save_path,features_train,val_or_test,annotation):
    out = []
    featuresfile = open(features_path+'/features_'+val_or_test+'.p','r') #obrim el diccionari de vectors de característiques de validació o de test
    train_featuresfile = open(features_train,'r') #obrim el diccionari de vectors de característiques d'entrenament
    annot = open(annotation+'/'+val_or_test+'/annotation.txt','r') #obrim el fixer anotacions del conjunt de validacio o de test
    desconegut = [] #Creem el vector desconegut en el qual guardarem totes les ids del conjunt de validació o test que tinguin categoria desconegut.
    for line in annot:
        tab = line.index("\t")
        fin = len(line)
        if str(line[tab+1:fin-1]) == "desconegut":
            desconegut.append(line[0:tab]) #Entrem totes les ids les quals la seva categoria sigui desconegut.
    rankfiles = pickle.load(featuresfile) #Carreguem el diccionari validació o test
    train = pickle.load(train_featuresfile) #carreguem el diccionari entrenament
    for k in rankfiles.keys(): #per cada clau del diccionari dels vectors de caracteristiques de validació
                                #o test ens crearà un fitxer .txt guardat a la seva carpeta corresponent,
                                #el qual tindrà el rànking aleatori de les claus del diccionari d'entrenament
        if desconegut.count(k) == 0: #Si la id (key del diccionari de validacio o test) no apareix en el vector desconeguts, que ens calculi el rànking.
            outfile = open(save_path+'/ranking_'+val_or_test+'/'+k+'.txt','w')
            for k in train.keys():
                out.insert(np.random.randint(0,451),k)
            for item in out:
                outfile.write("%s\n" % item)
            out = []
            outfile.close()
            # Finalment, totes les ids (key del diccionari de validacio o test) que pertenyen a la classe desconegut seran ignorades a l'hora de crear els rànkings
    featuresfile.close()
    train_featuresfile.close()
    annot.close()

ruta = os.path.dirname(os.path.abspath(__file__)) #ruta absoluta del projecte
rank(ruta+'/files',ruta+'/files',ruta+'/files/features_train.p',"val",ruta+'/TerrassaBuildings900') #crida a la funció rank pel diccionari de validació