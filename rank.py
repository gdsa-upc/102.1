# -*- coding: utf-8 -*-
import os
import pickle #carreguem la llibreria pickle per poder treballar amb els diccionaris
import random #carreguem la llibreria que utilitzarem per fer el rànking aleatori de cada fitxer.

def rank(features_path,save_path,features_train,val_or_test):
    featuresfile = open(features_path+'/features_'+val_or_test+'.p','r') #obrim el diccionari de vectors de característiques de validació o de test
    train_featuresfile = open(features_train,'r') #obrim el diccionari de vectors de característiques d'entrenament
    rankfiles = pickle.load(featuresfile) #carreguem el diccionari validació o test
    train = pickle.load(train_featuresfile) #carreguem el diccionari entrenament
    for k in rankfiles.keys(): #per cada clau del diccionari dels vectors de caracteristiques de validació
                                #o test ens crearà un fitxer .txt guardat a la seva carpeta corresponent,
                                #el qual tindrà el rànking aleatori de les claus del diccionari d'entrenament
        outfile = open(save_path+'/ranking_'+val_or_test+'/'+k+'.txt','w') #obrim un fitxer per cada clau del diccionari
        ranking = open(save_path+'/ranking_'+val_or_test+'/ranking.txt','w') #obrim un fitxer amb el qual anirem controlant que quan elegim una id d'entrenament aleatoriament,
                                                                            #no es torni a repetir en la mateixa clau de validació o test.
                                                                    # Al final del programa s'hauran obert i esborrat tants fitxers ranking com el número de claus del diccionari de validació o entrenament.
        for k in train.keys():
            ranking.write(k+"\n") #Escribim en el fitxer ranking les claus del diccionari d'entrenament
        ranking.close()
        for k in train.keys():
            ran = random.choice(open(save_path+'/ranking_'+val_or_test+'/ranking.txt').readlines()) #elegim una id del fitxer ranking aleatoriament
            outfile.write(ran) #escribim aquesta id aleatoria al fitxer de sortida de la clau de validacio o test que estem treballant
            ranking = open(save_path+'/ranking_'+val_or_test+'/ranking.txt','r')
            for linea in ranking:
                if ran in linea:
                    linea.replace(ran,'') #esborrem la linea on es troba la id escollida aleatoriament per assegurarnos que nos es torni a repetir en la clau que estem treballant
            ranking.close() #tanquem fitxer de control
            
        outfile.close() #tanquem fitxer de sortida
        os.remove(save_path+'/ranking_'+val_or_test+'/ranking.txt')# esborrem el fitxer de control per la clau en la qual estem treballant
    featuresfile.close() #tanquem diccionari validació o test
    train_featuresfile.close() #tanquem diccionari d'entrenament

ruta = os.path.dirname(os.path.abspath(__file__)) #ruta absoluta del projecte
rank(ruta+'/files',ruta+'/files',ruta+'/files/features_train.p',"val") #crida a la funció rank pel diccionari de validació