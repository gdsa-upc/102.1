# -*- coding: utf-8 -*-
from functions import *
import os
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances
import time 

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

def rank(features_path,save_path,features_train,val_or_test,annotation):
    out = []
    ordenada = []
    bow_train = 0
    featuresfile = open(features_path+'/bow_'+val_or_test+'.p','r') #obrim el diccionari de vectors de característiques de validació o de test
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
    entrenament = train.keys() #afegim totes les ids segons les tenim ordenades en el diccionari en el vector entrenament
    for k in rankfiles.keys(): #per cada clau del diccionari dels vectors de caracteristiques de validació
                                #o test ens crearà un fitxer .txt guardat a la seva carpeta corresponent,
                                #el qual tindrà el rànking aleatori de les claus del diccionari d'entrenament
        if desconegut.count(k) == 0: #Si la id (key del diccionari de validacio o test) no apareix en el vector desconeguts, que ens calculi el rànking.
            outfile = open(save_path+'/ranking_'+val_or_test+'/'+k+'.txt','w')
            bow = rankfiles[k] #treiem el vector de caracteristiques de cada id de validació o de test 
            for k2 in train.keys():
                bow_train = train[k2] #treiem el vector de caracteristiques de cada id d'entrenament
                dist = pairwise_distances(bow,bow_train,metric='euclidean',n_jobs=1) #calculem les distancies euclidees entre el Bow de validació/test i els BOW d'entrenament
                out.append(dist) #Guardem les distancies en out (cada distància esta guardada en aquest vector que la seva corresponent id en el vector d'entrenament)
                bow_train = 0
            ordenada = sorted(out) #Ordenem la llista out de menys a més distància a la imatge que estem estudiant. Ho guardem a l'array ordenada.
            for item in ordenada:
                position = out.index(item) #busquem la posicio en el vector out de les distancies ordenades
                outfile.write(entrenament[position]+"\n") #el fitxer de sortida marcarà les ids que més s'aproximin (en distàncies) en ordre.
                out[position] = 'f'
            out = []
            ordenada = []
            outfile.close()
            # Finalment, totes les ids (key del diccionari de validacio o test) que pertenyen a la classe desconegut seran ignorades a l'hora de crear els rànkings
    featuresfile.close()
    train_featuresfile.close()
    annot.close()

def evaluate_rank(dir_rank):
    nfiles = os.listdir(dir_rank)
    ground_truth_val = open("../TerrassaBuildings900/val/annotation.txt", "r")
    ground_truth_train = open("../TerrassaBuildings900/train/annotation.txt","r")
    truth = {} #inicialitzem una taula on l'index es la id de la imatge i conté la seva categoria
    AP = {}
    next(ground_truth_val)#eliminem la primera linia de l'arxiu ja que no ens interessa
    next(ground_truth_train)#eliminem la primera linia de l'arxiu ja que no ens interessa
    for line in ground_truth_val:
        id_foto = line.index("\t")
        final = line.index("\n")
        truth[line[0:id_foto]] = line[id_foto+1:final] #guardem la categoria de cada imatge a un vector
    for line in ground_truth_train:
        id_foto = line.index("\t")
        final = line.index("\n")
        truth[line[0:id_foto]] = line[id_foto+1:final] #guardem la categoria de cada imatge a un vector
    MAP = 0
    APC = 0
    for file in nfiles:
        ranking = open(dir_rank+"/"+file,"r")#obrim l'arxiu rank d'una imatge de cerca
        filename = file[0:file.index(".")]
        categoria = truth[filename] #assignem la categoria que té la imatge de cerca
        relevants = 0
        precision = 0
        AP[filename] = 0
        irrelevants = 0
        k = 0
        for line in ranking:
            final = line.index("\n")
            k += 1
            if truth[line[0:final]] == categoria: #si la id de la imatge coincideix amb la categoria sumem + 1 a relevants
                relevants += 1 
                precision = precision + (float(relevants)/float(k)) #calculem la precisio per cada k
            else:
                irrelevants += 1
        AP[filename] = float(precision)/float(relevants) #calculem la AP de cada imatge de cerca
        APC += AP[filename]  #calculem la AP acumulada de cada imatge de cerca
        ranking.close()
    MAP = APC/len(nfiles) #calcul del MAN
    return AP, MAP #retornem els valors de AP de cada imatge i de MAN
    
if __name__ == "__main__":
    dsc_all_train = pickle.load(open("../files/dsc_all_train.p",'rb'))
    dsc_ind_train = pickle.load(open("../files/dsc_ind_train.p",'rb'))
    dsc_ind_val = pickle.load(open("../files/dsc_ind_val.p",'rb'))
    nclusters = [800,1000,1200,1400,1600]
    graficas = open("../files/graficas.txt",'w')
    for i in nclusters:
        t = time.time()
        centroides = train(dsc_all_train,i) #calculem els centroides del codebook
        save_bow(centroides,dsc_ind_train,"train",i) #calculem i dessem els bow de les imatges d'entrenament
        save_bow(centroides,dsc_ind_val,"val",i) #guardem els bow de les imatges de validació
        rank('../files','../files','../files/bow_train.p',"val",'../TerrassaBuildings900') #crida a la funció rank pel diccionari de validació
        _,MAP = evaluate_rank("../files/ranking_val")
        graficas.write(str(i) + "\t" + str(MAP) + "\t" + str(time.time()-t) + "\n")
        print i
    graficas.close()