# -*- coding: utf-8 -*-
import numpy as np
#import cv2
#import matplotlib.pyplot as plt
import pickle
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte
#from itertools import islice
ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

def getfeatures(val_or_train):
    IDs_file = open(ruta+"/files/outfile_"+val_or_train+".txt", 'r') #obrim l'arxiu que conté les ids de les imatges
    features_file = open(ruta + "/files/features_"+val_or_train+".p",'w') #obrim l'arxiu en el que escriurem les caracteristiques
    feat_vec = dict() #inicialitzem el diccionari buit
    for line in IDs_file:
        features = np.random.rand(1,100)#Generem el vector de caracteristiques aleatori
        final = line.index("\n") #obtenim la posició del salt de línia
        feat_vec[line[0:final]] = features #afegim el vector de caracteristiques aleatori al diccionari
    IDs_file.close() #tanquem l'arxiu que conté només les ids de les imatges
    pickle.dump(feat_vec, features_file) #escribim el diccionari amb pickle
    features_file.close() #tanquem l'arxiu del diccionari
getfeatures("train") #cridem a la funcio en mode train
getfeatures("val") #cridem a la funció en mode val
