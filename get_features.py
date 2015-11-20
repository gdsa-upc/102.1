# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte
from itertools import islice
ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

def getfeatures(val_or_train):
    IDs_file = open(ruta+"/outfile_"+val_or_train+".txt", 'r') #obrim l'arxiu que conté les ids de les imatges
    features_file = open(ruta + "/features_"+val_or_train+".txt",'w') #obrim l'arxiu en el que escriurem les caracteristiques
    for line in IDs_file:
        features = np.random.rand(1,100)#Generem el vector de caracteristiques aleatori
        final = line.index("\n") #obtenim la posició del salt de línia
        features_file.write(line[0:final] + "\t" + str(features).replace("\n","").replace("[[","").replace("]]","") + "\n") #Escrbim la id de la imatge i el vector de caracteristiques
    features_file.close() #tanquem l'arxiu que conté les caracteristiques
    IDs_file.close() #tanquem l'arxiu que conté només les ids de les imatges

getfeatures("train")
getfeatures("val")