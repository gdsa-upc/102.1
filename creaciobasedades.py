# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte
from itertools import islice

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

""" TRAIN IMAGES """
infile_train = open(ruta+'/TerrassaBuildings900/train/annotation.txt','r')
outfile_train = open(ruta+'/outfile_train.txt','w') # Creem un fitxer per guardar les id's de train
it = islice(infile_train,1,None) #Salta la primera linia del fitxer .txt
for line in it:
    outfile_train.write(line.strip("\t")) # Assignem cada linea a una entrada de l'array

""" VALIDATION IMAGES """
