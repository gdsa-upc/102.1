# -*- coding: utf-8 -*-
import numpy as np
import os
from itertools import islice

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte
def evaluate_rank(dir_rank):
    nfiles = os.listdir(dir_rank)
    ground_truth_val = open(ruta+"/TerrassaBuildings900/val/annotation.txt", "r")
    ground_truth_train = open(ruta+"/TerrassaBuildings900/train/annotation.txt","r")
    truth = {} #inicialitzem una taula on l'index es la id de la imatge i conté la seva categoria
    it_ground_truth_val = islice(ground_truth_val,1,None)#eliminem la primera linia de l'arxiu ja que no ens interessa
    it_ground_truth_train = islice(ground_truth_train,1,None)#eliminem la primera linia de l'arxiu ja que no ens interessa
    for line in it_ground_truth_val:
        id_foto = line.index("\t")
        final = line.index("\n")
        truth[line[0:id_foto]] = line[id_foto+1:final] #guardem la categoria de cada imatge a un vector
    for line in it_ground_truth_train:
        id_foto = line.index("\t")
        final = line.index("\n")
        truth[line[0:id_foto]] = line[id_foto+1:final] #guardem la categoria de cada imatge a un vector
    for file in nfiles:
        ranking = open(dir_rank+"/"+file,"r")
        filename = file[0:file.index(".")]
        categoria = truth[filename]
        relevants = 0
        irrelevants = 0
        for line in ranking:
            final = line.index("\n")
            if truth[line[0:final]] == categoria: #si la id de la imatge coincideix amb la categoria sumem + 1 a relevants
                relevants += 1
            else:
                irrelevants += 1
            precision = float(relevants)/float((irrelevants+relevants))
        ranking.close()
        
    #return ap, man #retornem tant la average precision com la mean average precision

evaluate_rank(ruta + "/files/ranking_val")