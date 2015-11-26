# -*- coding: utf-8 -*-
import numpy as np
import os

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte
def evaluate_rank(dir_rank):
    nfiles = os.listdir(dir_rank)
    ground_truth_val = open(ruta+"/TerrassaBuildings900/val/annotation.txt", "r")
    ground_truth_train = open(ruta+"/TerrassaBuildings900/train/annotation.txt","r")
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
    MAN = 0
    for file in nfiles:
        ranking = open(dir_rank+"/"+file,"r")#obrim l'arxiu rank d'una imatge de cerca
        filename = file[0:file.index(".")]
        categoria = truth[filename] #assignem la categoria que té la imatge de cerca
        relevants = 0
        precision = 0
        APC = 0
        AP[filename] = 0
        irrelevants = 0
        k = 0
        for line in ranking:
            final = line.index("\n")
            k += 1
            if truth[line[0:final]] == categoria: #si la id de la imatge coincideix amb la categoria sumem + 1 a relevants
                relevants += 1
                precision = precision + float(relevants)/float(k)
            else:
                irrelevants += 1
        AP[filename] = float(precision)/float(relevants)
        APC = APC + AP[filename]
        ranking.close()
    MAN = APC/len(nfiles)
    return AP, MAN

AP,MAN = evaluate_rank(ruta + "/files/ranking_val")