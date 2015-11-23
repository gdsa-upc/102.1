# -*- coding: utf-8 -*-
import os
from itertools import islice
#Afegim les llibreries de scikit, sklearn.metrics per poder fer tots els càlculs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def evaluate_classification(automatic_classification, ground_truth, val_or_test):
    automatic_annotation = open(automatic_classification+'/classification_'+val_or_test+'.txt','r') #obrim el fitxer generat per la funció classify
    groundtruth_annotation = open(ground_truth, 'r') #obrim el fitxer d'annotacio donat
    it_automatic = islice(automatic_annotation,1,None) #no tenim en compte la primera linia del fitxer generat pel classify
    it_groundtruth = islice(groundtruth_annotation,1,None) # no tenim en compte la primera linia del fitxer donat.
    automatic = []
    ground_truth = []
    for line in it_automatic:
        inicio = line.index("\t")
        final = len(line)
        automatic.append(line[inicio:final]) #Afegim les categories a cada entrada de l'array
    for line in it_groundtruth:
        inicio = line.index("\t")
        final = len(line)
        ground_truth.append(line[inicio:final]) #Afegim les categories a cada entrada de l'array

    # CALCULEM LA MATRIU DE CONFUSIO:
    print("MATRIU DE CONFUSIO:")
    print(confusion_matrix(ground_truth,automatic))
    print("\n")
    
    # CALCULEM L'ACCURACY
    print("ACCURACY:")
    print(accuracy_score(ground_truth,automatic))
    print("\n")
    
    # CALCULEM LA PRECISSION
    print("PRECISSION:")
    print(precision_score(ground_truth,automatic,average='macro'))
    print("\n")
    
    # CALCULEM EL RECALL
    print("RECALL:")
    print(recall_score(ground_truth,automatic,average='macro'))
    print("\n")
    
    # CALCULEM EL F1
    print("F1:")
    print(f1_score(ground_truth,automatic,average='macro'))
    print("\n")


ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte
evaluate_classification(ruta+'/files',ruta+'/TerrassaBuildings900/val/annotation.txt',"val")