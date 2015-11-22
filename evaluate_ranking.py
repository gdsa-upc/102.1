# -*- coding: utf-8 -*-
# Creació de la funció "evaluate_ranking.py". Feta per: Eric Díaz Cívico.
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import os # Carreguem la llibreria corresponent a la funció de introducció de
# rutes absolutes d'un fitxer
from itertools import islice
from sklearn.metrics import average_precision_score


ruta_abs = os.path.dirname(os.abspath(__file__)) #Obtenim la ruta absoluta de la carpeta on es troben els fitxers
Direct_llista = rank(ruta_abs+'/files',ruta_abs+'/files',ruta+'/files/features_train.p',"valid") #Invoquem a la funció rank.py per obtenir el directori de la llista classificada d'imatge

def Evaluate_Ranking(Direct_llista,train_or_valid): #Funció declarada passant com a paràmetres 

    fitxer_anot = open(ruta_abs+"/annotation_"+train_or_valid+".txt" , "r") #Obrim els arxius d'annotació en funció del valor de 'train_or_valid')
    Final_file = open(ruta_abs+"/average_precision_"+train_or_valid+".txt", "w") #Obrim l'arxiu on escriurem el AP per cada consulta  
    Final_file2 = open(ruta_abs+"/Mean_average_precision_"+train_or_valid+".txt", "w") ##Obrim l'arxiu on escriurem el MAP per cada consulta  
    
    for line in Direct_llista:
        Final_file = np.random.rand(1,180) #Obrim el vector aleatori on s'inclouran el total de APs per cada consulta
        endline = line.index("\n") #Indicació del final de línea de casa vector de AP's
        if train_or_valid == "train":
            fitxer_anot = "annotation_valid.txt"
            APt = average_precision_score(Direct_llista,fitxer_anot)
            Final_file.append(APt)
            # A continuació escriurem en el fitxer cada línia de les APS per les imatges d'entrenament
            Final_file_train.write(line[0:final] + "For Query "+line+":\t" + str(Final_File).replace("\n","").replace("[[","").replace("]]","") + "\n")
        else:
            fitxer_anot = "annotation_valid.txt"
            APv = average_precision_score(Direct_llista,fitxer_anot)
            Final_file.append(APv)
            # A continuació escriurem en el fitxer cada línia de les APS per les imatges de validació
            Final_file_valid.write(line[0:final] + "For Query "+line+":\t"+ str(Final_File).replace("\n","").replace("[[","").replace("]]","") + "\n") 
    Final_file_train.close() #Tanquem el fitxer corresponent a les imatges d'entrenament
    Final_file_valid.close() #Tanquem el fitxer corresponent a les imatges de validació

    for line in Final_file_train:
        for element in line:
            suma_train = (sum(line))
            sum_elems = (sum(element))
        print("Aps d'entrenament sumats!")
        Final_file2 = [] #Creem el array necessari per col·locar el valor del MAP
        MAP_train = suma_train/suma_elems #Fem la peració per obtenir aquest valor
        Final_file2.append(MAP_train) #Introduïm el valor resultant dintre del array creat
        #A Continuació esciurem el valor resultant del MAP dintre del fitxer de sortida
        Final_file2_train.write(line[0:final] + "For Query "+line+":\t" + "Mean_AVerage_Precision = "+str(Final_file2).replace("\n","").replace("[[","").replace("]]","") + "\n")
    Final_file_train.close() #Tanquem el fitxer per on hem llegit les dades dels APS de cada consulta
    Final_file2_train_close() #Tanquem el ftixer per on hem esccrit els valors del MAP resultants per cada línia
    
    
    for line in Final_file_valid:
        for element in line:
            suma_valid = (sum(line))
            sum_elems = (sum(element))
        print("Aps d'entrenament sumats!")
        Final_file2 = [] #Creem el array necessari per col·locar el valor del MAP
        MAP_valid = suma_valid/suma_elems #Fem la peració per obtenir aquest valor
        Final_file2.append(MAP_valid) #Introduïm el valor resultant dintre del array creat
        #A Continuació esciurem el valor resultant del MAP dintre del fitxer de sortida
        Final_file2_valid.write(line[0:final] + "For Query "+line+":\t" + "Man_AVerage_Precision = "+str(Final_file2).replace("\n","").replace("[[","").replace("]]","") + "\n")
    Final_file_valid.close() #Tanquem el fitxer per on hem llegit les dades dels APS de cada consulta
    Final_file2_valid_close() #Tanquem el ftixer per on hem esccrit els valors del lMAP resultants per cada línia

Evaluate_Ranking("features_train.txt","train")
Evaluate_Ranking("features_valid.txt","valid")


