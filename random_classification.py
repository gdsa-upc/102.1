# -*- coding: utf-8 -*-
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte
import random #carreguem la llibreria d'aleatori
from itertools import islice

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte
def random_classification(features,save_to,labels):
    infile_features = open(features,'r') #obrim el fitxer on estan les features
    infile_labels = open(labels,'r') #obrim el fitxer on estan les possibles labels
    outfile = open(save_to+'/classification.txt.', 'w'); #Creem un fitxer on guardar les classificacions
    outfile.write("ImageID" "\t" "ClassID" "\n")
    it = islice(infile_features,1,None) #Salta la primera linia del fitxer .txt
    for line in it:
        tabulacio = line.index("\t") #busca la posicio on es troba el tabulador a la linia
        random_lines = random.choice(open(labels).readlines())# Busquem una linea aleatoria en labels.txt
        outfile.write(line[0:tabulacio]+"\t"+random_lines) # Posem un salt de linea i l'image ID actual
    outfile.close()
        
random_classification(ruta+'/TerrassaBuildings900/train/annotation.txt',ruta+"/files", ruta+"/files/labels.txt"); #crida a funció random_classification.