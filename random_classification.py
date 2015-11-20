# -*- coding: utf-8 -*-
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte
import random #carreguem la llibreria d'aleatori
from itertools import islice

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

def random_line(afile):
    line = next(afile)
    for num, aline in enumerate(afile):
      if random.randrange(num + 2): continue
      line = aline
    return line
    
def random_classification(features,save_to,labels):
    infile_features = open(features,'r') #obrim el fitxer on estan les features
    infile_labels = open(labels,'r') #obrim el fitxer on estan les possibles labels
    outfile = open(save_to+'/classification.txt.', 'w'); #Creem un fitxer on guardar les classificacions
    outfile.write("ImageID" "\t" "ClassID" "\n")
    it = islice(infile_features,1,None) #Salta la primera linia del fitxer .txt
    for line in it:
        tabulacio = line.index("\t") #busca la posicio on es troba el tabulador a la linia
        outfile.write("\n"+line[0:tabulacio]+"\t") # Posem un salt de linea i l'image ID actual
        outfile.write(random_line(infile_labels))
    outfile.close()
        
random_classification(ruta+'/TerrassaBuildings900/train/annotation.txt',ruta+"/Resultados", ruta+"/TerrassaBuildings900/labels.txt"); #creacio train database