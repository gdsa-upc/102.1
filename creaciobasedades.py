# -*- coding: utf-8 -*-
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte
from itertools import islice

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

def build_database(file,val_or_train):
    infile = open(file,'r') #obrim el fitxer on estan les anotacions
    outfile = open(ruta+'/outfile_'+val_or_train+'.txt','w') # Creem un fitxer per guardar les id's de train
    it = islice(infile,1,None) #Salta la primera linia del fitxer .txt
    for line in it:
        final = line.index("\t") #busca la posicio on es troba el tabulador a la linia
        outfile.write(line[0:final]+"\n") # Assignem cada id a una linia del nou fitxer
    outfile.close()
        
build_database(ruta+'/TerrassaBuildings900/train/annotation.txt',"train"); #creacio train database
build_database(ruta+'/TerrassaBuildings900/val/annotation.txt',"val"); #creacio val database