# -*- coding: utf-8 -*-
import os # llibreria os per poder accedir al directori de treball
def build_database(dir_entrada,val_or_train,dir_sortida):
    IDlist = [] #Creem un array
    images = os.listdir(dir_entrada) #Llegim el nom dels fitxers que hi ha en el directori que volem (el d'entrada)
    outfile = open(dir_sortida+'/outfile_'+val_or_train+'.txt','w') # Creem un fitxer per guardar les id's  
    for file in images:
        IDlist.append(file) #Guardem a l'array les dades de les imatges amb l'extensió .jpg
    for array in IDlist:
        outfile.write(array[0:-4]+"\n") #Per cada array ens escriu l'id de cada imatge en una linea en el nou fitxer (sense .jpg) 
    outfile.close() #Tanquem el fitxer
    
ruta = os.path.dirname(os.path.abspath(__file__)) #ruta absoluta del projecte   
build_database(ruta+'/TerrassaBuildings900/train/images',"train",ruta); #creacio train database
build_database(ruta+'/TerrassaBuildings900/val/images',"val",ruta); #creacio val database