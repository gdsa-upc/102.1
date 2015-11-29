#-*- coding: utf-8 -*-    import numpy as np
import numpy as np
import pickle
import os
from sklearn.preprocessing import normalize
from train_kmeans import train_codebook
from get_local_features import get_local_features
from compute_assignments import compute_assignments

def construct_bow_vector(assignments,train_codebook,id_image,val_or_train):
    #Paràmetres de la funcio: els assignments realitzadors sobre els descriptors, el codebook amb les centroides trobats, 
    # els decriptors, les ids de les imatges i si l'indicador de si la carpeta 
    #on estem es la de validacio o la d'entrenament
    normalized = [] #declaració del vector amb els valors de les assignacions normalitzades entre 0 i 1. 
    for element in assignments:
        div = normalize(assignments,train_codebook,id_image,val_or_train)
        n = assignments[element]/div
        normalized.append(n)
    return normalized
    
ruta = os.path.dirname(os.path.abspath(__file__)) # Definim la instrucció principal que busca la ruta absoluta del fitxer
IDs_file_T = open(ruta+"/files/outfile_train.txt", 'r') #obrim l'arxiu que conté les ids de les imatges d'entrenament
local_features_file_T = open(ruta + "/files/local_features_train.p",'w') #obrim l'arxiu en el que escriurem les caracteristiques
IDs_file_V = open(ruta+"/files/outfile_val.txt", 'r') #obrim l'arxiu que conté les ids de les imatges de validació
local_features_file_V = open(ruta + "/files/local_features_val.p",'w') #obrim l'arxiu en el que escriurem les caracteristiques
feat_vec = dict() #inicialitzem el diccionari buit
descriptors = [] #Declarem el vector de descriptors
assignments = [] #Declarem el vector d'assignacions

for line in IDs_file_T:
    BoW = np.zeros(1,100)#Generem el vector de paraules normalitzades.
    final = file.index("\n")
    dscrp = get_local_features("/TerrassaBuildings900/train/images"+file)
    descriptors.append(dscrp) #introduim el vector de descriptors
    centroide,_ = train_codebook(13,descriptors)
    assig = compute_assignments(centroide,_,dscrp,file,"train") #crida a la funció compute_assignments
    assignments.append(assig) #Insertem cada assignacio al vector de assignacions
    norm = construct_bow_vector(assig,centroide,_,dscrp,file,"train") #crida a la funció construct_bow_vector
    BoW.insert(norm) # insertem cada element normalitzat al vector BoW.
    feat_vec[line[0:final]] = BoW #Afegim al diccionari el vector de paraules normalitzades.
    pickle.dump(feat_vec, local_features_file_T) #Escribim el diccionari amb 'pickle'.
IDs_file_T.close() #Tanquem el directori on es trobaven les imatges d'entrenament.

for line in IDs_file_V:
    BoW = np.zeros(1,100)#Generem el vector de paraules normalitzades.
    final = file.index("\n")
    dscrp = get_local_features("/TerrassaBuildings900/val/images"+file)
    descriptors.append(dscrp)
    centroide,_ = train_codebook(13,descriptors)
    assig = compute_assignments(centroide,_,dscrp,file,"val") #crida a la funció compute_assignments
    assignments.append(assig) #Insertem cada assignacio al vector de assignacions
    norm = construct_bow_vector(assig,centroide,_,dscrp,file,"val")
    BoW.insert(norm) # insertem cada element normalitzat al vector BoW.
    feat_vec[line[0:final]] = BoW #Afegim al diccionari el vector de paraules normalitzades
    pickle.dump(feat_vec, IDs_file_V) #Escribim el diccionari amb 'pickle'
IDs_file_V.close() #Tanquem el directori on es trobaven les imatges de validació.
