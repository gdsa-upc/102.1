#-*- coding: utf-8 -*-    import numpy as np
import numpy as np
from sklearn import preprocessing
import os
from get_local_features import get_local_features
from train_codebook import train_codebook
from compute_assignments import compute_assignments

def construct_bow_vector(assignments):
    #Paràmetres de la funcio: els assignments realitzadors sobre els descriptors. 
    BoW_hist = np.bincount(assignments) 
    BoW_norm= preprocessing.normalize(BoW_hist)  #Normalitzem els valors entre 0 i  gràcies a la funció 'normalize'.  
    return BoW_norm

if __name__ == "__main__":
    nfiles = os.listdir("../imagen_primerscript")
    dsc = []
    for file in nfiles:
        desc = get_local_features("../imagen_primerscript/" + file)
        for feat in desc:
            dsc.append(feat)
    codebook,_ = train_codebook(5,dsc)
    clase = compute_assignments(codebook,dsc)
    BoW = construct_bow_vector(clase)