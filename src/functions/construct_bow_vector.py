#-*- coding: utf-8 -*-    import numpy as np
import numpy as np
from sklearn import preprocessing
import os
from get_local_features import get_local_features
from train_codebook import train_codebook
from compute_assignments import compute_assignments

def construct_bow_vector(assignments):
    #Paràmetres de la funcio: els assignments realitzadors sobre els descriptors. 
    BoW_hist = np.float64(np.reshape(np.bincount(assignments), (1,-1))) #corretgim el warning de reshape
    BoW_norm= preprocessing.normalize(BoW_hist)  #Normalitzem els valors entre 0 i  gràcies a la funció 'normalize'.
    return BoW_norm
if __name__ == "__main__":
    nfiles = os.listdir("../imagen_primerscript")
    dsc = []
    BoW = {}
    dsc_ind = {}
    for file in nfiles:
        filename = file[0:file.index(".")]
        dsc_ind[filename] = get_local_features("../imagen_primerscript/" + file)
        for feat in dsc_ind[filename]:
            dsc.append(feat)
    codebook,_ = train_codebook(5,dsc)
    for file in nfiles:
        filename = file[0:file.index(".")]
        clase = compute_assignments(codebook, dsc_ind[filename])
        BoW[filename] = construct_bow_vector(clase)