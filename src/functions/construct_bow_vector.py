#-*- coding: utf-8 -*-    import numpy as np
import numpy as np
from sklearn import preprocessing
import os
from get_local_features import get_local_features
from train_codebook import train_codebook
from compute_assignments import compute_assignments

def construct_bow_vector(assignments,k):
    #Paràmetres de la funcio: els assignments realitzadors sobre els descriptors.
    print assignments
    BoW_hist = np.zeros(k) # Creerm una llista buida de k(número de clusters) valors igualats a cero.
    for a in assignments:
        BoW_hist[a] += 1 # Per cada entrada a l'assigments sumem 1 al índex que l'hi pertoca en el histograma.
    BoW_hist = np.float64(np.reshape(BoW_hist, (1,-1))) #corretgim el warning de reshape
    BoW_norm= preprocessing.normalize(BoW_hist)  #Normalitzem els valors entre 0 i 1 gràcies a la funció 'normalize'.
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
<<<<<<< HEAD
    for file in nfiles:
        filename = file[0:file.index(".")]
        clase = compute_assignments(codebook, dsc_ind[filename])
        BoW[filename] = construct_bow_vector(clase)
=======
    clase,k = compute_assignments(codebook,dsc)
    BoW = construct_bow_vector(clase,k)
>>>>>>> 55049dc33d2baebe6ad99db3ddbf1dd6a044bc24
