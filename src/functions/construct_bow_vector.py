#-*- coding: utf-8 -*-    import numpy as np
import numpy as np
from sklearn import preprocessing

def construct_bow_vector(assignments):
    #Paràmetres de la funcio: els assignments realitzadors sobre els descriptors. 
    BoW_hist = np.bincount(assignments) 
    BoW_norm= preprocessing.normalize(BoW_hist)  #Normalitzem els valors entre 0 i  gràcies a la funció 'normalize'.  
    return BoW_norm