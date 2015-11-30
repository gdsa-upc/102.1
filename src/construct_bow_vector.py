#-*- coding: utf-8 -*-    import numpy as np
import numpy as np
import pickle
import os
from sklearn import preprocessing
from train_codebook import train_codebook
from get_local_features import get_local_features
from compute_assignments import compute_assignments

def construct_bow_vector(assignments):
    #Paràmetres de la funcio: els assignments realitzadors sobre els descriptors. 
    BoW_hist = np.bincount(assignments) #Contem el número de vegades que apareix cada número de cluster i els acumulem en 
                                        #ordre ascendent en cada casella del vector BoW_hist
    BoW_norm= preprocessing.normalize(BoW_hist)  #Normalitzem els valors entre 0 i  gràcies a la funció 'normalize'.  
    return BoW_norm