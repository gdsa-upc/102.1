#-*- coding: utf-8 -*-    
import os
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,whiten
from get_local_features import get_local_features
from train_codebook import train_codebook


ruta = os.path.dirname(os.path.abspath(__file__)) # Definim la instrucció principal que busca la ruta absoluta del fitxer

def compute_assignments(codebook,desc):
    #Paràmetres de la funcio: el codebook amb les centroides trobats i els decriptors
    return vq(whiten(desc), codebook)

if __name__ == "__main__":
    desc = get_local_features("../imagen_primerscript/people.jpg")
    desc2 = get_local_features("../imagen_primerscript/tiger.jpg")
    codebook,_ = train_codebook(5,desc)
    assig = compute_assignments(codebook,desc2)    
    print "Longuitud del assignments= " + str(len(assig))
    plt.scatter(desc2[:,0], desc2[:,1]), plt.scatter(desc[:,0],desc[:,1]), plt.scatter(codebook[:,0], codebook[:,1], color='r'), plt.show() 