#-*- coding: utf-8 -*-    
import os
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,whiten
from get_local_features import get_local_features
from train_codebook import train_codebook


ruta = os.path.dirname(os.path.abspath(__file__)) # Definim la instrucció principal que busca la ruta absoluta del fitxer

def compute_assignments(codebook,desc):
    #Paràmetres de la funcio: el codebook amb les centroides trobats i els decriptors
    return whiten(vq(whiten(desc), codebook)) #tornem les distancies normalitzades

if __name__ == "__main__":
    dsc = []
    desc = get_local_features("../imagen_primerscript/people.jpg")
    desc2 = get_local_features("../imagen_primerscript/tiger.jpg")
    for feat in desc:
        dsc.append(feat)
    for feat in desc2:
        dsc.append(feat)
    codebook,_ = train_codebook(5,dsc)
    assig = compute_assignments(codebook,dsc)
    #plt.scatter(codebook[:,0],codebook[:,1]),plt.scatter(dsc[:,0],dsc[:,1], color='r'), plt.show() 