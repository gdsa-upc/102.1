    #-*- coding: utf-8 -*-    
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,whiten
from train_codebook import train_codebook
from get_local_features import get_local_features

ruta = os.path.dirname(os.path.abspath(__file__)) # Definim la instrucció principal que busca la ruta absoluta del fitxer

def compute_assignments(codebook,desc):
    #Paràmetres de la funcio: el codebook amb les centroides trobats i els decriptors
    norm_desc = whiten(desc) # Normaliza descriptores
    assignments,_ = vq(norm_desc, codebook)
    return assignments # la funció vq elabora el vector d'assignacions i retorna el vector d'assignacions