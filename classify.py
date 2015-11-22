# -*- coding: utf-8 -*-
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte
import random #carreguem la llibreria d'aleatori
import pickle

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

def classify(features,save_to,labels):
    infile_features = open(features,'r') #obrim el fitxer on estan les features
    #infile_labels = open(labels,'r') #obrim el fitxer on estan les possibles labels
    outfile = open(save_to+'/classification.txt', 'w'); #Creem un fitxer on guardar les classificacions
    outfile.write("ImageID" "\t" "ClassID" "\n")
    features = pickle.load(infile_features)
    
    for k in features.keys():
        outfile.write(k + "\t" + random.choice(open(labels).readlines()))
    outfile.close()
    
classify(ruta+'/files/features_train.p',ruta+"/files", ruta+"/files/labels.txt"); #crida a funció random_classification.