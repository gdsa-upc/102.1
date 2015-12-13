# -*- coding: utf-8 -*-
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte
import pickle

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

def classify(features,save_to,trained_model):
    infile_features = open(features,'r') #obrim el fitxer on estan les features
    features_dic = pickle.load(infile_features) #carreguem el contingut de features a features_dic
    #infile_labels = open(labels,'r') #obrim el fitxer on estan les possibles labels
    outfile = open(save_to, 'w'); #Creem un fitxer on guardar les classificacions
    #outfile.write("ImageID" "\t" "ClassID" "\n")
    
    
    for image_id, image_features in features_dic.items():
        outfile.write(str(image_id) + "\t" + str(features_dic.predict(image_features)[0])+"\n")
    outfile.close()

        
#classify(ruta+'/files/features_val.p',ruta+"/files", ruta+"/files/labels.txt", "val"); #crida a funció random_classification.