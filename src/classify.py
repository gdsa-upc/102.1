# -*- coding: utf-8 -*-
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte
import pickle

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

def classify(features,save_to,trained_model):
    infile_features = open(features,'r') #obrim el fitxer on estan les features
    features_dic = pickle.load(infile_features) #carreguem el contingut de features a features_dic
    classifier = pickle.load(open(trained_model, 'r'))
    #infile_labels = open(labels,'r') #obrim el fitxer on estan les possibles labels
    outfile = open(save_to, 'w'); #Creem un fitxer on guardar les classificacions
    outfile.write("ImageID" "\t" "ClassID" "\n")
    
    
    for image_id, image_features in features_dic.items():
        outfile.write(str(image_id) + "\t" + str(classifier.predict(image_features)[0])+"\n")
    outfile.close()

if __name__ == "__main__":
    classify(ruta+"/../files/bow_val.p",ruta+"/../files/classified_files.txt", ruta+ "/../files/classifier.p")