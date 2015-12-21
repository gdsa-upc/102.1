# -*- coding: utf-8 -*-
from functions import *
import os
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances
import time
from sklearn import svm,grid_search
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def train(dsc,nclusters): #funció per entrenar el codebook
    centroides,_ = train_codebook(nclusters,dsc) 
    return centroides #retornem els centroides

def save_bow(centroides,dsc,val_or_train,nclusters):
    bow = dict() #inicialitzem el bow
    for l in dsc:
        assig = compute_assignments(centroides,dsc[l]) #obtenim els asignaments amb els descriptors d'entrada que li passem
        bow[l] = construct_bow_vector(assig,nclusters) #construim el bow corresponent a cada imatge
    bow_file = open("../files/bow_" + val_or_train + ".p",'w')
    pickle.dump(bow,bow_file) #escribim el bow a el diccionari
    bow_file.close()

def train_classify(annotations,path_bow_train):
    file_train = open(path_bow_train,'r') #obrim l'arxiu amb els bow d'entrenament
    bow_train = pickle.load(file_train) #carreguem el diccionari
    file_train.close()
    annotations_file = open(annotations,'r') #obrim l'arxiu d'anotacions d'entrenament
    next(annotations_file) #saltem la primera linea de l'arxiu d'anotacions per no llegir-la
    clases = [] #inicialitzem el vector clases
    for k in bow_train.keys():
        clases.append(k)
    ids = []
    for l in annotations_file:
        rec = l.split("\t")
        a = rec[1].split("\n")
        rec[1] = a[0]
        for i in range(0, len(clases)):
            if clases[i] == rec[0]:
                clases[i] = rec[1]
    dsc = [] #inicialitzem el vector d'arrays que contindrá els descriptors
    for i in bow_train.keys():
        dsc.append(bow_train[i]) #afegim al vector un array amb els bow de cada imatge d'entrenament
    weight = {} #inicialitzem el diccionari que contindrà una clase com a index i un pes com a valor
    nclases = len(set(clases)) #contem el número de clases que n'hi han
    for k in clases:
        if k not in weight:
            ncl = clases.count(k) #compta el nombre d'elements de cada clase
            if k == "desconegut":
                weight[k] = float(len(dsc))/(nclases*ncl) #calculem el pes de cada clase
            else:
                weight[k] = float(len(dsc))/(nclases*ncl)
    svr = svm.SVC(dsc,class_weight = weight)
    c = [10,1,100,1000,10000,100000] #establim la llista per que vagi de 1 a 20
    params = {'kernel':('linear','rbf'),'C':c}
    a = grid_search.GridSearchCV(svr,params) #busquem els millors parametres possibles pasar dsc y clases
    a.fit(dsc,clases) 
    best_params = a.best_params_ #guardem els millors parametres a la variable best_params
    clf = svm.SVC(C = best_params['C'], kernel = 'linear', class_weight = weight) #cridem al svc amb els millors parametresç
    clf.fit(dsc,clases)  #apliquem el fit per obtenir el model entrenat
    pickle.dump(clf, open("../files/classifier.p", "wb" ))  #guardem el model com a un diccionari
    return clf.predict(dsc)
    
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


def evaluate_classification(automatic_classification, ground_truth, val_or_test):
    automatic_annotation = open(automatic_classification+'/classification_'+val_or_test+'.txt','r') #obrim el fitxer generat per la funció classify
    groundtruth_annotation = open(ground_truth, 'r') #obrim el fitxer d'annotacio donat
    automatic = []
    id_automatic = []
    next(automatic_annotation) #Saltem la primera linia del fitxer
    for line in automatic_annotation:
        inicio = line.index("\t")
        final = line.index("\n")
        id_automatic.append(line[0:inicio])
        automatic.append(line[inicio+1:final]) #Afegim les categories a cada entrada de l'array
    next(groundtruth_annotation) #Saltem la primera linia del fitxer
    ground_truth = range(0, len(id_automatic))
    for l in groundtruth_annotation:
        id_annot = l[0:l.index("\t")]
        clase_annot = l[l.index("\t")+1:l.index("\n")]
        for i in range(0, len(id_automatic)):
            if id_automatic[i] == id_annot:
               ground_truth[i] = clase_annot

    # CALCULEM LA MATRIU DE CONFUSIO:
    cm = confusion_matrix(ground_truth,automatic)
    labe = np.unique(ground_truth) #ens treu el llistat de les categories en el ground_truth (una categoria només una vegada).
    # CALCULEM L'ACCURACY
    accuracy = accuracy_score(ground_truth,automatic)
    # CALCULEM LA PRECISSION
    precision = precision_score(ground_truth,automatic,average='macro')
    # CALCULEM EL RECALL
    recall = recall_score(ground_truth,automatic,average='macro')
    # CALCULEM EL F1
    f1 = f1_score(ground_truth,automatic,average='macro')
    return cm,labe,accuracy,precision,recall,f1
    

if __name__ == "__main__":
    dsc_all_train = pickle.load(open("../files/dsc_all_train.p",'rb'))
    dsc_ind_train = pickle.load(open("../files/dsc_ind_train.p",'rb'))
    dsc_ind_val = pickle.load(open("../files/dsc_ind_val.p",'rb'))
    nclusters = [800,1000,1200,1400,1600]
    graficas = open("../files/graficas_classification.txt",'w')
    for i in nclusters:
        t = time.time()
        centroides = train(dsc_all_train,i) #calculem els centroides del codebook
        save_bow(centroides,dsc_ind_train,"train",i) #calculem i dessem els bow de les imatges d'entrenament
        save_bow(centroides,dsc_ind_val,"val",i) #guardem els bow de les imatges de validació
        r = train_classify("../TerrassaBuildings900/train/annotation.txt","../files/bow_train.p")
        classify("../files/bow_val.p","../files/classification_val.txt", "../files/classifier.p")
        cm,labe,accuracy,precision,recall,f1 = evaluate_classification('../files','../TerrassaBuildings900/val/annotation.txt',"val")
        graficas.write(str(i) + "\t" + str(f1) + "\t" + str(time.time()-t) + "\n")
        print i
    graficas.close()