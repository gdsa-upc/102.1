# -*- coding: utf-8 -*-
from functions import *
import pickle
from sklearn import svm

def train_clasificador(annotations,path_bow_train):
    file_train = open(path_bow_train,'r')
    bow_train = pickle.load(file_train)
    file_train.close()
    annotations_file = open(annotations,'r')
    next(annotations_file)
    clases = []
    for l in annotations_file:
        line = l[l.index("\t"):l.index("\n")]
        clases.append(line)
    dsc = []
    ids = []
    for i in bow_train.keys():
        dsc.append(bow_train[i])
        ids.append(i)        
    clf = svm.SVC()
    a = clf.fit(dsc,clases)
    return a.predict(dsc)

if __name__ == "__main__":
    r = train_clasificador("../TerrassaBuildings900/train/annotation.txt","../files/bow_train.p")