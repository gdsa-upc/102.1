# -*- coding: utf-8 -*-
from functions import *
import pickle
from sklearn import svm,grid_search

def train_clasificador(annotations,path_bow_train):
    file_train = open(path_bow_train,'r')
    bow_train = pickle.load(file_train)
    file_train.close()
    annotations_file = open(annotations,'r')
    next(annotations_file)
    clases = []
    for l in annotations_file:
        line = l[l.index("\t")+1:l.index("\n")]
        clases.append(line)
    dsc = []
    ids = []
    for i in bow_train.keys():
        dsc.append(bow_train[i])
        ids.append(i)        
    weight = {}
    nclases = len(set(clases))
    for k in clases:
        if k not in weight:
            ncl = clases.count(k)
            weight[k] = float(len(dsc))/(nclases*ncl)
    """clf = svm.SVC(class_weight = weight)
    a = clf.fit(dsc,clases)
    """
    svr = svm.SVC()
    params = {'kernel':('linear','rbf'),'C':[1,1000]}
    a = grid_search.GridSearchCV(svr,params)
    a.fit(dsc,clases)
    bests_params = a.best_params_
    
    clf = svm.SVC(C = bests_params['C'], kernel = bests_params['kernel'], class_weight = weight)
    clf.fit(dsc,clases)    
    return clf.predict(dsc)

if __name__ == "__main__":
    r = train_clasificador("../TerrassaBuildings900/train/annotation.txt","../files/bow_train.p")