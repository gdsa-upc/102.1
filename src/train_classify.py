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
    svr = svm.SVC()
    params = {'kernel':('linear','rbf'),'C':[1,3,10,100,1000]}
    a = grid_search.GridSearchCV(svr,params)
    a.fit(dsc,clases)
    bests_params = a.best_params_
    print bests_params
    clf = svm.SVC(C =1, kernel = 'linear', class_weight = weight)
    clf.fit(dsc,clases)    
    return clf.predict(dsc)

if __name__ == "__main__":
    r = train_clasificador("../TerrassaBuildings900/train/annotation.txt","../files/bow_train.p")
    print str(list(r).count("desconegut"))
    print str(list(r).count("societat_general"))
    print str(list(r).count("farmacia_albinyana"))
    print str(list(r).count("castell_cartoixa"))
    print str(list(r).count("escola_enginyeria"))
    print str(list(r).count("mercat_independencia"))
    print str(list(r).count("teatre_principal"))
    print str(list(r).count("masia_freixa"))
    print str(list(r).count("mnactec"))
    print str(list(r).count("ajuntament"))
    print str(list(r).count("dona_treballadora"))
    print str(list(r).count("estacio_nord"))
    print str(list(r).count("catedral"))    