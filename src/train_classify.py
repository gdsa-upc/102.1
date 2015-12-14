# -*- coding: utf-8 -*-
import pickle
from sklearn import svm,grid_search

def train_classify(annotations,path_bow_train):
    file_train = open(path_bow_train,'r') #obrim l'arxiu amb els bow d'entrenament
    bow_train = pickle.load(file_train) #carreguem el diccionari
    file_train.close()
    annotations_file = open(annotations,'r') #obrim l'arxiu d'anotacions d'entrenament
    next(annotations_file) #saltem la primera linea de l'arxiu d'anotacions per no llegir-la
    clases = [] #inicialitzem el vector clases
    for l in annotations_file:
        line = l[l.index("\t")+1:l.index("\n")]
        clases.append(line) #fiquem totes les clases que trobem a l'arxiu d'anotacions al vector clases
    dsc = [] #inicialitzem el vector d'arrays que contindrá els descriptors
    for i in bow_train.keys():
        dsc.append(bow_train[i]) #afegim al vector un array amb els bow de cada imatge d'entrenament
    weight = {} #inicialitzem el diccionari que contindrà una clase com a index i un pes com a valor
    nclases = len(set(clases)) #contem el número de clases que n'hi han
    for k in clases:
        if k not in weight:
            ncl = clases.count(k) #compta el nombre d'elements de cada clase
            weight[k] = float(len(dsc))/(nclases*ncl) #calculem el pes de cada clase
    svr = svm.SVC()
    c = range(1,20) #establim la llista per que vagi de 1 a 20
    params = {'kernel':('linear','rbf'),'C':c}
    a = grid_search.GridSearchCV(svr,params) #busquem els millors parametres possibles
    a.fit(dsc,clases) 
    best_params = a.best_params_ #guardem els millors parametres a la variable best_params
    clf = svm.SVC(C = best_params['C'], kernel = best_params['kernel'], class_weight = weight) #cridem al svc amb els millors parametres
    clf.fit(dsc,clases)  #apliquem el fit per obtenir el model entrenat
    pickle.dump(clf, open("../files/classifier.p", "wb" ))  #guardem el model com a un diccionari
    return clf.predict(dsc)

if __name__ == "__main__":
    r = train_classify("../TerrassaBuildings900/train/annotation.txt","../files/bow_train.p")
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