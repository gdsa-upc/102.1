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
    g = [1.0,0.1,0.001,0.0001]
    params = {'kernel':('linear','rbf','poly','sigmoid'),'C':c, 'gamma':g}
    a = grid_search.GridSearchCV(svr,params) #busquem els millors parametres possibles pasar dsc y clases
    a.fit(dsc,clases) 
    best_params = a.best_params_ #guardem els millors parametres a la variable best_params
    clf = svm.SVC(C = best_params['C'], kernel = best_params['kernel'],gamma = best_params['gamma'], class_weight = weight) #cridem al svc amb els millors parametresç
    clf.fit(dsc,clases)  #apliquem el fit per obtenir el model entrenat
    pickle.dump(clf, open("../files/classifier.p", "wb" ))  #guardem el model com a un diccionari
    return clf.predict(dsc)

if __name__ == "__main__":
    r = train_classify("../TerrassaBuildings900/train/annotation.txt","../files/bow_train.p")
    print "\n"
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