from functions import *
import os
import pickle

def get_descriptors(path):
    nfiles = os.listdir(path)
    dsc_all = []
    dsc_ind = {}
    for file in nfiles:
        filename = file[0:file.index(".")]
        dsc_ind[filename] = get_local_features(path + "/" + file)
        for feat in dsc_ind[filename]:
            dsc_all.append(feat)
    return dsc_all,dsc_ind

def train(dsc,nclusters):
    centroides,_ = train_codebook(nclusters,dsc)
    return centroides

def save_bow(centroides,dsc,val_or_train,nclusters):
    bow = dict()
    for l in dsc:
        assig = compute_assignments(centroides,dsc[l])
        bow[l] = construct_bow_vector(assig,nclusters)
    bow_file = open("../files/bow_" + val_or_train + ".p",'w')
    pickle.dump(bow,bow_file)
    bow_file.close()

if __name__ == "__main__":
    """dsc_all_train, dsc_ind_train = get_descriptors("../TerrassaBuildings900/train/images")
    nclusters = 50
    centroides = train(dsc_all_train,nclusters)
    save_bow(centroides,dsc_ind_train,"train",nclusters)
    _, dsc_ind_val = get_descriptors("../TerrassaBuildings900/val/images")
    save_bow(centroides,dsc_ind_val,"val",nclusters)
    """
    f = open("../files/bow_val.p",'r')
    p = pickle.load(f)
    t = open("../files/bow_train.p",'r')
    tt = pickle.load(t)
    ln = []
    for i in tt:
        if i not in ln:
            ln.append(i)