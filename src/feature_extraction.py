from functions import *
import os
import pickle

def get_descriptors(path):
    nfiles = os.listdir(path)
    dsc = []
    for file in nfiles:
        desc = get_local_features(path + "/" + file)
        for features in desc:
            dsc.append(features)
    return dsc

def assignments_train(path):
    dsc = get_descriptors(path)
    centroides,_ = train_codebook(100,dsc)
    clase = compute_assignments(centroides,dsc)
    hist = construct_bow_vector(clase)
    return centroides, hist

def assignments_val(path,centroides):
     dsc = get_descriptors(path)
     clase = compute_assignments(centroides,dsc)
     hist = construct_bow_vector(clase)
     return hist

if __name__ == "__main__":
    centroides, hist_train = assignments_train("../TerrassaBuildings900/train/images")
    BoW_train_file = open("../files/features_train.p", 'w')
    pickle.dump(hist_train, BoW_train_file)
    BoW_train_file.close()
    hist_val = assignments_val("../TerrassaBuildings900/val/images",centroides)
    BoW_val_file = open("../files/features_val.p", 'w')
    pickle.dump(hist_val, BoW_val_file)
    BoW_val_file.close()