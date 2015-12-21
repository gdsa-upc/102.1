# -*- coding: utf-8 -*-
import os

def rank_kaggle(path,val_or_test):
    ranks = os.listdir(path+"/ranking_"+val_or_test) # ranks son els noms de tots els .txt que contenen el nostre rànking
    annot = open("../TerrassaBuildings900/val/annotation.txt",'r')
    definitive = open(path+"/rank_kaggle_"+val_or_test+".txt",'w')
    definitive.write("Query,RetrievedDocuments")
    definitive.write("\n")
    for a in annot:
        for item in ranks:
            if item[0:-4] == a[0:a.index("\t")]:
                definitive.write(item[0:-4])
                definitive.write(",")
                ranking = open(path+"/ranking_"+val_or_test+"/"+item,'r')
                for line in ranking:
                    definitive.write(line[0:line.index("\n")])
                    definitive.write(" ")
                definitive.write("\n")
                
def classify_kaggle(path,val_or_test):
    clas = open(path+"/classification_"+val_or_test+".txt",'r')
    kaggle = open(path+"/classification_kaggle_"+val_or_test+".txt",'w')
    kaggle.write("Id,Prediction")
    kaggle.write("\n")
    next(clas)
    for line in clas:
        kaggle.write(line[0:line.index("\t")])
        kaggle.write(",")
        kaggle.write(line[line.index("\t")+1:line.index("\n")])
        kaggle.write("\n")
    
rank_kaggle("../files","val")
classify_kaggle("../files","val")