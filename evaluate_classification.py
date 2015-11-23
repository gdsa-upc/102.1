import os
from itertools import islice

def evaluate_classification(automatic_classification, ground_truth, val_or_test):
    automatic_annotation = open(automatic_classification+'/classification_'+val_or_test+'.txt','r')
    groundtruth_annotation = open(ground_truth, 'r')
    #it_automatic = islice(automatic_annotation,1,None)
    #it_groundtruth = islice(groundtruth_annotation,1,None)
    k = len(automatic_annotation.readlines())-1
    automatic_annotation.close()
    groundtruth_annotation.close()

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte
evaluate_classification(ruta+'/files',ruta+'/TerrassaBuildings900/val/annotation_valid.txt',"val")