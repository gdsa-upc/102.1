# -*- coding: utf-8 -*-
import cv2
import os

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

def get_local_features(query):
    img = cv2.imread(ruta+query,1) #obrim la imatge que hi es a la carpeta images
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(resized,None) #cambiar resized por image cuando queramos hacerl ocon las imagenes completas
    print len(kp) #mostra el nom de punts d'interés que hi han a la imatge
    print (len(des[4])) #mostra num de descriptors del kp 4
    for k in range(1, len(kp)):
        print kp[k] 
        print des[k]

get_local_features('/imagen_primerscript/people.jpg') #tendremos que cambiar la foto
    