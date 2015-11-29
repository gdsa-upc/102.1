# -*- coding: utf-8 -*-
import cv2
import os
from scipy.cluster.vq import *

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte

def get_local_features(query):
    ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte
    print(ruta+"../"+query)
    img = cv2.imread(ruta+"/.."+query,1) #obrim la imatge que hi es a la carpeta images…
    r = 128.0 / img.shape[1]
    dim = (128, int(img.shape[0] * r)) # Tamaño de 128 orientativo, probar mas adelante con diferentes tamaños
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) #redimensiono la imatge per interpolacio fent-la de tamany 500 mes o menys
    sift = cv2.SIFT() # inicialitzo SIFT 
    kp,des = sift.detectAndCompute(resized,None) # obtinc els key points amb els seus descriptors
    #cambiar resized por image cuando queramos hacerl ocon las imagenes completas
    """ print len(kp) #mostra el nom de punts d'interés que hi han a la imatge
    print (len(des[4])) #mostra num de descriptors del kp 4
    for k in range(1, len(kp)): # mostro tots els key points amb els seus descriptors corresponents
        print kp[k] 
        print des[k]"""
    return whiten(des[0:5], check_finite = True)#Si no se deben retornar normalizados cambiamos a return des, devolvemos los 5 primeros descriptores para ahorrar memoria y tiempo

#des=get_local_features('/imagen_primerscript/tiger.jpg') #tendremos que cambiar la foto
