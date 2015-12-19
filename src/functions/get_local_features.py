# -*- coding: utf-8 -*-
from rootsift import RootSIFT
import cv2
import matplotlib.pyplot as plt

def get_local_features_rootSIFT(query):
    image = cv2.imread(query,0)
    r = 300.0 / image.shape[1]
    dim = (300, int(image.shape[0] * r)) # Tamaño de 128 orientativo, probar mas adelante con diferentes tamaños
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #redimensiono la imatge per interpolacio fent-la de tamany 500 mes o menys    
    # detect Difference of Gaussian keypoints in the image
    detector = cv2.FeatureDetector_create("SIFT")
    kps = detector.detect(image)
    # extract RootSIFT descriptors
    rs = RootSIFT()
    (kps, descs) = rs.compute(image, kps)
    #kp,descs = cv2.SIFT().detectAndCompute(image,None)
    return descs #return the descriptors
def get_local_features_SURF(query):
    image = cv2.imread(query,0)
    r = 300.0 / image.shape[1]
    dim = (300, int(image.shape[0] * r)) # Tamaño de 128 orientativo, probar mas adelante con diferentes tamaños
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #redimensiono la imatge per interpolacio fent-la de tamany 500 mes o menys
    surf = cv2.SURF() #posem el llindar hessian a 4000
    kp, des = surf.detectAndCompute(image, None)   
    return des
if __name__ == "__main__":
    a = get_local_features_rootSIFT("../TerrassaBuildings900/train/images/38-29833-926.jpg")
    print len(a)
    b = get_local_features_SURF("../TerrassaBuildings900/train/images/38-29833-926.jpg")
    print len(b)