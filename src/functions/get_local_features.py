# -*- coding: utf-8 -*-
from rootsift import RootSIFT
import cv2

def get_local_features(query):
    image = cv2.imread(query,0)
    r = 128.0 / image.shape[1]
    dim = (128, int(image.shape[0] * r)) # Tamaño de 128 orientativo, probar mas adelante con diferentes tamaños
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #redimensiono la imatge per interpolacio fent-la de tamany 500 mes o menys    
    
    # detect Difference of Gaussian keypoints in the image
    detector = cv2.FeatureDetector_create("SIFT")
    kps = detector.detect(image)
    
    # extract RootSIFT descriptors
    rs = RootSIFT()
    (kps, descs) = rs.compute(image, kps)
    return descs[0:5] #return the first 5 elements
    
if __name__ == "__main__":
    a = get_local_features("../imagen_primerscript/people.jpg")
    print a