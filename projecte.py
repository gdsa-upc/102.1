import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

ruta = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(ruta+'/images/people.jpg',1)
surf = cv2.SURF(4000)
kp, des = surf.detectAndCompute(img, None)
img2 = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
plt.imshow(img2),plt.show()
print len(kp)