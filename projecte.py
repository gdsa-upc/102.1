import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/HP/UNI/6. TERCER + QUART/Q5 + Q7/GDSA/102.1/images/people.jpg',1)
surf = cv2.SURF(4000)
kp, des = surf.detectAndCompute(img, None)
img2 = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
plt.imshow(img2),plt.show()
print len(kp)