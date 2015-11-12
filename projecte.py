from skimage import data
from skimage.feature import CENSURE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img1 = rgb2gray(data.astronaut())

detector = CENSURE()

plt.gray()

detector.detect(img1)

plt.imshow(img1)
plt.axis('off')
plt.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')


plt.show()