import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread( 'test_norm.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
mean = np.mean(gray)
std = np.std(gray)
norm=(gray-mean)/std
plt.subplot(121)
plt.imshow(gray, 'gray')
plt.title('gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(norm, 'gray')
plt.title('normalization')
plt.axis('off'),plt.show()