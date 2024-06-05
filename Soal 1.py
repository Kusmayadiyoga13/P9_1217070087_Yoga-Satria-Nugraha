import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('daun.jpg', 0)  
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobelx, sobely)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
canny = cv2.Canny(img, 100, 200)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.figure(figsize=(15, 8))
plt.subplot(2,3,1), plt.imshow(img, cmap='gray'), plt.title('Citra Asli')
plt.subplot(2,3,2), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X (Gx)')
plt.subplot(2,3,3), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y (Gy)')
plt.subplot(2,3,4), plt.imshow(sobel_magnitude, cmap='gray'), plt.title('Sobel Magnitude')
plt.subplot(2,3,5), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.subplot(2,3,6), plt.imshow(canny, cmap='gray'), plt.title('Canny')
plt.show()
plt.figure(figsize=(8, 5))
plt.imshow(thresh, cmap='gray'), plt.title('Hasil Segmentasi')
plt.show()

#Yoga Satria Nugraha 
#1217070087