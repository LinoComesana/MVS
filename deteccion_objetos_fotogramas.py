#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:54:09 2022

@author: lino
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
import cv2
from skimage.io import imread, imshow
from skimage.color import rgb2hsv


ruta_fotogramas = os.getcwd()+'/fotogramas'
os.chdir(ruta_fotogramas)
archivo_fotograma = 'fotograma_2_depth_map.png'

 
# Leemos la imagen:
pixeles_imagen_original = np.asarray(PIL.Image.open(archivo_fotograma))
alto_imagen = pixeles_imagen_original.shape[0]
ancho_imagen = pixeles_imagen_original.shape[1]

# Procesamos la imagen para detectar objetos:

    


img = cv2.imread(archivo_fotograma)
imagen_original = np.copy(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

Z = np.float32(img.reshape((-1,3)))

N_clusters = 20

# Define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(Z, N_clusters, None, criteria, 6, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make the original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = np.uint8(label.reshape(img.shape[:2]))
# res2.shape

# plt.imshow(res2)
# plt.show()



# PARÓN PARA VISUALIZAR
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].imshow(imagen_original)
ax[0].set_title('Imagen original')

ax[1].imshow(res2)
ax[1].set_title('K-MEANS')

fig.tight_layout()


clases = np.unique(res2)
print(clases)










