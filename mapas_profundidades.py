#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 11:50:34 2022

@author: lino
"""
import os
import inspect
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation

print(os.path.dirname(inspect.getfile(GLPNForDepthEstimation))  )
    
    
    
ruta_script = '/home/lino/Documentos/personal/LiDAR_movil'
ruta_fotogramas = ruta_script+'/fotogramas'
# os.chdir(ruta_fotogramas)
archivo_fotograma_1 = ruta_fotogramas+'/fotograma_2.jpeg'







feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
# model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu",from_tf=True)
model = GLPNForDepthEstimation.from_pretrained("/home/lino/Descargas/glpn-nyu")


# load and resize the input image
image = Image.open(archivo_fotograma_1)
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# get the prediction from the model
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# remove borders
pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

# visualize the prediction
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[0].set_title('Imagen original')
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].set_title('Mapa de profundidades')
fig.colorbar(ax[1].imshow(output, cmap='plasma'), orientation='vertical')
plt.tight_layout()
plt.pause(5)