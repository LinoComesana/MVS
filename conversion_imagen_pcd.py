#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 19:47:23 2022

@author: lino
"""



import sys
sys.path.insert(1, '/home/lino/Documentos/programas_pruebas_varias/modulo_visualizacion') # Ruta a mi visor customizado y módulo de lecturas
import visualizaciones_customizadas_open3d as visor
import lecturas_nubes_de_puntos as lectura





import numpy as np
import os
import open3d as o3d
import PIL

archivo_imagen_profundidad = '/fotograma_3_depth_map.png'

ruta_script = os.getcwd()
ruta_imagen = ruta_script + '/ejemplos'

pixeles_imagen = np.asarray(PIL.Image.open(ruta_imagen+archivo_imagen_profundidad))
alto_imagen = pixeles_imagen.shape[0]
ancho_imagen = pixeles_imagen.shape[1]

# Hay una cuarta columna a parte de la de RGB que vale siempre 255. No la tengo 
# en cuenta:
pixeles_imagen = pixeles_imagen[:,:,0:3]

# Como los mapas de profundidad los guardo en escalas de blanco-negro, siendo
# blanco píxeles próximos a la cámara y negro alejados, puedo considerar la su-
# ma de cada elemento en los vectores RGB y aquellos que tengan una suma mayor
# estarán más cerca de la cámara:
    
pixeles_sumas = np.sum(pixeles_imagen,axis=2)

Lista_puntos = []
Lista_colores = []

eje_x = np.linspace(0,ancho_imagen,ancho_imagen)
eje_y = np.linspace(0,alto_imagen,alto_imagen)

for i in range(len(eje_y)): # filas (eje y)
    for j in range(len(eje_x)): # columnas (eje x)
        Lista_puntos.append([-j,i,pixeles_sumas[i][j]])
        Lista_colores.append(pixeles_imagen[i][j])
        
puntos = np.array(Lista_puntos)
colores = np.array(Lista_colores)/255.

nube = o3d.geometry.PointCloud()
nube.points = o3d.utility.Vector3dVector(puntos)
nube.colors = o3d.utility.Vector3dVector(colores)
nube.estimate_normals()
normales = np.array(nube.normals)


nube_eje = o3d.geometry.PointCloud()
puntos_eje = np.array([[0,0,0],[0,0,10]])
colores_eje = np.array([[0,0,1],[0,0,1]])
nube_eje.points = o3d.utility.Vector3dVector(puntos_eje)
nube_eje.colors = o3d.utility.Vector3dVector(colores_eje)
# Parón para visualizar
# visor.custom_draw_geometry_with_key_callback(nube)
o3d.visualization.draw(nube+nube_eje)






def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


angulos = []

for i in range(len(normales)):
    angulo = angle_between(normales[i], puntos_eje[1])
    angulos.append(angulo)

PUNTOS_DEF = []
COLORES_DEF = []
NORMALES_DEF = []

for i in range(len(angulos)):
    angulo = angulos[i]
    if angulo == np.pi:
        PUNTOS_DEF.append(puntos[i])
        COLORES_DEF.append(colores[i])
        NORMALES_DEF.append(normales[i])
    
NUBE = o3d.geometry.PointCloud()
NUBE.points = o3d.utility.Vector3dVector(PUNTOS_DEF)
NUBE.colors = o3d.utility.Vector3dVector(COLORES_DEF)
NUBE.normals = o3d.utility.Vector3dVector(NORMALES_DEF)
# Parón para visualizar
o3d.visualization.draw(NUBE)




