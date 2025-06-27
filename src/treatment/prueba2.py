"""
extraer_histogramas.py

Descripción:
Este script recorre una estructura de carpetas con imágenes (frames) organizadas por bioma,
calcula un histograma de color para cada imagen y guarda la información en un archivo JSON.
El histograma se utiliza para representar la distribución de colores en cada imagen,
lo cual puede servir para análisis posteriores como clustering o clasificación.

Uso:
Ejecutar este script después de haber extraído los frames de los videos.
Genera una base de datos en formato JSON con histogramas normalizados para cada imagen.

Dependencias:
- OpenCV (cv2)
- NumPy
- glob (módulo estándar)
- os (módulo estándar)
- json (módulo estándar)

Asegúrate de tener instaladas las librerías necesarias, por ejemplo:
pip install opencv-python numpy
"""

import os
import cv2
import numpy as np
import json
from glob import glob

# Ruta raíz donde están las carpetas con imágenes (cada carpeta es un bioma)
FRAME_ROOT = './src/assets/frames'

# Archivo JSON donde se guardará la base de datos con histogramas
SALIDA_JSON = './src/base_indexada.json'

# Lista para almacenar los datos de cada imagen
base_datos = []

# Recorrer cada carpeta (bioma) dentro de FRAME_ROOT
for carpeta_bioma in os.listdir(FRAME_ROOT):
    ruta_bioma = os.path.join(FRAME_ROOT, carpeta_bioma)

    # Buscar todas las imágenes PNG dentro de esta carpeta
    frames = sorted(glob(os.path.join(ruta_bioma, '*.png')))

    for path in frames:
        # Leer la imagen
        imagen = cv2.imread(path)
        if imagen is None:
            # Si la imagen no se puede leer, saltarla
            continue

        # Calcular histograma de color en los 3 canales (B,G,R) con 8 bins por canal
        hist = cv2.calcHist([imagen], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # Normalizar el histograma para que la suma sea 1 y aplanarlo en un vector 1D
        hist = cv2.normalize(hist, hist).flatten().tolist()

        # Añadir la información a la base de datos
        base_datos.append({
            'bioma': carpeta_bioma,
            'path': path,
            'histograma': hist
        })

# Guardar toda la base de datos en un archivo JSON con formato legible
with open(SALIDA_JSON, 'w') as f:
    json.dump(base_datos, f, indent=2)