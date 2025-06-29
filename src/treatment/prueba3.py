"""
extraer_frames_muestra.py

Descripción:
Este script selecciona frames de muestra a intervalos regulares (cada cierto número de segundos)
de una colección de carpetas con imágenes (frames) organizadas por bioma.
Los frames seleccionados se copian a una carpeta de salida con un prefijo que indica el bioma.

Uso:
Ejecutar este script después de extraer todos los frames completos.
Permite reducir la cantidad de frames para anotación o análisis posterior, 
seleccionando uno cada INTERVAL_SECONDS segundos.

Dependencias:
- OpenCV (cv2)
- os (módulo estándar)

Nota:
Se asume que los frames originales fueron extraídos a 60 fps, por eso se multiplica INTERVAL_SECONDS por 60 para seleccionar cada N-ésimo frame.
"""

import os
import cv2

# Carpeta raíz con frames originales organizados en subcarpetas por bioma
FRAME_INPUT_ROOT = './src/assets/frames'

# Carpeta donde se guardarán los frames seleccionados
FRAME_OUTPUT_ROOT = './src/frames_muestra'

# Intervalo en segundos para seleccionar un frame (ejemplo: cada 5 segundos)
INTERVAL_SECONDS = 30

# Crear carpeta de salida si no existe
os.makedirs(FRAME_OUTPUT_ROOT, exist_ok=True)

# Recorrer cada carpeta (bioma) dentro de la carpeta de frames originales
for bioma in os.listdir(FRAME_INPUT_ROOT):
    carpeta = os.path.join(FRAME_INPUT_ROOT, bioma)
    
    # Obtener lista ordenada de todos los archivos PNG en la carpeta
    frames = sorted([f for f in os.listdir(carpeta) if f.endswith('.png')])
    
    # Seleccionar un frame cada INTERVAL_SECONDS * 60 frames (suponiendo 60 fps)
    seleccionados = frames[::INTERVAL_SECONDS * 60]
    
    # Copiar los frames seleccionados a la carpeta de salida, renombrando con prefijo de bioma
    for f in seleccionados:
        src = os.path.join(carpeta, f)
        dst = os.path.join(FRAME_OUTPUT_ROOT, f"{bioma}_{f}")
        
        # Leer y escribir la imagen para copiarla
        cv2.imwrite(dst, cv2.imread(src))