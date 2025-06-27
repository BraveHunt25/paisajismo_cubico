"""
extraer_frames_completos.py

Descripción:
Este script extrae todos los frames de varios videos almacenados en carpetas específicas,
guardando cada frame como una imagen PNG en carpetas organizadas por nombre del video (bioma).

Uso:
Ejecutar este script para convertir videos en secuencias de imágenes que serán usadas
como base para análisis posteriores, como clustering o anotación manual.

Dependencias:
- OpenCV (cv2)
- logging (módulo estándar)
- os (módulo estándar)

Asegúrate de tener instaladas las librerías necesarias, por ejemplo:
pip install opencv-python
"""

import cv2
import os
import logging

# Configurar logging para mostrar información en la consola
logging.basicConfig(level=logging.INFO)

# Diccionario con nombres de biomas y rutas a los videos originales
VIDEOS = {
    'bosque_abedul': './src/assets/original/bosque_abedul.mp4',
    'bosque_cerezo': './src/assets/original/bosque_cerezo.mp4',
    'bosque_helado': './src/assets/original/bosque_helado.mp4',
    'bosque_palido': './src/assets/original/bosque_palido.mp4',
    'jungla': './src/assets/original/jungla.mp4'
}

# Carpeta raíz donde se guardarán los frames extraídos
FRAMES_DIR = './src/assets/frames'

# Para cada video (bioma) en el diccionario...
for nombre, ruta_video in VIDEOS.items():
    # Crear carpeta de salida para este bioma si no existe
    ruta_salida = os.path.join(FRAMES_DIR, nombre)
    os.makedirs(ruta_salida, exist_ok=True)

    # Abrir el video con OpenCV
    cap = cv2.VideoCapture(ruta_video)

    # Obtener frames por segundo (fps) para calcular tiempos de frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    # Leer frames hasta que no queden más
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Salir si no hay más frames

        # Calcular el tiempo en segundos del frame actual
        tiempo = frame_idx / fps

        # Crear un nombre para el archivo del frame con índice y tiempo
        nombre_frame = f'frame_{frame_idx:05d}_t{tiempo:.3f}.png'

        # Guardar la imagen del frame en la carpeta correspondiente
        cv2.imwrite(os.path.join(ruta_salida, nombre_frame), frame)

        frame_idx += 1

    # Liberar el objeto VideoCapture
    cap.release()

    # Informar que la extracción para este video terminó
    logging.info(f"Frames extraídos para {nombre}")