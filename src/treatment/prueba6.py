"""
bioma_prediccion.py

Descripción:
Este script selecciona aleatoriamente un frame de una carpeta de imágenes de biomas, realiza
clustering K-Means sobre los píxeles de la imagen para identificar regiones similares, y luego
muestra parches representativos para que el usuario los describa.

Con base en las descripciones ingresadas, el script compara el texto con una base previa vectorizada,
predice a qué bioma corresponde cada parche y finalmente determina el bioma más probable del frame.
También reproduce el resultado hablado usando síntesis de voz.

Uso:
Para interactuar con imágenes segmentadas en clusters y predecir biomas de forma semi-automatizada.

Dependencias:
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn (KMeans, TfidfVectorizer, cosine_similarity)
- pyttsx3 (síntesis de voz)
- json, os, random (módulos estándar)
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3

# --- CONFIGURACIONES ---
FRAME_INPUT_ROOT = './src/assets/frames'          # Carpeta raíz con frames de biomas
JSON_DESCRIPCIONES = './src/etiquetas_agrupadas.json'  # Archivo JSON con descripciones etiquetadas y agrupadas
TAM_PARCHE = 50                                   # Tamaño del parche cuadrado para mostrar
K = 3                                             # Número de clusters para K-Means

# --- SELECCIONAR FRAME ALEATORIO ---
biomas = os.listdir(FRAME_INPUT_ROOT)             # Listar carpetas de biomas
bioma_random = random.choice(biomas)               # Escoger un bioma al azar
carpeta = os.path.join(FRAME_INPUT_ROOT, bioma_random)
frame_random = random.choice([f for f in os.listdir(carpeta) if f.endswith('.png')])  # Escoger frame aleatorio
img_path = os.path.join(carpeta, frame_random)

# Leer imagen
img = cv2.imread(img_path)
if img is None:
    print("Error al cargar imagen")
    exit()

# Preparar datos para clustering
h, w, _ = img.shape
pixels = img.reshape((-1, 3))

# Aplicar K-Means para segmentar la imagen en K clusters
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_.reshape(h, w)

# --- CARGAR BASE DE DESCRIPCIONES ---
with open(JSON_DESCRIPCIONES, 'r') as f:
    base = json.load(f)

# Extraer descripciones y asociarlas con biomas (a partir del nombre del frame)
descripciones = [e['descripcion'] for e in base]
biomas_asociados = [e['frame'].split('_')[0] for e in base]

# Vectorizar todas las descripciones de la base para comparación posterior
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descripciones)

conteo_biomas = {}  # Diccionario para contar predicciones de bioma

# --- PREDICCIÓN POR CADA CLUSTER ---
for i in range(K):
    ys, xs = np.where(labels == i)  # Coordenadas de píxeles que pertenecen al cluster i
    if len(xs) == 0:
        continue
    
    idx = np.random.randint(len(xs))  # Elegir un píxel aleatorio dentro del cluster
    x, y = xs[idx], ys[idx]
    x, y = max(0, x - TAM_PARCHE//2), max(0, y - TAM_PARCHE//2)  # Ajustar para obtener parche centrado
    
    parche = img[y:y+TAM_PARCHE, x:x+TAM_PARCHE]
    if parche.shape[0] != TAM_PARCHE or parche.shape[1] != TAM_PARCHE:
        continue

    # Mostrar parche al usuario para que lo describa
    plt.imshow(cv2.cvtColor(parche, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Cluster {i}')
    plt.show()

    descripcion = input("Describe el parche mostrado: ")

    # Vectorizar la descripción ingresada y calcular similitud con la base
    input_vec = vectorizer.transform([descripcion])
    similitudes = cosine_similarity(input_vec, X).flatten()

    # Encontrar el índice de la descripción más similar
    top_idx = np.argmax(similitudes)
    bioma_predicho = biomas_asociados[top_idx]

    # Contar cuántas veces se predice cada bioma
    conteo_biomas[bioma_predicho] = conteo_biomas.get(bioma_predicho, 0) + 1

# --- DECISIÓN FINAL ---
if conteo_biomas:
    # Seleccionar el bioma con más votos
    pred_bioma = max(conteo_biomas, key=conteo_biomas.get)
    print(f"\n🔍 Bioma predicho: {pred_bioma}")

    # --- HABLAR ---
    engine = pyttsx3.init()
    engine.say(f"Este bioma es probablemente {pred_bioma}")
    engine.runAndWait()
else:
    print("No se pudo predecir el bioma.")