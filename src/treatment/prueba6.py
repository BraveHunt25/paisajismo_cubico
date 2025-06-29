"""
bioma_prediccion.py

Descripción:
Este script selecciona aleatoriamente un frame de una carpeta de imágenes de biomas, realiza
clustering K-Means sobre los píxeles de la imagen para identificar regiones similares, y luego
muestra parches representativos para que el usuario los describa.

Con base en las descripciones ingresadas, el script compara el texto con una base previa vectorizada,
predice a qué bioma corresponde cada parche y finalmente determina el bioma más probable del frame.
También reproduce el resultado hablado usando síntesis de voz.

Dependencias:
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn (KMeans, TfidfVectorizer, cosine_similarity)
- pyttsx3
- json, os, random
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

# --- CONFIGURACIÓN ---
FRAME_INPUT_ROOT = './src/assets/frames'
JSON_DESCRIPCIONES = './src/etiquetas_agrupadas.json'
TAM_PARCHE = 100
K = 3  # Clusters K-means

# --- SELECCIONAR FRAME ALEATORIO ---
biomas = os.listdir(FRAME_INPUT_ROOT)
bioma_random = random.choice(biomas)
carpeta = os.path.join(FRAME_INPUT_ROOT, bioma_random)
frame_random = random.choice([f for f in os.listdir(carpeta) if f.endswith('.png')])
img_path = os.path.join(carpeta, frame_random)

# --- LEER IMAGEN ---
img = cv2.imread(img_path)
if img is None:
    print("Error al cargar imagen")
    exit()

# --- MOSTRAR IMAGEN COMPLETA ---
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Imagen completa del frame seleccionado")
plt.show()

# --- K-MEANS ---
h, w, _ = img.shape
pixels = img.reshape((-1, 3))
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_.reshape(h, w)

# --- VISUALIZAR ZONAS SEGMENTADAS POR K-MEANS ---
colores = (np.random.rand(K, 3) * 255).astype(np.uint8)  # Colores aleatorios para cada cluster
img_segmentada = colores[labels]  # Asignar color a cada pixel según su cluster

plt.figure(figsize=(8, 6))
plt.imshow(img_segmentada)
plt.axis('off')
plt.title("Zonas identificadas por K-Means")
plt.show()

# --- CARGAR BASE DE DESCRIPCIONES ---
with open(JSON_DESCRIPCIONES, 'r') as f:
    base = json.load(f)

descripciones = [e['descripcion'] for e in base]
biomas_asociados = ['_'.join(e['frame'].split('_')[:2]) for e in base]

# --- VECTORIZAR BASE DE TEXTO ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descripciones)

conteo_biomas = {}

# --- PREDICCIÓN POR CLUSTER ---
for i in range(K):
    ys, xs = np.where(labels == i)
    if len(xs) == 0:
        continue

    intentos = 0
    parche = None

    while intentos < 10:
        idx = np.random.randint(len(xs))
        x, y = xs[idx], ys[idx]
        x, y = max(0, x - TAM_PARCHE // 2), max(0, y - TAM_PARCHE // 2)
        parche = img[y:y + TAM_PARCHE, x:x + TAM_PARCHE]
        if parche.shape[0] == TAM_PARCHE and parche.shape[1] == TAM_PARCHE:
            break
        intentos += 1

    if parche is None or parche.shape[0] != TAM_PARCHE or parche.shape[1] != TAM_PARCHE:
        continue

    plt.imshow(cv2.cvtColor(parche, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Parche del Cluster {i}')
    plt.show()

    descripcion = input("Describe el parche mostrado: ")

    input_vec = vectorizer.transform([descripcion])
    similitudes = cosine_similarity(input_vec, X).flatten()
    top_idx = np.argmax(similitudes)
    bioma_predicho = biomas_asociados[top_idx]

    conteo_biomas[bioma_predicho] = conteo_biomas.get(bioma_predicho, 0) + 1

# --- DECISIÓN FINAL ---
if conteo_biomas:
    pred_bioma = max(conteo_biomas, key=conteo_biomas.get)
    print(f"\n🔍 Bioma predicho: {pred_bioma}")

    # --- FORMATEAR NOMBRE DE BIOMA PARA VOZ ---
    def formatear_bioma(nombre):
        mapa = {
            'bosque_abedul': 'Bosque de abedules',
            'bosque_cerezo': 'Bosque de cerezos',
            'bosque_helado': 'Bosque helado',
            'bosque_palido': 'Bosque pálido',
            'jungla': 'Jungla'
        }
        return mapa.get(nombre, nombre.replace('_', ' ').capitalize())

    texto_bioma = formatear_bioma(pred_bioma)
    print(f"🗣️ Descripción hablada: {texto_bioma}")

    engine = pyttsx3.init()
    engine.say(f"Este bioma es probablemente {texto_bioma}")
    engine.runAndWait()
else:
    print("No se pudo predecir el bioma.")