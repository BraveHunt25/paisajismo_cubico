import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans

# Carpeta con las imágenes de entrada (frames seleccionados previamente)
IMAGES_DIR = './src/frames_muestra'

# Archivo JSON donde se guardarán las descripciones hechas por el usuario
RESULTADOS_PATH = './src/etiquetas_clusters_usuario.json'

# Tamaño del parche cuadrado que se extraerá de cada cluster (en pixeles)
TAM_PARCHE = 100

# Número de clusters para segmentar la imagen con K-Means
K = 3

# Lista donde se almacenarán los resultados de anotaciones
resultados = []

# Obtener la lista de archivos en la carpeta de imágenes, ordenada alfabéticamente
archivos = sorted(os.listdir(IMAGES_DIR))
total = len(archivos)

# Procesar cada imagen (frame) una a una
for idx, archivo in enumerate(archivos):
    print(f"\nProcesando frame {idx + 1}/{total}: {archivo}")
    
    img_path = os.path.join(IMAGES_DIR, archivo)
    img = cv2.imread(img_path)
    if img is None:
        print("Imagen no cargada correctamente, se omite.")
        continue

    h, w, _ = img.shape
    
    # Reorganizar los pixeles para que K-Means trabaje con ellos como muestras RGB
    pixels = img.reshape((-1, 3))

    # Aplicar K-Means para segmentar pixeles en K clusters
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(pixels)
    
    # Reorganizar etiquetas para que coincidan con la forma original de la imagen
    labels = kmeans.labels_.reshape(h, w)

    # Para cada cluster, tomar un parche aleatorio y pedir descripción al usuario
    for i in range(K):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue
        
        idx_random = np.random.randint(len(xs))
        x, y = xs[idx_random], ys[idx_random]
        
        # Ajustar para extraer el parche centrado en el punto seleccionado
        x, y = max(0, x - TAM_PARCHE // 2), max(0, y - TAM_PARCHE // 2)
        parche = img[y:y+TAM_PARCHE, x:x+TAM_PARCHE]
        
        # Asegurarse que el parche tiene el tamaño correcto
        if parche.shape[0] != TAM_PARCHE or parche.shape[1] != TAM_PARCHE:
            continue

        # Mostrar el parche para anotación manual
        plt.imshow(cv2.cvtColor(parche, cv2.COLOR_BGR2RGB))
        plt.title(f'{archivo} - Cluster {i}')
        plt.axis('off')
        plt.show()

        # Solicitar al usuario que describa el parche
        descripcion = input("Describe el parche mostrado: ")
        
        # Guardar la descripción junto con información del parche y cluster (convertido a int)
        resultados.append({
            'frame': archivo,
            'x': int(x),
            'y': int(y),
            'cluster': int(i),
            'descripcion': descripcion
        })

# Guardar todas las anotaciones en un archivo JSON
with open(RESULTADOS_PATH, 'w') as f:
    json.dump(resultados, f, indent=2)

print(f"\nProceso completado. Resultados guardados en: {RESULTADOS_PATH}")