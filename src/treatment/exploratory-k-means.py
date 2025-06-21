import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 1. Cargar imagen
image_path = r'src\assets\frames\bosque_cerezo\frame_05444_t94.639.png'
logging.info(f'Cargando imagen desde: {image_path}')
image = cv2.imread(image_path)

if image is None:
    logging.error("No se pudo cargar la imagen. Verifica la ruta.")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape
logging.info(f'Tamaño de la imagen: {h}x{w}, Canales: {c}')

# 2. Vector de colores RGB
pixels = image.reshape((-1, 3))
logging.info(f'Número de píxeles procesados: {pixels.shape[0]}')

# 3. Normalizar
logging.info('Normalizando características RGB...')
scaler = StandardScaler()
pixels_scaled = scaler.fit_transform(pixels)

# 4. Aplicar K-Means
k = 5
logging.info(f'Aplicando K-Means con k={k}')
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels_scaled)
labels = kmeans.labels_
logging.info('Entrenamiento de K-Means completado.')

# 5. Imagen segmentada general y máscaras individuales
output_image = np.zeros_like(pixels)
cluster_images = []

for i in range(k):
    idxs = (labels == i)
    cluster_colors = pixels[idxs]
    avg_color = np.mean(cluster_colors, axis=0)
    output_image[idxs] = avg_color

    logging.info(f'Cluster {i}: {len(cluster_colors)} píxeles, Color promedio: {avg_color.astype(int)}')

    # Crear imagen con solo los píxeles del clúster en color y fondo negro
    cluster_img = np.zeros_like(pixels)
    cluster_img[idxs] = avg_color
    cluster_img = cluster_img.reshape((h, w, 3)).astype(np.uint8)
    cluster_images.append(cluster_img)

# 6. Reconstruir imagen segmentada
output_image = output_image.reshape((h, w, 3)).astype(np.uint8)

# 7. Mostrar resultados
logging.info('Mostrando resultados...')
n_cols = 3
n_rows = int(np.ceil((k + 2) / n_cols))

fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axs = axs.flatten()

# Imagen original
axs[0].imshow(image)
axs[0].set_title("Imagen original")
axs[0].axis('off')

# Imagen segmentada total
axs[1].imshow(output_image)
axs[1].set_title(f"Segmentación K-Means (k={k})")
axs[1].axis('off')

# Imágenes por clúster
for i in range(k):
    axs[i + 2].imshow(cluster_images[i])
    axs[i + 2].set_title(f"Cluster {i}")
    axs[i + 2].axis('off')

# Eliminar subplots vacíos (si los hay)
for j in range(k + 2, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()
