import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

INPUT_PATH = './src/etiquetas_clusters_usuario.json'
OUTPUT_PATH = './src/etiquetas_agrupadas.json'

with open(INPUT_PATH, 'r') as f:
    etiquetas = json.load(f)

descripciones = [e['descripcion'] for e in etiquetas]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descripciones)

kmeans = KMeans(n_clusters=10, random_state=0)
labels = kmeans.fit_predict(X)

for i, etiqueta in enumerate(etiquetas):
    etiqueta['grupo'] = int(labels[i])

with open(OUTPUT_PATH, 'w') as f:
    json.dump(etiquetas, f, indent=2)