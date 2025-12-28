from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def find_best_k_with_balance(X, k_range=range(2, 10), min_frac=0.05):
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Tamaños de clusters
        unique, counts = np.unique(labels, return_counts=True)
        props = counts / len(labels)

        # descartar clusters muy pequeños
        if np.any(props < min_frac):
            print(f"Skipping k={k} (cluster demasiado pequeño: {counts})")
            continue

        score = silhouette_score(X, labels)
        results.append((k, score))

        print(f"k={k} | Silhouette={score:.4f} | Clusters={counts}")

    return results