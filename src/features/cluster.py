from sklearn.cluster import KMeans

def fit_kmeans(X_final, random_state: int = 42 , n=2):
    kmeans = KMeans(
        n_clusters=n,
        random_state=random_state,
        n_init="auto"
    )
    labels = kmeans.fit_predict(X_final)
    return kmeans, labels
