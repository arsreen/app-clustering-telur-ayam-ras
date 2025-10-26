from sklearn.cluster import KMeans

def run_kmeans(X_scaled, k):
    """
    Menjalankan algoritma K-Means clustering.
    Parameter:
    - X_scaled: data hasil normalisasi
    - k: jumlah cluster
    """
    model = KMeans(
        n_clusters=k,
        n_init=10,     # jumlah inisialisasi ulang
        max_iter=300   # batas iterasi biar gak stuck
    )
    labels = model.fit_predict(X_scaled)
    return labels
