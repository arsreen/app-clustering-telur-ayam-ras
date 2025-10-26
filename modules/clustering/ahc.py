from sklearn.cluster import AgglomerativeClustering

def run_ahc(X_scaled, k):
    """
    Menjalankan algoritma Agglomerative Hierarchical Clustering (AHC)
    dengan single linkage.
    
    Parameter:
    - X_scaled: data hasil normalisasi (numpy array atau DataFrame)
    - k: jumlah cluster
    """
    try:
        # ✅ versi scikit-learn >= 1.2
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            metric="euclidean",       # tambahkan metrik eksplisit
            compute_distances=False   # hemat waktu & memori
        )
    except TypeError:
        # ✅ fallback untuk scikit-learn < 1.2
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            affinity="euclidean"
        )

    labels = model.fit_predict(X_scaled)
    return labels
