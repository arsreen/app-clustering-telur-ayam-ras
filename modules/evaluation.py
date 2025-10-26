from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clusters(X_scaled, labels):
    """
    Menghitung metrik evaluasi clustering:
    - Silhouette Score → semakin tinggi semakin baik.
    - Davies–Bouldin Index → semakin rendah semakin baik.
    """
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    return sil, dbi
