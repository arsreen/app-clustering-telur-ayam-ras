import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# =============================================
# 🔹 Fungsi bantu: Median Absolute Deviation
# =============================================
def median_absolute_deviation(data):
    med = np.median(data)
    return np.median(np.abs(data - med))


# =============================================
# 🔹 Fungsi utama: Intelligent K-Medoids Auto
# =============================================
def intelligent_kmedoids_auto(X, alpha=3.0, max_iter=200, tol=1e-5, max_cluster_pref=3):
    """
    Intelligent K-Medoids Auto-Cluster (versi adaptif dan stabil).
    Menambah medoid baru otomatis berdasar deviasi (MAD)
    dan berhenti bila silhouette score sudah stabil atau mencapai 3 cluster.
    """
    n = len(X)
    D = cdist(X, X, metric="euclidean")

    # === Inisialisasi dua medoid awal (paling jauh) ===
    m1 = np.argmax(np.mean(D, axis=1))
    m2 = np.argmax(D[m1])
    medoids = [m1, m2]
    prev_cost = np.inf
    prev_sil = -1
    no_improve_count = 0

    print("🚀 Menjalankan Intelligent K-Medoids Auto (Prefer 3 Cluster)...")
    for iteration in range(max_iter):
        # === Assign ke medoid terdekat ===
        dist_to_medoids = D[:, medoids]
        labels = np.argmin(dist_to_medoids, axis=1)
        cost = np.sum(np.min(dist_to_medoids, axis=1))

        # === Evaluasi silhouette score ===
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = -1

        print(f"Iter {iteration+1:03d} | medoid={len(medoids)} | silhouette={sil:.4f} | cost={cost:.4f}")

        # === Cek peningkatan silhouette ===
        if sil < prev_sil + 0.005:  # tidak naik signifikan
            no_improve_count += 1
        else:
            no_improve_count = 0
        prev_sil = sil

        # === Kondisi berhenti ===
        if (no_improve_count >= 2) or \
           (len(medoids) >= max_cluster_pref and no_improve_count >= 1):
            print(f"🛑 Berhenti di iterasi {iteration+1}: "
                  f"Jumlah cluster = {len(medoids)}, Silhouette stabil.")
            break

        if abs(prev_cost - cost) < tol:
            print(f"✅ Konvergen di iterasi ke-{iteration+1}.")
            break

        prev_cost = cost

        # === Swap step ===
        improved = False
        for i, m in enumerate(medoids):
            cluster_points = np.where(labels == i)[0]
            for p in cluster_points:
                temp_medoids = medoids.copy()
                temp_medoids[i] = p
                new_cost = np.sum(np.min(D[:, temp_medoids], axis=1))
                if new_cost < cost:
                    cost = new_cost
                    medoids = temp_medoids
                    improved = True

        # === Threshold adaptif (MAD) ===
        dxi = np.min(D[:, medoids], axis=1)
        med = np.median(dxi)
        mad = median_absolute_deviation(dxi)
        threshold = med + alpha * mad

        # === Tambah medoid baru kalau deviasi besar ===
        far_points = np.where(np.abs(dxi - med) > threshold)[0]
        if len(far_points) > 0:
            new_medoid = far_points[np.argmax(dxi[far_points])]
            if new_medoid not in medoids:
                medoids.append(new_medoid)
                alpha += 0.5
        else:
            if not improved:
                break

    # === Hasil akhir ===
    D_final = D[:, medoids]
    labels_final = np.argmin(D_final, axis=1)
    return np.array(medoids), labels_final, cost


# =============================================
# 🔹 Fungsi versi Streamlit
# =============================================
def run_intelligent_kmedoids_streamlit(X_scaled):
    """
    Wrapper agar bisa dipakai langsung di Streamlit.
    Mengembalikan labels dan jumlah cluster otomatis.
    """
    medoids, labels, cost = intelligent_kmedoids_auto(X_scaled, alpha=3.0)

    # jumlah cluster hasil auto
    k_auto = len(np.unique(labels))
    print(f"✅ Jumlah cluster otomatis: {k_auto}")
    return labels, k_auto
