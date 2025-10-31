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
# 🔹 Intelligent K-Medoids (stabil untuk data antar tahun)
# =============================================
def intelligent_kmedoids_auto(
    X, alpha=2.5, max_iter=200, tol=1e-5, max_cluster_pref=3, sil_target=0.5
):
    """
    Versi stabil dan seimbang dari Intelligent K-Medoids:
    ✅ Lebih stabil untuk data lintas tahun (multi-variabel).
    ✅ Tambah medoid hanya jika threshold tinggi & cluster imbalance.
    ✅ Berhenti bila silhouette ≥ 0.5 atau cluster sudah 3.
    """

    n = len(X)
    D = cdist(X, X, metric="euclidean")

    # === Hitung Center of Mass (CoM) ===
    m1 = np.argmax(np.mean(D, axis=1))
    m2 = np.argmax(D[m1])
    medoids = [m1, m2]

    prev_cost = np.inf
    prev_sil = -1
    no_improve_count = 0

    print("🚀 Intelligent K-Medoids (Stabil untuk Data Tahunan)...")

    for iteration in range(max_iter):
        # === Assign ke medoid terdekat ===
        dist_to_medoids = D[:, medoids]
        labels = np.argmin(dist_to_medoids, axis=1)
        cost = np.sum(np.min(dist_to_medoids, axis=1))

        # === Hitung silhouette ===
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = -1

        print(f"Iter {iteration+1:03d} | Cluster={len(medoids)} | Sil={sil:.4f} | Cost={cost:.4f}")

        # === Berhenti kalau silhouette sudah cukup tinggi ===
        if sil >= sil_target or len(medoids) >= max_cluster_pref:
            print(f"🛑 Stop: Silhouette ≥ {sil_target} ({sil:.3f}) atau cluster = {len(medoids)}.")
            break

        # === Cek konvergensi cost ===
        if abs(prev_cost - cost) < tol:
            print(f"✅ Konvergen di iterasi ke-{iteration+1}.")
            break

        prev_cost = cost

        # === Peningkatan silhouette stagnan ===
        if sil < prev_sil + 0.002:
            no_improve_count += 1
        else:
            no_improve_count = 0
        if no_improve_count >= 3:
            print("⚠️ Silhouette stagnan — stop iterasi.")
            break
        prev_sil = sil

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

        # === Threshold MAD adaptif ===
        dxi = np.min(D[:, medoids], axis=1)
        med = np.median(dxi)
        mad = median_absolute_deviation(dxi)
        threshold = med + alpha * mad

        # === Tambah medoid baru jika perlu ===
        far_points = np.where(dxi > threshold)[0]
        if len(far_points) > 0 and len(medoids) < max_cluster_pref:
            # 🧠 Pilih titik yang paling "representatif" dari jarak besar, bukan yang ekstrem
            new_medoid = far_points[np.argmax(np.mean(D[far_points][:, medoids], axis=1))]
            if new_medoid not in medoids:
                medoids.append(new_medoid)
                alpha += 0.2
                print(f"➕ Tambah medoid baru (total={len(medoids)}), α={alpha:.2f}")
        else:
            if not improved:
                break

    # === Re-cluster final (biar distribusi merata & logis) ===
    D_final = cdist(X, X[medoids], metric="euclidean")
    labels_final = np.argmin(D_final, axis=1)

    # === Perbaiki distribusi jika ada cluster < 5% data ===
    unique, counts = np.unique(labels_final, return_counts=True)
    min_size = 0.05 * len(X)
    small_clusters = [u for u, c in zip(unique, counts) if c < min_size]
    if small_clusters:
        print(f"⚠️ Ada cluster kecil: {small_clusters}, reassigning...")
        for sc in small_clusters:
            idx = np.where(labels_final == sc)[0]
            dist_rest = np.delete(D_final[idx], sc, axis=1)
            labels_final[idx] = np.argmin(dist_rest, axis=1)

    # === Hitung silhouette akhir ===
    try:
        final_sil = silhouette_score(X, labels_final)
    except Exception:
        final_sil = -1

    print(f"✅ Selesai: {len(np.unique(labels_final))} cluster | Silhouette akhir={final_sil:.4f}")
    return np.array(medoids), labels_final, cost, final_sil


# =============================================
# 🔹 Fungsi versi Streamlit
# =============================================
def run_intelligent_kmedoids_streamlit(X_scaled):
    """
    Wrapper untuk Streamlit.
    Mengembalikan labels, jumlah cluster otomatis, dan nilai silhouette.
    """
    medoids, labels, cost, sil = intelligent_kmedoids_auto(X_scaled, alpha=2.5)
    k_auto = len(np.unique(labels))
    print(f"✅ Jumlah cluster otomatis: {k_auto} | Silhouette: {sil:.4f}")
    return labels, k_auto, sil
