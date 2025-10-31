import pandas as pd
import numpy as np

def interpretasi_cluster_otomatis(df, var):
    """
    Fungsi untuk memberikan interpretasi otomatis berdasarkan distribusi nilai per cluster.
    Berdasarkan median (untuk posisi) dan IQR/std (untuk stabilitas).
    Output: dict {cluster: deskripsi singkat}
    """
    if "Cluster" not in df.columns:
        raise ValueError("Kolom 'Cluster' tidak ditemukan dalam DataFrame!")

    stats = df.groupby("Cluster")[var].agg(
        median="median",
        mean="mean",
        std="std",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75)
    ).reset_index()

    stats["iqr"] = stats["q3"] - stats["q1"]
    stats = stats.sort_values("median").reset_index(drop=True)

    interpretasi = {}
    for i, row in stats.iterrows():
        cluster = int(row["Cluster"])
        median = row["median"]
        iqr = row["iqr"]

        # Posisi relatif (rendah, sedang, Tinggi)
        if i == 0:
            posisi = "rendah"
        elif i == len(stats) - 1:
            posisi = "tinggi"
        else:
            posisi = "sedang"

        # Stabilitas (berdasarkan IQR)
        if iqr < stats["iqr"].median() * 0.8:
            stabilitas = "stabil"
        elif iqr > stats["iqr"].median() * 1.2:
            stabilitas = "fluktuatif"
        else:
            stabilitas = "cukup stabil"

        interpretasi[cluster] = f"{var} {posisi} dan {stabilitas}"

    return interpretasi


def interpretasi_multi_variabel(df, fitur):
    """
    Versi multi-variabel: menghasilkan interpretasi untuk semua variabel sekaligus.
    Output: dict {variabel: {cluster: deskripsi}}
    """
    hasil = {}
    for var in fitur:
        hasil[var] = interpretasi_cluster_otomatis(df, var)
    return hasil


def tampilkan_interpretasi_streamlit(st, df, fitur):
    """
    Fungsi helper agar bisa langsung dipakai di Streamlit.
    Menampilkan hasil interpretasi otomatis dalam format tabel rapi.
    """
    hasil = interpretasi_multi_variabel(df, fitur)

    st.subheader("🧠 Interpretasi Otomatis dari Distribusi Boxplot")
    for var, interp in hasil.items():
        st.markdown(f"**📊 {var}**")
        df_tampil = pd.DataFrame(list(interp.items()), columns=["Cluster", "Interpretasi"])
        st.dataframe(df_tampil, hide_index=True, use_container_width=True)

    return hasil


import numpy as np
import pandas as pd

def interpretasi_untuk_legend_otomatis(df, fitur, cluster_palette=None):
    """
    Menghasilkan label interpretasi otomatis berdasarkan median tiap variabel per cluster.
    - 1 variabel → "Harga tinggi", "Harga rendah"
    - 2 variabel → gabungan dua deskripsi (contoh: "Harga tinggi, Konsumsi rendah")
    - ≥3 variabel → tetap deskriptif per variabel, tanpa kategori 'Pasar'
    """

    cluster_labels = {}
    cluster_ids = sorted(df["Cluster"].unique())
    n_clusters = len(cluster_ids)

    # Label level berdasarkan jumlah cluster
    label_levels = {
        2: ["Rendah", "Tinggi"],
        3: ["Rendah", "Sedang", "Tinggi"],
        4: ["Sangat Rendah", "Rendah", "Tinggi", "Sangat Tinggi"],
        5: ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"],
        6: ["Sangat Rendah", "Rendah", "Cukup Rendah", "Cukup Tinggi", "Tinggi", "Sangat Tinggi"],
        7: ["Sangat Rendah", "Rendah", "Cukup Rendah", "Sedang", "Cukup Tinggi", "Tinggi", "Sangat Tinggi"],
    }
    label_dipakai = label_levels.get(n_clusters, label_levels[7])

    # Hitung median tiap cluster
    median_per_cluster = df.groupby("Cluster")[fitur].median()

    # Fungsi bantu: ubah median jadi label
    def nilai_ke_label(series):
        sorted_clusters = series.sort_values().index.tolist()
        mapping = {cl: label_dipakai[i] for i, cl in enumerate(sorted_clusters)}
        return series.index.map(mapping)

    # ======================
    # 🔸 Interpretasi Umum
    # ======================
    for cluster in cluster_ids:
        label_parts = []
        for var in fitur:
            sorted_clusters = median_per_cluster[var].sort_values().index.tolist()
            rank = sorted_clusters.index(cluster)
            rank = min(rank, len(label_dipakai) - 1)
            deskripsi = f"{var.split('(')[0].strip()} {label_dipakai[rank]}"
            label_parts.append(deskripsi)
        cluster_labels[cluster] = ", ".join(label_parts)

    return cluster_labels




