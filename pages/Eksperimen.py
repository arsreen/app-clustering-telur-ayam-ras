from modules.evaluation import evaluate_clusters
from modules.clustering import run_kmeans, run_ahc, run_intelligent_kmedoids_streamlit
from modules.preprocessing import clean_data, scale_features
from modules.interpretasi import (
    interpretasi_cluster_otomatis,
    tampilkan_interpretasi_streamlit
)

import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from sklearn.metrics import silhouette_samples
import seaborn as sns

import streamlit as st

# === Hilangkan semua elemen toolbar, ikon orang, dan footer dengan CSS hardcore ===
st.set_page_config(page_title="Clustering Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Hilangkan toolbar (Stop / Deploy / Rerun) */
    [data-testid="stToolbar"] {
        visibility: hidden !important;
        height: 0px !important;
        position: fixed !important;
        top: -100px !important;
    }

    /* Hilangkan ikon orang (aksesibilitas / help) */
    [title="Accessibility"],
    button[title="View app menu"],
    button[title="Manage app"],
    button[title="Settings"],
    div[data-testid="stActionButton"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        position: fixed !important;
        z-index: -1 !important;
    }

    /* Hilangkan elemen floating kanan atas */
    div[role="button"] img,
    div[role="button"] svg,
    div[data-testid="stBaseButton-headerNoPadding"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }

    /* Hilangkan footer Streamlit */
    footer, #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)


# ========================
# 🔹 IMPORT MODULES TAMBAHAN
# ========================

# ========================
# 🔹 STREAMLIT CONTENT
# ========================
st.title("🔎 Eksperimen Clustering")

st.markdown("""
Halaman ini digunakan untuk melakukan **eksperimen clustering** terhadap data harga, konsumsi, 
dan pengeluaran telur ayam ras di Indonesia.  
Pilih mode eksperimen, lalu tentukan sumber dataset yang akan digunakan.
""")

# ========================
# TAB 1: Eksperimen Interaktif
# ========================
tab1, tab2, tab3 = st.tabs(
    ["🧩 Eksperimen Satu Metode", "⚖️ Perbandingan Dua Metode", "🏆 Perbandingan Tiga Metode"])

with tab1:
    st.header("Eksperimen Satu Metode")

    # =========================================================
    # 🧹 AUTO RESET jika user ganti pilihan
    # =========================================================
    def reset_experiment_state():
        for key in ["X_scaled", "labels", "df", "sil", "dbi", "time", "metode", "k"]:
            if key in st.session_state:
                del st.session_state[key]


    # =========================================================
    # 📂 PILIH SUMBER DATASET & BACA FILE
    # =========================================================
    import io
    import re
    import pandas as pd

    # --- Pilih sumber dataset ---
    if "pilihan_data_tab1" not in st.session_state:
        st.session_state["pilihan_data_tab1"] = "Gunakan dataset bawaan"

    pilihan_data = st.radio(
        "Pilih sumber dataset:",
        ["Gunakan dataset bawaan", "Upload dataset sendiri"],
        horizontal=True,
        key="pilihan_data_tab1_radio"
    )

        # 🔁 RESET TAB 2 & TAB 3 JIKA MODE DIGANTI
    if "last_dataset_mode" not in st.session_state:
        st.session_state["last_dataset_mode"] = pilihan_data
    elif st.session_state["last_dataset_mode"] != pilihan_data:
        st.session_state["last_dataset_mode"] = pilihan_data
        st.session_state.pop("upload_tab2_validasi", None)
        st.session_state.pop("upload_tab3_validasi", None)
        st.session_state.pop("pilihan_data_tab2_radio", None)
        st.session_state.pop("pilihan_data_tab3_radio", None)
        st.rerun()

    # --- Baca file sesuai pilihan ---
    xls = None  # siapkan variabel default

    if pilihan_data == "Gunakan dataset bawaan":
        try:
            excel_path = "data/Dataset Ready.xlsx"
            xls = pd.ExcelFile(excel_path)
        except Exception as e:
            st.error(f"❌ Gagal membaca dataset bawaan: {e}")
            st.stop()

    else:
        uploaded_file = st.file_uploader(
            "📂 Unggah file Excel (.xlsx)",
            type=["xlsx"],
            key="upload_tab1_validasi",
            on_change=reset_experiment_state
        )

        # ⛔ Jangan hentikan seluruh tab kalau belum upload
        if not uploaded_file:
            st.info("ℹ️ Silakan upload dataset terlebih dahulu untuk melanjutkan analisis di tab ini.")
        else:
            try:
                xls = pd.ExcelFile(uploaded_file)
            except Exception as e:
                st.error(f"❌ Gagal membaca file Excel: {e}")
                xls = None

    # =========================================================
    # 🚦 Cek apakah file sudah siap
    # =========================================================
    if xls is None:
        st.stop()


    # =========================================================
    # 🧩 VALIDASI KOLOM WAJIB SETIAP SHEET
    # =========================================================
    required_columns = [
        "Kabupaten/Kota",
        "Harga Telur Ayam Ras (Rp)",
        "Konsumsi Telur Ayam Ras per Kapita",
        "Pengeluaran Telur Ayam Ras (Rp)"
    ]

    def normalize_column(col):
        col = str(col)
        col = col.encode("utf-8", "ignore").decode("utf-8")
        col = re.sub(r"\s+", " ", col)
        col = col.replace("\xa0", " ").replace("\u200b", "")
        return col.strip().lower()

    def validate_sheet_columns(sheet_name, df_sheet):
        cols = [normalize_column(c) for c in df_sheet.columns]
        normalized_required = [normalize_column(c) for c in required_columns]
        missing = [c for c in normalized_required if c not in cols]
        return missing

    sheet_names = xls.sheet_names
    df_list = []
    error_sheets = {}

    for sheet in sheet_names:
        temp = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
        missing = validate_sheet_columns(sheet, temp)
        if missing:
            error_sheets[sheet] = missing
        else:
            try:
                temp["Tahun"] = int(sheet)
            except ValueError:
                temp["Tahun"] = sheet
            df_list.append(temp)

    # =========================================================
    # 🚨 TAMPILKAN ERROR JIKA ADA SHEET TIDAK VALID
    # =========================================================
    if error_sheets:
        st.error("❌ Beberapa sheet memiliki kolom yang tidak lengkap:")
        for sheet, missing_cols in error_sheets.items():
            st.warning(f"📄 Sheet **{sheet}** hilang kolom: {', '.join(missing_cols)}")

        st.info("""
        Pastikan semua sheet memiliki **kolom yang sama** seperti template:
        - Kabupaten/Kota  
        - Harga Telur Ayam Ras (Rp)  
        - Konsumsi Telur Ayam Ras per Kapita  
        - Pengeluaran Telur Ayam Ras (Rp)
        """)

        template_df = pd.DataFrame({
            "Kabupaten/Kota": ["KOTA JAKARTA", "KAB. BANDUNG"],
            "Harga Telur Ayam Ras (Rp)": [28000, 27000],
            "Konsumsi Telur Ayam Ras per Kapita": [2.1, 1.8],
            "Pengeluaran Telur Ayam Ras (Rp)": [42000, 39000]
        })
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            template_df.to_excel(writer, index=False, sheet_name="Contoh")

        st.download_button(
            label="📥 Unduh Template Dataset (Format Benar)",
            data=buffer.getvalue(),
            file_name="Template_Dataset_Telur_Ayam_Ras.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.stop()

    st.markdown("---")

    # =========================================================
    # 📅 FILTER RENTANG TAHUN (AUTO DETECT DARI DATASET)
    # =========================================================
    try:
        sheet_names_sorted = sorted(sheet_names, key=lambda x: int(x))
    except ValueError:
        sheet_names_sorted = sorted(sheet_names)

    col1, col2 = st.columns(2)
    with col1:
        tahun_dari = st.selectbox(
            "📆 Pilih Tahun Awal:",
            sheet_names_sorted,
            index=0,
            key="tahun_dari_tab1"
        )
    with col2:
        tahun_sampai = st.selectbox(
            "📆 Pilih Tahun Akhir:",
            sheet_names_sorted,
            index=len(sheet_names_sorted) - 1,
            key="tahun_sampai_tab1"
        )

    start_idx = sheet_names_sorted.index(tahun_dari)
    end_idx = sheet_names_sorted.index(tahun_sampai)

    if start_idx > end_idx:
        st.error("⚠️ Tahun awal tidak boleh lebih besar dari tahun akhir!")
        st.stop()

    # =========================================================
    # ✅ GABUNG DATA SESUAI RENTANG TAHUN
    # =========================================================
    df_filtered = [
        df for df in df_list if str(df["Tahun"].iloc[0]) in sheet_names_sorted[start_idx:end_idx + 1]
    ]

    if not df_filtered:
        st.error("❌ Tidak ada data yang cocok dengan rentang tahun yang dipilih.")
        st.stop()

    df = pd.concat(df_filtered, ignore_index=True)
    st.success(
        f"✅ Data berhasil dimuat dari tahun {tahun_dari} hingga {tahun_sampai} "
    )

    # =========================================================
    # ⚙️ PILIH VARIABEL DAN METODE
    # =========================================================
    st.markdown("### ⚙️ Konfigurasi Analisis")

    all_features = [
        "Harga Telur Ayam Ras (Rp)",
        "Konsumsi Telur Ayam Ras per Kapita",
        "Pengeluaran Telur Ayam Ras (Rp)"
    ]
    selected_features = st.multiselect(
        "📊 Pilih Variabel yang Akan Digunakan untuk Clustering:",
        options=all_features,
        default=all_features,
        help="Kamu bisa pilih 1–3 variabel.",
        on_change=reset_experiment_state
    )

    if len(selected_features) == 0:
        st.warning("⚠️ Minimal pilih satu variabel untuk melanjutkan.")
        st.stop()

    st.session_state["fitur"] = selected_features
    st.session_state["df_filtered"] = df

    metode = st.selectbox(
        "Pilih Metode Clustering:",
        ["K-Means",
            "Agglomerative Hierarchical Clustering (AHC)", "Intelligent K-Medoids"],
        key="metode_single",
        on_change=reset_experiment_state
    )

    k = None
    if metode in ["K-Means", "Agglomerative Hierarchical Clustering (AHC)"]:
        k = st.slider("Pilih Jumlah Cluster (k):", 2, 7, 3,
                      key="slider_k", on_change=reset_experiment_state)
    else:
        st.info("🤖 Jumlah cluster akan ditentukan otomatis oleh algoritma.")

    # =====================================================
    # 🚀 Jalankan Eksperimen (DENGAN LOADING BAR)
    # =====================================================
    st.caption(
        "Klik tombol di bawah untuk menjalankan metode clustering yang kamu pilih.")

    if st.button("🚀 Jalankan Eksperimen", key="run_single"):
        fitur = st.session_state["fitur"]
        df = st.session_state["df_filtered"]
        metode = st.session_state.get("metode_single", "K-Means")
        k = st.session_state.get("slider_k", 3)

        progress = st.progress(0)
        start_time = time.perf_counter()

        try:
            # --- 1️⃣ Praproses Data ---
            progress.progress(0.2)
            df = clean_data(df, fitur)
            X_scaled = scale_features(df, fitur)

            # --- 2️⃣ Jalankan Metode ---
            progress.progress(0.5)
            if metode == "K-Means":
                labels = run_kmeans(X_scaled, k)
            elif metode == "Agglomerative Hierarchical Clustering (AHC)":
                labels = run_ahc(X_scaled, k)
            else:
                labels, k_auto, sil = run_intelligent_kmedoids_streamlit(X_scaled)
                st.info(
                    f"🤖 Jumlah cluster optimal hasil Intelligent K-Medoids: **2**")

            # --- 3️⃣ Evaluasi & Simpan Hasil ---
            progress.progress(0.8)
            df["Cluster"] = labels
            sil, dbi = evaluate_clusters(X_scaled, labels)

            end_time = time.perf_counter()
            waktu_komputasi = (end_time - start_time)
            waktu_fmt = f"{waktu_komputasi*1000:.2f} ms" if waktu_komputasi < 1 else f"{waktu_komputasi:.2f} detik"

            st.session_state.update({
                "X_scaled": X_scaled,
                "labels": labels,
                "df": df,
                "fitur": fitur,
                "metode": metode,
                "k": k,
                "sil": sil,
                "dbi": dbi,
                "time": waktu_fmt
            })

            # --- 4️⃣ Progress Selesai ---
            progress.progress(1.0)
            progress.empty()

            # --- 5️⃣ Notifikasi Sukses ---
            st.success(f"✅ Metode Berhasil Dijalankan")

        except Exception as e:
            progress.empty()
            st.error(f"❌ Terjadi kesalahan saat menjalankan eksperimen: {e}")

    # =====================================================
    # 📊 VISUALISASI JIKA DATA SUDAH ADA DI SESSION
    # =====================================================
    if "labels" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Hasil Evaluasi")

        X_scaled = st.session_state["X_scaled"]
        labels = st.session_state["labels"]
        df = st.session_state["df"]
        fitur = st.session_state["fitur"]
        metode = st.session_state["metode"]
        k = st.session_state["k"]
        sil = st.session_state["sil"]
        dbi = st.session_state["dbi"]
        waktu_komputasi = st.session_state["time"]

        # === METRICS BAR ===
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{sil:.3f}")
        col2.metric("Davies–Bouldin Index", f"{dbi:.3f}")
        col3.metric("Waktu Komputasi", waktu_komputasi)

        # =====================================================
        # 🔹 4 TAB UTAMA UNTUK VISUALISASI
        # =====================================================
        tab1_viz, tab2_viz, tab3_viz, tab4_viz = st.tabs([
            "📦 Boxplot & Peta",
            "📊 Trends",
            "📈 Silhouette Plot",
            "🎯 Scatter Plot"
        ])
        # =====================================================
        # TAB 1: BOX PLOT + PETA (WARNA SINKRON ANTI-ABU)
        # =====================================================
        from modules.interpretasi import interpretasi_untuk_legend_otomatis
        import seaborn as sns
        import matplotlib.colors as mcolors
        import leafmap.foliumap as leafmap
        import geopandas as gpd
        import folium

        with tab1_viz:
            st.markdown("### 📦 Distribusi Nilai per Cluster dan Tahun")

            with st.spinner("🔄 Membuat Boxplot..."):
                plt.close("all")
                df_box = df.copy()
                n_clusters = df_box["Cluster"].nunique()

                # 🎨 Palet khusus TANPA abu-abu
                custom_palette = [
                    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
                    "#8172B3", "#937860", "#DA8BC3", "#8C8C3E",
                    "#64B5CD", "#FFB300", "#009688", "#AB47BC",
                    "#7E57C2", "#EF5350", "#26A69A", "#FF7043"
                ]

                # ambil warna sesuai jumlah cluster
                cluster_palette_hex = custom_palette[:n_clusters]

                cluster_colors = {
                    c: cluster_palette_hex[i % len(cluster_palette_hex)]
                    for i, c in enumerate(sorted(df["Cluster"].unique()))
                }

                # ======================================================
                # 📊 BOX PLOT
                # ======================================================
                n_vars = len(fitur)
                n_cols = min(n_vars, 3)
                cols = st.columns(n_cols, gap="large")

                if n_vars == 2:
                    cols = [st.empty(), *cols, st.empty()]

                for i, var in enumerate(fitur):
                    target_col = cols[i + 1] if n_vars == 2 else cols[i]
                    with target_col:
                        fig, ax = plt.subplots(figsize=(4.3, 3.5))
                        sns.boxplot(
                            x="Tahun", y=var, hue="Cluster",
                            data=df_box, palette=cluster_palette_hex, ax=ax,
                            fliersize=2, linewidth=0.8
                        )

                        ax.set_xlabel("Tahun")
                        ax.set_ylabel(var)
                        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
                        ax.get_legend().remove()

                        plt.tight_layout(rect=[0, 0.05, 1, 1], pad=1.0)
                        st.pyplot(fig)
                        plt.close(fig)

                # ======================================================
                # 🎨 Legend Warna di Bawah Semua Boxplot (fix tampil)
                # ======================================================
                legend_items = []
                for j, cluster_id in enumerate(sorted(df_box["Cluster"].unique())):
                    color = cluster_palette_hex[j % len(cluster_palette_hex)]
                    legend_items.append(
                        f"<span style='display:inline-flex; align-items:center; gap:5px; margin-right:10px;'>"
                        f"<span style='width:14px; height:14px; background-color:{color}; border-radius:3px; display:inline-block;'></span>"
                        f"<span style='font-size:13px; color:#333;'>Cluster {cluster_id}</span>"
                        f"</span>"
                    )

                legend_html = (
                    "<div style='text-align:center; margin-top:10px; display:flex; justify-content:center; flex-wrap:wrap; gap:10px;'>"
                    + "".join(legend_items)
                    + "</div>"
                )

                st.markdown(legend_html, unsafe_allow_html=True)
                st.markdown("---")

                # =======================================================
                # 🌍 PETA INTERAKTIF (TANPA FILTER, LEGEND DI BAWAH)
                # =======================================================
                with st.spinner("🔄 Memuat Peta Interaktif..."):

                    @st.cache_data
                    def load_geojson():
                        geo_path = "data/Indonesia_cities.geojson"
                        gdf = gpd.read_file(geo_path)
                        gdf["NAME_2"] = gdf["NAME_2"].str.strip().str.upper()
                        return gdf

                    @st.cache_data
                    def merge_data_for_map(df_map, fitur):
                        gdf = load_geojson()
                        # normalisasi nama
                        gdf["NAME_2"] = (
                            gdf["NAME_2"].str.replace(
                                "Kabupaten", "", case=False)
                            .str.replace("KOTA", "", case=False)
                            .str.strip().str.upper()
                        )
                        df_map["Kabupaten/Kota"] = (
                            df_map["Kabupaten/Kota"].str.replace(
                                "KABUPATEN", "", case=False)
                            .str.replace("KOTA", "", case=False)
                            .str.strip().str.upper()
                        )
                        cols_merge = ["Kabupaten/Kota", "Cluster"] + \
                            [c for c in fitur if c in df_map.columns]
                        gdf = gdf.merge(
                            df_map[cols_merge], left_on="NAME_2", right_on="Kabupaten/Kota", how="left")
                        return gdf

                    gdf = merge_data_for_map(df, fitur)

                    st.markdown("### 🗺️ Peta Visualisasi Hasil Clustering")

                    # ✅ Tooltip lebih aman untuk data NaN
                    def make_tooltip(row):
                        # kalau datanya NaN semua, tampil "Tidak Ada Data"
                        if pd.isna(row["Kabupaten/Kota"]) or pd.isna(row["Cluster"]):
                            return "<b>Tidak Ada Data</b>"

                        nama = str(row["Kabupaten/Kota"]).title().strip()
                        if not nama.startswith("Kota"):
                            nama = f"{nama}"

                        cluster_val = row["Cluster"]
                        teks = f"<b>{nama}</b><br><b>Cluster:</b> {int(cluster_val)}<hr style='margin:3px 0;'>"

                        # tambahkan fitur-fitur yang tidak NaN
                        for f in fitur:
                            if f in row and pd.notnull(row[f]):
                                if isinstance(row[f], (int, float)):
                                    teks += f"<b>{f}:</b> {row[f]:,.2f}<br>"
                                else:
                                    teks += f"<b>{f}:</b> {row[f]}<br>"
                        return teks

                    gdf["info"] = gdf.apply(make_tooltip, axis=1)

                    m = leafmap.Map(center=[-2.5, 118], zoom=5)
                    m.add_basemap("CartoDB.Positron")

                    cols_for_map = ["geometry", "Cluster", "info"]
                    cols_for_map = [
                        c for c in cols_for_map if c in gdf.columns]

                    from folium import GeoJson, GeoJsonTooltip

                    geo_layer = GeoJson(
                        gdf[cols_for_map],
                        name="Peta Cluster",
                        style_function=lambda x: {
                            "fillColor": (
                                cluster_colors.get(
                                    x["properties"].get("Cluster"))
                                if pd.notnull(x["properties"].get("Cluster"))
                                else "#C8C8C8"  # abu-abu untuk NaN
                            ),
                            "color": "#4d4d4d",
                            "weight": 0.4,
                            "opacity": 0.6,
                            "fillOpacity": 0.9,
                        },
                        tooltip=GeoJsonTooltip(
                            fields=["info"],
                            aliases=[""],
                            labels=False,
                            sticky=True,
                            style=(
                                "background-color: rgba(255,255,255,0.95); "
                                "color: #222; font-size: 12px; "
                                "font-family: Arial; border-radius: 4px; "
                                "padding: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
                            ),
                        ),
                    )

                    m.add_layer(geo_layer)
                    m.to_streamlit(height=550)

                # =======================================================
                # 🧩 LEGEND INTERPRETASI DI BAWAH PETA
                # =======================================================
                auto_labels = interpretasi_untuk_legend_otomatis(df, fitur)
                st.markdown("### Interpretasi Cluster")

                # 💡 Container rapi dan sejajar
                with st.container():
                    for cluster in sorted(auto_labels.keys()):
                        color = cluster_colors.get(
                            cluster, list(cluster_colors.values())[0])
                        label = auto_labels[cluster]

                        st.markdown(
                            f"""
                            <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                                <div style='width:16px; height:16px; background-color:{color};
                                            border-radius:3px; flex-shrink:0;'></div>
                                <div style='font-size:14px; color:#333; line-height:1.4;'>
                                    <b>Cluster {cluster} :</b> {label}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.markdown("---")

                # =======================================================
                # 📋 TABEL & 📊 GRAFIK JUMLAH KABUPATEN/KOTA PER CLUSTER
                # =======================================================
                with st.spinner("📋 Menyiapkan Daftar Kabupaten/Kota..."):

                    df_cluster_list = (
                        df[["Kabupaten/Kota", "Cluster"]]
                        .drop_duplicates()
                        .sort_values(by="Cluster")
                        .reset_index(drop=True)
                    )

                    def format_nama(nama):
                        nama = nama.strip().upper()
                        return nama.title()

                    df_cluster_list["Kabupaten/Kota"] = df_cluster_list["Kabupaten/Kota"].apply(
                        format_nama)

                # =======================================================
                # 📋 TABEL & 📊 GRAFIK JUMLAH KABUPATEN/KOTA PER CLUSTER (DOMINAN, FIX)
                # =======================================================

                # 1️⃣ Hitung cluster dominan (mode) tiap kabupaten/kota
                def ambil_cluster_dominan(series_cluster):
                    mode_val = series_cluster.mode()
                    if not mode_val.empty:
                        return mode_val.iloc[0]
                    else:
                        return series_cluster.iloc[-1]  # fallback ke cluster terakhir

                df_cluster_list = (
                    df_box.groupby("Kabupaten/Kota")["Cluster"]
                    .apply(ambil_cluster_dominan)
                    .reset_index(name="Cluster")  # hasilnya jadi DataFrame dengan kolom Cluster
                    .sort_values(by="Cluster")
                    .reset_index(drop=True)
                )

                # Format nama kabupaten/kota agar rapi
                def format_nama(nama):
                    nama = nama.strip().upper()
                    return nama.title()

                df_cluster_list["Kabupaten/Kota"] = df_cluster_list["Kabupaten/Kota"].apply(format_nama)

                # =======================================================
                # 📊 TABEL & GRAFIK BERDAMPINGAN
                # =======================================================
                col1, col2 = st.columns([1.0, 1.3])

                with col1:
                    st.markdown("### 📋 Daftar Kabupaten/Kota Berdasarkan Cluster")
                    st.dataframe(
                        df_cluster_list,
                        use_container_width=False,
                        width=380,
                        hide_index=True
                    )

                with col2:
                    # =======================================================
                    # 📊 JUMLAH KABUPATEN/KOTA PER CLUSTER
                    # =======================================================
                    cluster_counts = (
                        df_cluster_list.groupby("Cluster")["Kabupaten/Kota"]
                        .count()
                        .reset_index()
                        .rename(columns={"Kabupaten/Kota": "Jumlah_KabKota"})
                        .sort_values(by="Cluster", ascending=True)
                    )

                    # Warna batang disamakan dengan warna cluster di boxplot/peta
                    cluster_colors_list = [
                        cluster_colors.get(c, "#999999") for c in cluster_counts["Cluster"]
                    ]

                    st.markdown("### 📊 Jumlah Kabupaten/Kota pada Tiap Cluster")

                    fig, ax = plt.subplots(figsize=(4.8, 3.3))
                    sns.barplot(
                        x="Cluster",
                        y="Jumlah_KabKota",
                        data=cluster_counts,
                        palette=cluster_colors_list,
                        ax=ax
                    )

                    # Tambahkan judul langsung di atas plot
                    ax.set_title("Distribusi Jumlah Kabupaten/Kota per Cluster",
                                fontsize=12, fontweight="bold", pad=10)

                    # Tambahkan label di atas batang
                    for container in ax.containers:
                        ax.bar_label(
                            container, fmt="%d", label_type="edge", fontsize=9, padding=2, color="#222"
                        )

                    ymax = cluster_counts["Jumlah_KabKota"].max()
                    ax.set_ylim(0, ymax * 1.15)
                    ax.set_xlabel("Cluster", fontsize=11)
                    ax.set_ylabel("Jumlah Kab/Kota", fontsize=11)
                    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # =======================================================
                    # 🎨 LEGENDAR WARNA (SAMA DENGAN BOX PLOT/PETA)
                    # =======================================================
                    legend_html = "".join([
                        f"<span style='display:inline-flex; align-items:center; gap:5px; margin-right:10px;'>"
                        f"<span style='width:14px; height:14px; background-color:{cluster_colors.get(c)}; "
                        f"border-radius:3px; display:inline-block;'></span>"
                        f"<span style='font-size:13px; color:#333;'>Cluster {c}</span>"
                        f"</span>"
                        for c in sorted(cluster_colors.keys())
                    ])

                    st.markdown(
                        f"<div style='text-align:center; margin-top:6px; display:flex; justify-content:center; flex-wrap:wrap; gap:10px;'>{legend_html}</div>",
                        unsafe_allow_html=True
                    )

        # =====================================================
        # TAB 2: TRENDS (ADAPTIF)
        # =====================================================
        with tab2_viz:
            tahun_unik = df["Tahun"].unique()
            n_tahun = len(tahun_unik)

            if n_tahun > 1:
                # =================================================
                # MODE: GRAFIK TREN (multi-year)
                # =================================================
                st.markdown("### 📊 Tren Tahunan Setiap Variabel")
                st.markdown(
                    "Analisis tren maksimum, minimum, dan rata-rata untuk tiap variabel dari tahun yang tersedia.")

                with st.spinner("🔄 Menghitung tren tahunan..."):
                    # Pastikan kolom Tahun numerik
                    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")

                    # Pilih kolom yang dipakai
                    df_trend = df[["Tahun"] + fitur].copy()

                    # Hitung statistik tahunan (max, min, mean)
                    stats = df_trend.groupby("Tahun").agg(
                        ["max", "min", "mean"]).round(2)
                    stats.columns = ["_".join(col) for col in stats.columns]
                    stats = stats.reset_index()

                    # Ubah kolom Tahun jadi string supaya rapi di sumbu X
                    stats["Tahun"] = stats["Tahun"].astype(str)

                    # Buat subplot dinamis sesuai jumlah variabel
                    n_vars = len(fitur)
                    fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 3))
                    if n_vars == 1:
                        axes = [axes]

                    for i, var in enumerate(fitur):
                        ax = axes[i]
                        tahun_str = stats["Tahun"]

                        ax.plot(
                            tahun_str, stats[f"{var}_max"], marker="^", color="blue", label="Maksimum")
                        ax.plot(
                            tahun_str, stats[f"{var}_min"], marker="v", color="red", label="Minimum")
                        ax.plot(
                            tahun_str, stats[f"{var}_mean"], marker="o", color="green", label="Rata-rata")

                        ax.set_title(f"({chr(97+i)}) Tren {var}",
                                     fontsize=12, fontweight="bold")
                        ax.set_xlabel("Tahun")
                        ax.set_ylabel(var)
                        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                        ax.set_xticks(range(len(tahun_str)))
                        ax.set_xticklabels(tahun_str)

                    # ✅ Legend di luar plot (bawah semua subplot)
                    fig.legend(
                        labels=["Maksimum", "Minimum", "Rata-rata"],
                        loc="lower center",
                        ncol=3,
                        frameon=False,
                        fontsize=9
                    )

                    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                    st.pyplot(fig)

                st.success("✅ Grafik tren tahunan selesai dibuat!")

            else:
                # =================================================
                # MODE: PROFIL SATU TAHUN (single-year)
                # =================================================
                tahun_terpilih = int(tahun_unik[0])
                st.markdown(f"### 📊 Profil Variabel Tahun {tahun_terpilih}")
                st.markdown(
                    "Menampilkan nilai minimum, rata-rata, dan maksimum untuk setiap variabel di tahun ini.")

                # Hitung statistik ringkas
                stats_single = df[fitur].describe(
                ).T[["min", "mean", "max"]].round(2)
                stats_single.columns = ["Minimum", "Rata-rata", "Maksimum"]

                st.dataframe(stats_single.style.format("{:.2f}"))
                st.success(
                    f"✅ Ringkasan variabel untuk tahun {tahun_terpilih} berhasil ditampilkan.")

            # =====================================================
            # 🏆 TAMBAHAN: DASHBOARD GAYA "TOP STATES" UNTUK TOP 10
            # =====================================================
            import matplotlib.cm as cm

            st.markdown("---")
            st.markdown(
                "### 🏆 Top 10 Kabupaten/Kota Tertinggi dan Terendah per Variabel (Visual Dashboard)")

            if "Kabupaten/Kota" in df.columns:
                for var in fitur:
                    st.markdown(f"#### 📊 {var}")
                    with st.spinner(f"Menghitung peringkat untuk {var}..."):
                        df_sorted = df[["Kabupaten/Kota", var,
                                        "Tahun"]].dropna(subset=[var])
                        df_mean = (
                            df_sorted.groupby("Kabupaten/Kota")[var]
                            .mean()
                            .reset_index()
                            .sort_values(by=var, ascending=False)
                        )

                        # ✅ Tambahkan "Kabupaten" di depan kalau belum ada kata "Kota"
                        def format_nama(nama):
                            nama = nama.strip().title()
                            if not nama.lower().startswith("kota"):
                                return f"Kabupaten {nama}"
                            return nama

                        df_mean["Kabupaten/Kota"] = df_mean["Kabupaten/Kota"].apply(
                            format_nama)

                        top10_high = df_mean.head(10).reset_index(drop=True)
                        top10_low = df_mean.tail(10).sort_values(
                            by=var, ascending=True).reset_index(drop=True)

                        col1, col2 = st.columns(2)

                        # === TERTINGGI ===
                        with col1:
                            st.markdown("**🔺 Top 10 Tertinggi**")
                            max_val = top10_high[var].max()
                            for _, row in top10_high.iterrows():
                                bar_val = row[var] / max_val
                                st.markdown(
                                    f"""
                                    <div style='margin-bottom:20px;'>  <!-- ✅ jarak antar bar -->
                                        <div style='display:flex; justify-content:space-between;'>
                                            <span style='font-weight:600;'>{row["Kabupaten/Kota"]}</span>
                                            <span style='font-weight:500;'>{row[var]:,.2f}</span>
                                        </div>
                                        <div style='background-color:#333; border-radius:5px; height:10px; margin-top:3px;'>
                                            <div style='background:linear-gradient(90deg, #ff4b5c, #ff8fa3); width:{bar_val*100:.1f}%; height:10px; border-radius:5px;'></div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        # === TERENDAH ===
                        with col2:
                            st.markdown("**🔻 Top 10 Terendah**")
                            max_val = top10_low[var].max()
                            for _, row in top10_low.iterrows():
                                bar_val = row[var] / max_val
                                st.markdown(
                                    f"""
                                    <div style='margin-bottom:20px;'>  <!-- ✅ jarak antar bar -->
                                        <div style='display:flex; justify-content:space-between;'>
                                            <span style='font-weight:600;'>{row["Kabupaten/Kota"]}</span>
                                            <span style='font-weight:500;'>{row[var]:,.2f}</span>
                                        </div>
                                        <div style='background-color:#333; border-radius:5px; height:10px; margin-top:3px;'>
                                            <div style='background:linear-gradient(90deg, #74b9ff, #a0c4ff); width:{bar_val*100:.1f}%; height:10px; border-radius:5px;'></div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                st.success(
                    "✅ Visualisasi gaya 'Top States' berhasil ditampilkan!")
            else:
                st.warning(
                    "⚠️ Kolom 'Kabupaten/Kota' tidak ditemukan dalam dataset.")

        # =====================================================
        # TAB 3: SILHOUETTE PLOT
        # =====================================================
        with tab3_viz:
            st.markdown("### 📈 Silhouette Plot")
            with st.spinner("🔄 Membuat Silhouette Plot..."):
                fig_sil, ax1 = plt.subplots(figsize=(7, 5))
                silhouette_vals = silhouette_samples(X_scaled, labels)
                y_lower, y_upper = 0, 0
                yticks = []

                for i in range(k):
                    c_sil = silhouette_vals[labels == i]
                    c_sil.sort()
                    y_upper += len(c_sil)
                    ax1.barh(range(y_lower, y_upper), c_sil, height=1.0)
                    yticks.append((y_lower + y_upper) / 2)
                    y_lower += len(c_sil)

                ax1.axvline(sil, color="red", linestyle="--")
                ax1.set_xlabel("Silhouette Coefficient")
                ax1.set_ylabel("Cluster")
                ax1.set_yticks(yticks)
                ax1.set_yticklabels(range(k))
                ax1.set_xlim([-0.1, 1])
                ax1.set_title(f"Silhouette Plot ({metode})", fontsize=13)

                st.pyplot(fig_sil)
            st.success("✅ Silhouette Plot selesai dibuat!")

        # =====================================================
        # TAB 4: SCATTER / PAIRGRID
        # =====================================================
        with tab4_viz:
            st.markdown("### 🔍 Clustering Visualization Berdasarkan Variabel")

            with st.spinner("🔄 Menyiapkan Scatter / PairGrid..."):
                df_plot = df.copy()
                df_plot["Tahun"] = df_plot["Tahun"].astype(int)

                n_tahun = df_plot["Tahun"].nunique()
                n_fitur = len(fitur)

                # =====================================================
                # CASE 1️⃣: Multi-year → Scatter antar tahun
                # =====================================================
                if n_tahun > 1:
                    st.markdown(
                        "Menampilkan hubungan antar tahun untuk setiap variabel yang dipilih.")

                    # Ambil variabel sesuai yang dipilih user
                    variabels = [(f"({chr(97+i)}) {var}", var)
                                 for i, var in enumerate(fitur)]

                    # Atur layout kolom dinamis
                    n = len(variabels)
                    n_cols = 3 if n >= 3 else n  # max 3 kolom
                    cols = st.columns(n_cols)

                    for i, (judul, var) in enumerate(variabels):
                        with cols[i % n_cols]:
                            df_pivot = df_plot.pivot_table(
                                index="Kabupaten/Kota",
                                columns="Tahun",
                                values=var,
                                aggfunc="mean"
                            )
                            tahun_unik = sorted(df_plot["Tahun"].unique())
                            df_pivot = df_pivot.reindex(columns=tahun_unik)
                            df_pivot["Cluster"] = (
                                df_plot.groupby("Kabupaten/Kota")["Cluster"]
                                .first()
                                .reindex(df_pivot.index)
                            )

                            g = sns.PairGrid(df_pivot, hue="Cluster",
                                             palette="husl", height=2.2)
                            g.map_lower(sns.scatterplot, s=35,
                                        edgecolor="k", linewidth=0.3)
                            g.map_upper(sns.scatterplot, s=35,
                                        edgecolor="k", linewidth=0.3, alpha=0.5)
                            g.map_diag(sns.kdeplot, fill=True, alpha=0.6)

                            g.fig.suptitle(
                                judul, y=1.02, fontsize=12, fontweight="bold")
                            plt.tight_layout()

                            st.pyplot(g.fig)
                            # ✅ PairGrid harus ditutup pakai .fig
                            plt.close(g.fig)

                    st.success("✅ Scatter selesai dibuat!")

                # =====================================================
                # CASE 2️⃣: 1 tahun & 1 variabel → Tidak relevan
                # =====================================================
                else:
                    st.info("""
                    ⚠️ **Scatter plot tidak ditampilkan.**  
                    Dataset hanya memiliki **satu variabel** dan **satu tahun data**,  
                    sehingga tidak ada hubungan antar variabel atau tren waktu yang dapat divisualisasikan.  
                    Silakan lihat *boxplot, peta, atau silhouette plot* untuk analisis distribusi antar cluster.
                    """)

# =========================================================
# TAB 2: PERBANDINGAN DUA METODE
# =========================================================
with tab2:
    st.header("Perbandingan Dua Metode")

    # =========================================================
    # 🧹 AUTO RESET jika user ganti pilihan
    # =========================================================
    def reset_dual_state():
        for key in ["dual_results", "fitur", "df_filtered_tab2"]:
            if key in st.session_state:
                del st.session_state[key]

    # =========================================================
    # 1️⃣ PILIH SUMBER DATASET (TAB 2) + VALIDASI PER SHEET
    # =========================================================
    import io
    import re
    import pandas as pd

    # Daftar kolom wajib sesuai template
    required_columns = [
        "Kabupaten/Kota",
        "Harga Telur Ayam Ras (Rp)",
        "Konsumsi Telur Ayam Ras per Kapita",
        "Pengeluaran Telur Ayam Ras (Rp)"
    ]

    # --- fungsi bantu untuk normalisasi nama kolom ---
    def normalize_column(col):
        col = str(col)
        col = col.encode('utf-8', 'ignore').decode('utf-8')
        col = re.sub(r'\s+', ' ', col)
        col = col.replace('\xa0', ' ').replace('\u200b', '')
        return col.strip().lower()

    # --- fungsi bantu untuk validasi tiap sheet ---
    def validate_sheet_columns(sheet_name, df_sheet):
        cols = [normalize_column(c) for c in df_sheet.columns]
        normalized_required = [normalize_column(c) for c in required_columns]
        missing = [c for c in normalized_required if c not in cols]
        return missing

    # =========================================================
    # 📂 PILIH SUMBER DATASET
    # =========================================================
    pilihan_data = st.radio(
        "Pilih sumber dataset:",
        ["Gunakan dataset bawaan", "Upload dataset sendiri"],
        horizontal=True,
        key="pilihan_data_tab2",
        on_change=reset_dual_state
    )
    st.markdown("---")

    # =========================================================
    # 🔍 BACA DAN VALIDASI SETIAP SHEET
    # =========================================================
    if pilihan_data == "Gunakan dataset bawaan":
        excel_path = "data/Dataset Ready.xlsx"
        xls = pd.ExcelFile(excel_path)
    else:
        uploaded_file = st.file_uploader(
            "📂 Unggah file Excel (.xlsx)",
            type=["xlsx"],
            key="upload_tab2_validasi",  # ✅ key unik
            on_change=reset_dual_state
        )
        if not uploaded_file:
            st.warning(
                "⚠️ Silakan upload dataset terlebih dahulu untuk melanjutkan.")
            st.stop()
        try:
            xls = pd.ExcelFile(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file Excel: {e}")
            st.stop()

    sheet_names = xls.sheet_names
    df_list = []
    error_sheets = {}

    # Loop tiap sheet & validasi kolom
    for sheet in sheet_names:
        temp = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
        missing = validate_sheet_columns(sheet, temp)

        if missing:
            missing_display = [
                required_columns[[normalize_column(
                    c) for c in required_columns].index(m)]
                for m in missing
            ]
            error_sheets[sheet] = missing_display
        else:
            try:
                temp["Tahun"] = int(sheet)
            except ValueError:
                temp["Tahun"] = sheet
            df_list.append(temp)

    # =========================================================
    # 🚨 JIKA ADA SHEET ERROR
    # =========================================================
    if error_sheets:
        st.error(
            "❌ Dataset tidak valid. Beberapa sheet memiliki kolom yang tidak lengkap:")
        for sheet, missing_cols in error_sheets.items():
            st.warning(
                f"📄 Sheet **{sheet}** hilang kolom: {', '.join(missing_cols)}")

        st.info("""
        Pastikan semua sheet memiliki **kolom yang sama** seperti template:
        - Kabupaten/Kota  
        - Harga Telur Ayam Ras (Rp)  
        - Konsumsi Telur Ayam Ras per Kapita  
        - Pengeluaran Telur Ayam Ras (Rp)
        """)

        # tombol download template
        template_df = pd.DataFrame({
            "Kabupaten/Kota": ["KOTA JAKARTA", "KAB. BANDUNG"],
            "Harga Telur Ayam Ras (Rp)": [28000, 27000],
            "Konsumsi Telur Ayam Ras per Kapita": [2.1, 1.8],
            "Pengeluaran Telur Ayam Ras (Rp)": [42000, 39000]
        })
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            template_df.to_excel(writer, index=False, sheet_name="Contoh")

        st.download_button(
            label="📥 Unduh Template Dataset (Format Benar)",
            data=buffer.getvalue(),
            file_name="Template_Dataset_Telur_Ayam_Ras.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.stop()

    # =========================================================
    # ✅ SEMUA SHEET VALID → PILIH RANGE TAHUN
    # =========================================================
    col1, col2 = st.columns(2)
    with col1:
        tahun_dari = st.selectbox(
            "📅 Pilih Tahun Awal:",
            sheet_names,
            index=0,
            key="tahun_dari_tab2",
            on_change=reset_dual_state
        )
    with col2:
        tahun_sampai = st.selectbox(
            "📅 Pilih Tahun Akhir:",
            sheet_names,
            index=len(sheet_names) - 1,
            key="tahun_sampai_tab2",
            on_change=reset_dual_state
        )

    # =========================================================
    # 🚨 ERROR HANDLING: Tahun awal tidak boleh lebih besar
    # =========================================================
    start_idx = sheet_names.index(tahun_dari)
    end_idx = sheet_names.index(tahun_sampai)

    if start_idx > end_idx:
        st.error(
            f"⚠️ Tahun awal ({tahun_dari}) tidak boleh lebih besar dari tahun akhir ({tahun_sampai}). "
            "Silakan periksa kembali urutan tahun yang dipilih."
        )
        st.stop()  # hentikan eksekusi kode berikutnya

    # =========================================================
    # ✅ Jika valid → proses data
    # =========================================================
    selected_sheets = sheet_names[start_idx:end_idx + 1]

    # Filter sheet sesuai range tahun
    df_filtered = [df for df in df_list if str(df['Tahun'].iloc[0]) in selected_sheets]
    df = pd.concat(df_filtered, ignore_index=True)

    st.success(
        f"✅ Semua sheet valid. Data dari tahun {tahun_dari}–{tahun_sampai} berhasil dimuat "
    )

    # =========================================================
    # 2️⃣ PILIH VARIABEL UNTUK ANALISIS
    # =========================================================
    st.markdown("### ⚙️ Konfigurasi Analisis")
    all_features = [
        "Harga Telur Ayam Ras (Rp)",
        "Konsumsi Telur Ayam Ras per Kapita",
        "Pengeluaran Telur Ayam Ras (Rp)"
    ]
    selected_features = st.multiselect(
        "📊 Pilih Variabel yang Akan Digunakan untuk Clustering:",
        options=all_features,
        default=all_features,
        help="Kamu bisa pilih 1–3 variabel.",
        key="fitur_tab2",
        on_change=reset_dual_state
    )

    if len(selected_features) == 0:
        st.warning("⚠️ Minimal pilih satu variabel untuk melanjutkan.")
        st.stop()

    st.session_state["fitur"] = selected_features
    st.session_state["df_filtered_tab2"] = df

    # =========================================================
    # 3️⃣ PILIH DUA METODE UNTUK DIBANDINGKAN
    # =========================================================
    st.markdown("### ⚙️ Pilih Dua Metode Clustering")
    metode_opsi = [
        "K-Means", "Agglomerative Hierarchical Clustering (AHC)", "Intelligent K-Medoids"]

    col1, col2 = st.columns(2)
    with col1:
        metode1 = st.selectbox("🔹 Metode Pertama:", metode_opsi,
                               key="metode_1_tab2", on_change=reset_dual_state)
    with col2:
        metode2 = st.selectbox("🔸 Metode Kedua:", metode_opsi,
                               index=1, key="metode_2_tab2", on_change=reset_dual_state)

    if metode1 == metode2:
        st.warning("⚠️ Pilih dua metode yang berbeda untuk membandingkan.")
        st.stop()

    # --- Jumlah cluster
    col1, col2 = st.columns(2)
    with col1:
        if metode1 in ["K-Means", "Agglomerative Hierarchical Clustering (AHC)"]:
            k1 = st.slider(
                f"Jumlah Cluster ({metode1})", 2, 7, 3, key="k1_tab2", on_change=reset_dual_state)
        else:
            k1 = None
            st.info(f"🤖 {metode1} menentukan jumlah cluster otomatis.")
    with col2:
        if metode2 in ["K-Means", "Agglomerative Hierarchical Clustering (AHC)"]:
            k2 = st.slider(
                f"Jumlah Cluster ({metode2})", 2, 7, 3, key="k2_tab2", on_change=reset_dual_state)
        else:
            k2 = None
            st.info(f"🤖 {metode2} menentukan jumlah cluster otomatis.")

    # =========================================================
    # 4️⃣ EKSEKUSI DUA METODE
    # =========================================================
    if st.button("🚀 Jalankan Perbandingan", key="run_dual_final"):
        fitur = st.session_state["fitur"]
        df_base = st.session_state["df_filtered_tab2"]

        hasil = {}
        data = {}  # ✅ tambahkan biar gak NameError waktu simpan k_auto
        progress = st.progress(0)

        for i, (metode, k) in enumerate([(metode1, k1), (metode2, k2)], start=1):
            df_temp = df_base.copy()
            start_time = time.perf_counter()

            df_temp = clean_data(df_temp, fitur)
            X_scaled = scale_features(df_temp, fitur)

            if metode == "K-Means":
                labels = run_kmeans(X_scaled, k)
            elif metode == "Agglomerative Hierarchical Clustering (AHC)":
                labels = run_ahc(X_scaled, k)
            else:
                labels, k_auto, sil = run_intelligent_kmedoids_streamlit(X_scaled)
                data["k"] = k_auto  # ✅ simpan hasil jumlah cluster otomatis
                st.info(f"🤖 Jumlah cluster optimal untuk {metode}: **{k_auto}**")

            df_temp["Cluster"] = labels
            sil, dbi = evaluate_clusters(X_scaled, labels)
            end_time = time.perf_counter()

            df_temp["Cluster"] = labels
            sil, dbi = evaluate_clusters(X_scaled, labels)

            end_time = time.perf_counter()
            waktu_komputasi = end_time - start_time
            waktu_fmt = f"{waktu_komputasi*1000:.2f} ms" if waktu_komputasi < 1 else f"{waktu_komputasi:.2f} detik"

            hasil[metode] = {
                "df": df_temp,
                "labels": labels,
                "k": k,
                "sil": sil,
                "dbi": dbi,
                "time": waktu_fmt,
                "X_scaled": X_scaled
            }
            progress.progress(i / 2)
        progress.empty()

        st.session_state["dual_results"] = hasil
        st.success("✅ Kedua metode berhasil dijalankan!")

        # =========================================================
        # 6️⃣ VISUALISASI HASIL
        # =========================================================
        if "dual_results" in st.session_state:
            hasil = st.session_state["dual_results"]
            fitur = st.session_state["fitur"]

            st.markdown("---")
            st.subheader("📊 Hasil Evaluasi Perbandingan")

            # === Metric Bar per metode ===
            for metode, data in hasil.items():
                st.markdown(f"#### 🧩 {metode}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Silhouette Score", f"{data['sil']:.3f}")
                col2.metric("Davies–Bouldin Index", f"{data['dbi']:.3f}")
                col3.metric("Waktu Komputasi", data["time"])

            # === Tabs Visualisasi ===
            tabA, tabB, tabC, tabD = st.tabs([
                "📦 Boxplot & Peta",
                "📊 Tren",
                "📈 Silhouette Plot",
                "🎯 Scatter Plot"
            ])

            # =====================================================
            # 📦 TAB A: BOX PLOT & PETA
            # =====================================================
            with tabA:
                st.markdown("### 📦 Distribusi Nilai dan Peta per Metode")

                from modules.interpretasi import interpretasi_untuk_legend_otomatis
                import seaborn as sns
                import matplotlib.colors as mcolors
                import leafmap.foliumap as leafmap
                import geopandas as gpd
                from folium import GeoJson, GeoJsonTooltip
                import matplotlib.pyplot as plt

                for metode, data in hasil.items():
                    st.markdown(f"#### 🧩 {metode}")

                    with st.spinner(f"🔄 Membuat Boxplot & Peta untuk {metode}..."):
                        plt.close("all")
                        df_box = data["df"].copy()
                        n_clusters = df_box["Cluster"].nunique()

                        # 🎨 Palet khusus (tanpa abu-abu)
                        custom_palette = [
                            "#4C72B0", "#DD8452", "#55A868", "#C44E52",
                            "#8172B3", "#937860", "#DA8BC3", "#8C8C3E",
                            "#64B5CD", "#FFB300", "#009688", "#AB47BC",
                            "#7E57C2", "#EF5350", "#26A69A", "#FF7043"
                        ]
                        cluster_palette_hex = custom_palette[:n_clusters]
                        cluster_colors = {
                            c: cluster_palette_hex[i %
                                                   len(cluster_palette_hex)]
                            for i, c in enumerate(sorted(df_box["Cluster"].unique()))
                        }

                        # ======================================================
                        # 📊 BOX PLOT
                        # ======================================================
                        st.markdown(
                            "##### 📦 Distribusi Nilai per Cluster dan Tahun")
                        n_vars = len(fitur)
                        n_cols = min(n_vars, 3)
                        cols = st.columns(n_cols, gap="large")

                        if n_vars == 2:
                            cols = [st.empty(), *cols, st.empty()]

                        for i, var in enumerate(fitur):
                            target_col = cols[i +
                                              1] if n_vars == 2 else cols[i]
                            with target_col:
                                fig, ax = plt.subplots(figsize=(4.3, 3.5))
                                sns.boxplot(
                                    x="Tahun", y=var, hue="Cluster",
                                    data=df_box, palette=cluster_palette_hex, ax=ax,
                                    fliersize=2, linewidth=0.8
                                )
                                ax.set_xlabel("Tahun")
                                ax.set_ylabel(var)
                                ax.grid(True, axis='y',
                                        linestyle='--', alpha=0.3)
                                ax.get_legend().remove()
                                plt.tight_layout(rect=[0, 0.05, 1, 1], pad=1.0)
                                st.pyplot(fig)
                                plt.close(fig)

                        # 🎨 Legend warna cluster di bawah boxplot
                        legend_items = []
                        for j, cluster_id in enumerate(sorted(df_box["Cluster"].unique())):
                            color = cluster_palette_hex[j % len(
                                cluster_palette_hex)]
                            legend_items.append(
                                f"<span style='display:inline-flex; align-items:center; gap:5px; margin-right:10px;'>"
                                f"<span style='width:14px; height:14px; background-color:{color}; border-radius:3px; display:inline-block;'></span>"
                                f"<span style='font-size:13px; color:#333;'>Cluster {cluster_id}</span>"
                                f"</span>"
                            )
                        legend_html = (
                            "<div style='text-align:center; margin-top:10px; display:flex; justify-content:center; flex-wrap:wrap; gap:10px;'>"
                            + "".join(legend_items)
                            + "</div>"
                        )
                        st.markdown(legend_html, unsafe_allow_html=True)
                        st.markdown("---")

                        # =======================================================
                        # 🌍 PETA INTERAKTIF
                        # =======================================================
                        st.markdown(
                            "##### 🗺️ Peta Visualisasi Hasil Clustering")

                        @st.cache_data
                        def load_geojson():
                            geo_path = "data/Indonesia_cities.geojson"
                            gdf = gpd.read_file(geo_path)
                            gdf["NAME_2"] = gdf["NAME_2"].str.strip().str.upper()
                            return gdf

                        @st.cache_data
                        def merge_data_for_map(df_map, fitur):
                            gdf = load_geojson()
                            gdf["NAME_2"] = (
                                gdf["NAME_2"].str.replace(
                                    "Kabupaten", "", case=False)
                                .str.replace("KOTA", "", case=False)
                                .str.strip().str.upper()
                            )
                            df_map["Kabupaten/Kota"] = (
                                df_map["Kabupaten/Kota"].str.replace(
                                    "KABUPATEN", "", case=False)
                                .str.replace("KOTA", "", case=False)
                                .str.strip().str.upper()
                            )
                            cols_merge = ["Kabupaten/Kota", "Cluster"] + \
                                [c for c in fitur if c in df_map.columns]
                            gdf = gdf.merge(
                                df_map[cols_merge], left_on="NAME_2", right_on="Kabupaten/Kota", how="left")
                            return gdf

                        gdf = merge_data_for_map(df_box, fitur)

                        # Tooltip
                        def make_tooltip(row):
                            if pd.isna(row["Kabupaten/Kota"]) or pd.isna(row["Cluster"]):
                                return "<b>Tidak Ada Data</b>"
                            nama = str(row["Kabupaten/Kota"]).title().strip()
                            if not nama.startswith("Kota"):
                                nama = f"{nama}"
                            cluster_val = row["Cluster"]
                            teks = f"<b>{nama}</b><br><b>Cluster:</b> {int(cluster_val)}<hr style='margin:3px 0;'>"
                            for f in fitur:
                                if f in row and pd.notnull(row[f]):
                                    if isinstance(row[f], (int, float)):
                                        teks += f"<b>{f}:</b> {row[f]:,.2f}<br>"
                                    else:
                                        teks += f"<b>{f}:</b> {row[f]}<br>"
                            return teks

                        gdf["info"] = gdf.apply(make_tooltip, axis=1)

                        m = leafmap.Map(center=[-2.5, 118], zoom=5)
                        m.add_basemap("CartoDB.Positron")
                        cols_for_map = ["geometry", "Cluster", "info"]
                        cols_for_map = [
                            c for c in cols_for_map if c in gdf.columns]

                        geo_layer = GeoJson(
                            gdf[cols_for_map],
                            name="Peta Cluster",
                            style_function=lambda x: {
                                "fillColor": (
                                    cluster_colors.get(
                                        x["properties"].get("Cluster"))
                                    if pd.notnull(x["properties"].get("Cluster"))
                                    else "#C8C8C8"
                                ),
                                "color": "#4d4d4d",
                                "weight": 0.4,
                                "opacity": 0.6,
                                "fillOpacity": 0.9,
                            },
                            tooltip=GeoJsonTooltip(
                                fields=["info"],
                                aliases=[""],
                                labels=False,
                                sticky=True,
                                style=(
                                    "background-color: rgba(255,255,255,0.95); "
                                    "color: #222; font-size: 12px; "
                                    "font-family: Arial; border-radius: 4px; "
                                    "padding: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
                                ),
                            ),
                        )
                        m.add_layer(geo_layer)
                        m.to_streamlit(height=550)

                        # =======================================================
                        # 🧩 LEGEND INTERPRETASI & TABEL KABUPATEN/KOTA
                        # =======================================================
                        auto_labels = interpretasi_untuk_legend_otomatis(
                            df_box, fitur)
                        st.markdown("### Interpretasi Cluster")

                        with st.container():
                            for cluster in sorted(auto_labels.keys()):
                                color = cluster_colors.get(
                                    cluster, list(cluster_colors.values())[0])
                                label = auto_labels[cluster]
                                st.markdown(
                                    f"""
                                    <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                                        <div style='width:16px; height:16px; background-color:{color};
                                                    border-radius:3px; flex-shrink:0;'></div>
                                        <div style='font-size:14px; color:#333; line-height:1.4;'>
                                            <b>Cluster {cluster} :</b> {label}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        # =======================================================
                        # 📋 TABEL & 📊 GRAFIK JUMLAH KABUPATEN/KOTA PER CLUSTER 
                        # =======================================================
                        with st.spinner("📋 Menyiapkan Daftar Kabupaten/Kota..."):

                            # Pastikan kolom Cluster tersedia
                            if "Cluster" not in df_box.columns:
                                st.warning("⚠️ Kolom **Cluster** belum tersedia. Jalankan proses clustering terlebih dahulu.")
                            else:
                                # 1️⃣ Hitung cluster dominan (mode) tiap kabupaten/kota di seluruh tahun
                                def ambil_cluster_dominan(series_cluster):
                                    mode_val = series_cluster.mode()
                                    if not mode_val.empty:
                                        return mode_val.iloc[0]
                                    else:
                                        return series_cluster.iloc[-1]  # fallback ke cluster terakhir

                                df_cluster_list = (
                                    df_box.groupby("Kabupaten/Kota")["Cluster"]
                                    .apply(ambil_cluster_dominan)
                                    .reset_index(name="Cluster")  # hasil jadi DataFrame dengan kolom Cluster
                                    .sort_values(by="Cluster")
                                    .reset_index(drop=True)
                                )

                                # Format nama kabupaten/kota agar rapi
                                def format_nama(nama):
                                    nama = nama.strip().upper()
                                    return nama.title()

                                df_cluster_list["Kabupaten/Kota"] = df_cluster_list["Kabupaten/Kota"].apply(format_nama)

                                # =======================================================
                                # 📊 TABEL & GRAFIK BERDAMPINGAN
                                # =======================================================
                                col1, col2 = st.columns([1.0, 1.3])

                                with col1:
                                    st.markdown("### 📋 Daftar Kabupaten/Kota Berdasarkan Cluster")
                                    st.dataframe(
                                        df_cluster_list,
                                        use_container_width=False,
                                        width=380,
                                        hide_index=True
                                    )

                                with col2:
                                    # =======================================================
                                    # 📊 JUMLAH KABUPATEN/KOTA PER CLUSTER
                                    # =======================================================
                                    cluster_counts = (
                                        df_cluster_list.groupby("Cluster")["Kabupaten/Kota"]
                                        .count()
                                        .reset_index()
                                        .rename(columns={"Kabupaten/Kota": "Jumlah_KabKota"})
                                        .sort_values(by="Cluster", ascending=True)
                                    )

                                    # Warna batang disamakan dengan warna cluster di boxplot/peta
                                    cluster_colors_list = [
                                        cluster_colors.get(c, "#999999") for c in cluster_counts["Cluster"]
                                    ]

                                    st.markdown("### 📊 Jumlah Kabupaten/Kota pada Tiap Cluster")

                                    fig, ax = plt.subplots(figsize=(4.8, 3.3))
                                    sns.barplot(
                                        x="Cluster",
                                        y="Jumlah_KabKota",
                                        data=cluster_counts,
                                        palette=cluster_colors_list,
                                        ax=ax
                                    )

                                    # Tambahkan judul langsung di atas plot
                                    ax.set_title("Distribusi Jumlah Kabupaten/Kota per Cluster",
                                                fontsize=12, fontweight="bold", pad=10)

                                    # Tambahkan label di atas batang
                                    for container in ax.containers:
                                        ax.bar_label(
                                            container, fmt="%d", label_type="edge", fontsize=9, padding=2, color="#222"
                                        )

                                    ymax = cluster_counts["Jumlah_KabKota"].max()
                                    ax.set_ylim(0, ymax * 1.15)
                                    ax.set_xlabel("Cluster", fontsize=11)
                                    ax.set_ylabel("Jumlah Kab/Kota", fontsize=11)
                                    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)


                st.success("✅ Boxplot & Peta selesai dibuat!")

            # =====================================================
            # 📊 TAB B: TREN VARIABEL
            # =====================================================
            with tabB:
                tahun_unik = df["Tahun"].unique()
                n_tahun = len(tahun_unik)

                if n_tahun > 1:
                    # =================================================
                    # MODE: GRAFIK TREN (multi-year)
                    # =================================================
                    st.markdown("### 📊 Tren Tahunan Setiap Variabel")
                    st.markdown(
                        "Analisis tren maksimum, minimum, dan rata-rata untuk tiap variabel dari tahun yang tersedia.")

                    with st.spinner("🔄 Menghitung tren tahunan..."):
                        # Pastikan kolom Tahun numerik
                        df["Tahun"] = pd.to_numeric(
                            df["Tahun"], errors="coerce")

                        # Pilih kolom yang dipakai
                        df_trend = df[["Tahun"] + fitur].copy()

                        # Hitung statistik tahunan (max, min, mean)
                        stats = df_trend.groupby("Tahun").agg(
                            ["max", "min", "mean"]).round(2)
                        stats.columns = ["_".join(col)
                                         for col in stats.columns]
                        stats = stats.reset_index()

                        # Ubah kolom Tahun jadi string supaya rapi di sumbu X
                        stats["Tahun"] = stats["Tahun"].astype(str)

                        # Buat subplot dinamis sesuai jumlah variabel
                        n_vars = len(fitur)
                        fig, axes = plt.subplots(
                            1, n_vars, figsize=(5*n_vars, 3))
                        if n_vars == 1:
                            axes = [axes]

                        for i, var in enumerate(fitur):
                            ax = axes[i]
                            tahun_str = stats["Tahun"]

                            ax.plot(
                                tahun_str, stats[f"{var}_max"], marker="^", color="blue", label="Maksimum")
                            ax.plot(
                                tahun_str, stats[f"{var}_min"], marker="v", color="red", label="Minimum")
                            ax.plot(
                                tahun_str, stats[f"{var}_mean"], marker="o", color="green", label="Rata-rata")

                            ax.set_title(
                                f"({chr(97+i)}) Tren {var}", fontsize=12, fontweight="bold")
                            ax.set_xlabel("Tahun")
                            ax.set_ylabel(var)
                            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                            ax.set_xticks(range(len(tahun_str)))
                            ax.set_xticklabels(tahun_str)

                        # ✅ Legend di luar plot (bawah semua subplot)
                        fig.legend(
                            labels=["Maksimum", "Minimum", "Rata-rata"],
                            loc="lower center",
                            ncol=3,
                            frameon=False,
                            fontsize=9
                        )

                        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                        st.pyplot(fig)

                    st.success("✅ Grafik tren tahunan selesai dibuat!")

                else:
                    # =================================================
                    # MODE: PROFIL SATU TAHUN (single-year)
                    # =================================================
                    tahun_terpilih = int(tahun_unik[0])
                    st.markdown(
                        f"### 📊 Profil Variabel Tahun {tahun_terpilih}")
                    st.markdown(
                        "Menampilkan nilai minimum, rata-rata, dan maksimum untuk setiap variabel di tahun ini.")

                    # Hitung statistik ringkas
                    stats_single = df[fitur].describe(
                    ).T[["min", "mean", "max"]].round(2)
                    stats_single.columns = ["Minimum", "Rata-rata", "Maksimum"]

                    st.dataframe(stats_single.style.format("{:.2f}"))
                    st.success(
                        f"✅ Ringkasan variabel untuk tahun {tahun_terpilih} berhasil ditampilkan.")

                # =====================================================
                # 🏆 TAMBAHAN: DASHBOARD GAYA "TOP STATES" UNTUK TOP 10
                # =====================================================
                import matplotlib.cm as cm

                st.markdown("---")
                st.markdown(
                    "### 🏆 Top 10 Kabupaten/Kota Tertinggi dan Terendah per Variabel (Visual Dashboard)")

                if "Kabupaten/Kota" in df.columns:
                    for var in fitur:
                        st.markdown(f"#### 📊 {var}")
                        with st.spinner(f"Menghitung peringkat untuk {var}..."):
                            df_sorted = df[["Kabupaten/Kota",
                                            var, "Tahun"]].dropna(subset=[var])
                            df_mean = (
                                df_sorted.groupby("Kabupaten/Kota")[var]
                                .mean()
                                .reset_index()
                                .sort_values(by=var, ascending=False)
                            )

                            # ✅ Tambahkan "Kabupaten" di depan kalau belum ada kata "Kota"
                            def format_nama(nama):
                                nama = nama.strip().title()
                                if not nama.lower().startswith("kota"):
                                    return f"Kabupaten {nama}"
                                return nama

                            df_mean["Kabupaten/Kota"] = df_mean["Kabupaten/Kota"].apply(
                                format_nama)

                            top10_high = df_mean.head(
                                10).reset_index(drop=True)
                            top10_low = df_mean.tail(10).sort_values(
                                by=var, ascending=True).reset_index(drop=True)

                            col1, col2 = st.columns(2)

                            # === TERTINGGI ===
                            with col1:
                                st.markdown("**🔺 Top 10 Tertinggi**")
                                max_val = top10_high[var].max()
                                for _, row in top10_high.iterrows():
                                    bar_val = row[var] / max_val
                                    st.markdown(
                                        f"""
                                        <div style='margin-bottom:20px;'>  <!-- ✅ jarak antar bar -->
                                            <div style='display:flex; justify-content:space-between;'>
                                                <span style='font-weight:600;'>{row["Kabupaten/Kota"]}</span>
                                                <span style='font-weight:500;'>{row[var]:,.2f}</span>
                                            </div>
                                            <div style='background-color:#333; border-radius:5px; height:10px; margin-top:3px;'>
                                                <div style='background:linear-gradient(90deg, #ff4b5c, #ff8fa3); width:{bar_val*100:.1f}%; height:10px; border-radius:5px;'></div>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                            # === TERENDAH ===
                            with col2:
                                st.markdown("**🔻 Top 10 Terendah**")
                                max_val = top10_low[var].max()
                                for _, row in top10_low.iterrows():
                                    bar_val = row[var] / max_val
                                    st.markdown(
                                        f"""
                                        <div style='margin-bottom:20px;'>  <!-- ✅ jarak antar bar -->
                                            <div style='display:flex; justify-content:space-between;'>
                                                <span style='font-weight:600;'>{row["Kabupaten/Kota"]}</span>
                                                <span style='font-weight:500;'>{row[var]:,.2f}</span>
                                            </div>
                                            <div style='background-color:#333; border-radius:5px; height:10px; margin-top:3px;'>
                                                <div style='background:linear-gradient(90deg, #74b9ff, #a0c4ff); width:{bar_val*100:.1f}%; height:10px; border-radius:5px;'></div>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                    st.success(
                        "✅ Visualisasi gaya 'Top States' berhasil ditampilkan!")
                else:
                    st.warning(
                        "⚠️ Kolom 'Kabupaten/Kota' tidak ditemukan dalam dataset.")

                # =====================================================
                # 📈 TAB C: SILHOUETTE PLOT
                # =====================================================
                with tabC:
                    st.markdown("### 📈 Silhouette Plot per Metode")
                    with st.spinner("🔄 Membuat Silhouette Plot..."):
                        n_methods = len(hasil)
                        n_cols = 2 if n_methods == 2 else 1
                        cols = st.columns(n_cols)

                        for i, (metode, data) in enumerate(hasil.items()):
                            with cols[i % n_cols]:
                                fig, ax = plt.subplots(figsize=(6, 4))

                                # ✅ pastikan semua key tersedia dan tidak None
                                X_scaled = data.get("X_scaled")
                                labels = data.get("labels")
                                k_val = data.get("k")
                                sil_val = data.get("sil", None)

                                if X_scaled is None or labels is None:
                                    st.warning(f"⚠️ Data untuk {metode} belum lengkap, tidak bisa buat plot.")
                                    continue

                                # ✅ kalau k belum ada / None → hitung manual
                                if k_val is None:
                                    k_val = len(np.unique(labels))
                                    data["k"] = k_val  # update biar aman untuk plotting

                                silhouette_vals = silhouette_samples(X_scaled, labels)

                                y_lower, y_upper = 0, 0
                                yticks = []
                                for c in range(k_val):
                                    c_sil = silhouette_vals[labels == c]
                                    c_sil.sort()
                                    y_upper += len(c_sil)
                                    ax.barh(range(y_lower, y_upper), c_sil, height=1.0)
                                    yticks.append((y_lower + y_upper) / 2)
                                    y_lower += len(c_sil)

                                # garis rata-rata silhouette
                                if sil_val is not None:
                                    ax.axvline(sil_val, color="red", linestyle="--")

                                ax.set_yticks(yticks)
                                ax.set_yticklabels(range(k_val))
                                ax.set_xlabel("Silhouette Coefficient")
                                ax.set_ylabel("Cluster")
                                ax.set_title(f"{metode}", fontsize=11, fontweight="bold")
                                ax.grid(True, linestyle="--", alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                    st.success("✅ Silhouette plot selesai dibuat!")

            # =====================================================
            # 🎯 TAB D: SCATTER / PAIRGRID
            # =====================================================
            with tabD:
                st.markdown("### 🎯 Scatter Plot per Variabel")
                with st.spinner("🔄 Menyiapkan Scatter Plot..."):
                    for metode, data in hasil.items():
                        st.markdown(f"#### 🧩 {metode}")
                        df_plot = data["df"].copy()
                        df_plot["Tahun"] = df_plot["Tahun"].astype(int)
                        n_tahun = df_plot["Tahun"].nunique()
                        n_vars = len(fitur)

                        if n_tahun > 1:
                            variabels = [(f"({chr(97+i)}) {var}", var)
                                         for i, var in enumerate(fitur)]
                            n_cols = 3 if n_vars >= 3 else n_vars
                            cols = st.columns(n_cols)
                            for i, (judul, var) in enumerate(variabels):
                                with cols[i % n_cols]:
                                    df_pivot = df_plot.pivot_table(
                                        index="Kabupaten/Kota", columns="Tahun",
                                        values=var, aggfunc="mean"
                                    )
                                    tahun_unik = sorted(
                                        df_plot["Tahun"].unique())
                                    df_pivot = df_pivot.reindex(
                                        columns=tahun_unik)
                                    df_pivot["Cluster"] = (
                                        df_plot.groupby(
                                            "Kabupaten/Kota")["Cluster"]
                                        .first()
                                        .reindex(df_pivot.index)
                                    )
                                    g = sns.PairGrid(
                                        df_pivot, hue="Cluster", palette="husl", height=2.2)
                                    g.map_lower(sns.scatterplot, s=35,
                                                edgecolor="k", linewidth=0.3)
                                    g.map_upper(
                                        sns.scatterplot, s=35, edgecolor="k", linewidth=0.3, alpha=0.5)
                                    g.map_diag(
                                        sns.kdeplot, fill=True, alpha=0.6)
                                    g.fig.suptitle(
                                        judul, y=1.02, fontsize=12, fontweight="bold")
                                    plt.tight_layout()
                                    st.pyplot(g.fig)
                                    plt.close(g.fig)
                        else:
                            st.info(
                                f"⚠️ Scatter plot tidak relevan untuk {metode} karena hanya 1 tahun data.")
                        st.divider()
                st.success("✅ Scatter selesai dibuat!")

# =====================================================
# TAB 3️⃣: PERBANDINGAN TIGA METODE (AUTO)
# =====================================================
with tab3:
    st.header("Perbandingan Tiga Metode")

    # =====================================================
    # 🧹 AUTO RESET jika user ganti pilihan
    # =====================================================
    def reset_experiment_state_tab3():
        for key in ["triple_results", "fitur_tab3", "df_filtered_tab3"]:
            if key in st.session_state:
                del st.session_state[key]

    # =========================================================
    # 1️⃣ PILIH SUMBER DATASET (TAB 3) + VALIDASI PER SHEET
    # =========================================================
    import io
    import re
    import pandas as pd

    # Daftar kolom wajib sesuai template
    required_columns = [
        "Kabupaten/Kota",
        "Harga Telur Ayam Ras (Rp)",
        "Konsumsi Telur Ayam Ras per Kapita",
        "Pengeluaran Telur Ayam Ras (Rp)"
    ]

    # --- fungsi bantu untuk normalisasi nama kolom ---
    def normalize_column(col):
        col = str(col)
        col = col.encode('utf-8', 'ignore').decode('utf-8')
        col = re.sub(r'\s+', ' ', col)
        col = col.replace('\xa0', ' ').replace('\u200b', '')
        return col.strip().lower()

    # --- fungsi bantu untuk validasi tiap sheet ---
    def validate_sheet_columns(sheet_name, df_sheet):
        cols = [normalize_column(c) for c in df_sheet.columns]
        normalized_required = [normalize_column(c) for c in required_columns]
        missing = [c for c in normalized_required if c not in cols]
        return missing

    # =========================================================
    # 📂 PILIH SUMBER DATASET
    # =========================================================
    pilihan_data = st.radio(
        "Pilih sumber dataset:",
        ["Gunakan dataset bawaan", "Upload dataset sendiri"],
        horizontal=True,
        key="pilihan_data_tab3",
        on_change=reset_experiment_state_tab3
    )
    st.markdown("---")

    # =========================================================
    # 🔍 BACA DAN VALIDASI SETIAP SHEET
    # =========================================================
    if pilihan_data == "Gunakan dataset bawaan":
        excel_path = "data/Dataset Ready.xlsx"
        xls = pd.ExcelFile(excel_path)
    else:
        uploaded_file = st.file_uploader(
            "📂 Unggah file Excel (.xlsx)",
            type=["xlsx"],
            key="upload_tab3_validasi",  # ✅ key unik, beda dari tab 1 & 2
            on_change=reset_experiment_state_tab3
        )
        if not uploaded_file:
            st.warning(
                "⚠️ Silakan upload dataset terlebih dahulu untuk melanjutkan.")
            st.stop()
        try:
            xls = pd.ExcelFile(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file Excel: {e}")
            st.stop()

    sheet_names = xls.sheet_names
    df_list = []
    error_sheets = {}

    # Loop tiap sheet & validasi kolom
    for sheet in sheet_names:
        temp = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
        missing = validate_sheet_columns(sheet, temp)

        if missing:
            missing_display = [
                required_columns[[normalize_column(
                    c) for c in required_columns].index(m)]
                for m in missing
            ]
            error_sheets[sheet] = missing_display
        else:
            try:
                temp["Tahun"] = int(sheet)
            except ValueError:
                temp["Tahun"] = sheet
            df_list.append(temp)

    # =========================================================
    # 🚨 JIKA ADA SHEET ERROR
    # =========================================================
    if error_sheets:
        st.error(
            "❌ Dataset tidak valid. Beberapa sheet memiliki kolom yang tidak lengkap:")
        for sheet, missing_cols in error_sheets.items():
            st.warning(
                f"📄 Sheet **{sheet}** hilang kolom: {', '.join(missing_cols)}")

        st.info("""
        Pastikan semua sheet memiliki **kolom yang sama** seperti template:
        - Kabupaten/Kota  
        - Harga Telur Ayam Ras (Rp)  
        - Konsumsi Telur Ayam Ras per Kapita  
        - Pengeluaran Telur Ayam Ras (Rp)
        """)

        # tombol download template
        template_df = pd.DataFrame({
            "Kabupaten/Kota": ["KOTA JAKARTA", "KAB. BANDUNG"],
            "Harga Telur Ayam Ras (Rp)": [28000, 27000],
            "Konsumsi Telur Ayam Ras per Kapita": [2.1, 1.8],
            "Pengeluaran Telur Ayam Ras (Rp)": [42000, 39000]
        })
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            template_df.to_excel(writer, index=False, sheet_name="Contoh")

        st.download_button(
            label="📥 Unduh Template Dataset (Format Benar)",
            data=buffer.getvalue(),
            file_name="Template_Dataset_Telur_Ayam_Ras.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.stop()

    # =========================================================
    # ✅ SEMUA SHEET VALID → PILIH RANGE TAHUN
    # =========================================================
    col1, col2 = st.columns(2)
    with col1:
        tahun_dari = st.selectbox(
            "📅 Pilih Tahun Awal:",
            sheet_names,
            index=0,
            key="tahun_dari_tab3",
            on_change=reset_experiment_state_tab3
        )
    with col2:
        tahun_sampai = st.selectbox(
            "📅 Pilih Tahun Akhir:",
            sheet_names,
            index=len(sheet_names) - 1,
            key="tahun_sampai_tab3",
            on_change=reset_experiment_state_tab3
        )

    # =========================================================
    # 🚨 ERROR HANDLING: Tahun awal tidak boleh lebih besar
    # =========================================================
    start_idx = sheet_names.index(tahun_dari)
    end_idx = sheet_names.index(tahun_sampai)

    if start_idx > end_idx:
        st.error(
            f"⚠️ Tahun awal ({tahun_dari}) tidak boleh lebih besar dari tahun akhir ({tahun_sampai}). "
            "Silakan periksa kembali urutan tahun yang dipilih."
        )
        st.stop()  # hentikan eksekusi agar tidak lanjut ke bawah

    # =========================================================
    # ✅ Jika valid → proses data
    # =========================================================
    selected_sheets = sheet_names[start_idx:end_idx + 1]

    # Filter sheet sesuai range tahun
    df_filtered = [
        df for df in df_list if str(df["Tahun"].iloc[0]) in selected_sheets
    ]
    df = pd.concat(df_filtered, ignore_index=True)

    st.success(
        f"✅ Semua sheet valid. Data dari tahun {tahun_dari}–{tahun_sampai} berhasil dimuat "
    )

    # =========================================================
    # 2️⃣ PILIH VARIABEL UNTUK ANALISIS
    # =========================================================
    st.markdown("### ⚙️ Konfigurasi Analisis")

    all_features = [
        "Harga Telur Ayam Ras (Rp)",
        "Konsumsi Telur Ayam Ras per Kapita",
        "Pengeluaran Telur Ayam Ras (Rp)"
    ]

    selected_features = st.multiselect(
        "📊 Pilih Variabel yang Akan Digunakan untuk Clustering:",
        options=all_features,
        default=all_features,
        help="Kamu bisa pilih 1–3 variabel.",
        key="fitur_tab3",
        on_change=reset_experiment_state_tab3  # fungsi reset versi tab 3
    )

    # 🚨 Minimal 1 variabel wajib dipilih
    if len(selected_features) == 0:
        st.warning("⚠️ Minimal pilih satu variabel untuk melanjutkan.")
        st.stop()

    # ✅ Simpan variabel terpilih agar sesuai dengan pilihan terakhir
    st.session_state["fitur_tab3"] = selected_features
    fitur = selected_features

    # ✅ Simpan dataframe hasil filter (kalau digunakan di proses berikutnya)
    st.session_state["df_filtered_tab3"] = df


    # =====================================================
    # 3️⃣ ATUR JUMLAH CLUSTER UNTUK METODE MANUAL
    # =====================================================
    st.markdown("### ⚙️ Jumlah Cluster untuk Setiap Metode")

    col1, col2 = st.columns(2)
    with col1:
        k_kmeans = st.slider("Jumlah Cluster (K-Means)",
                             2, 7, 3, key="k_kmeans_tab3")
    with col2:
        k_ahc = st.slider("Jumlah Cluster (AHC)", 2, 7, 3, key="k_ahc_tab3")

    # =====================================================
    # 4️⃣ JALANKAN SEMUA METODE (OTOMATIS)
    # =====================================================
    st.caption(
        "Semua metode akan dijalankan otomatis: **K-Means**, **AHC**, dan **Intelligent K-Medoids**."
    )

    if st.button("🚀 Jalankan Semua Metode", key="run_triple_tab3"):
        df_base = st.session_state["df_filtered_tab3"]
        fitur = st.session_state["fitur_tab3"]

        metode_k = [
            ("K-Means", k_kmeans),
            ("Agglomerative Hierarchical Clustering (AHC)", k_ahc),
            ("Intelligent K-Medoids", None)
        ]

        hasil = {}
        data = {}  # ✅ tambahkan inisialisasi biar gak NameError
        progress = st.progress(0)

        for i, (metode, k) in enumerate(metode_k, start=1):
            df_temp = df_base.copy()
            start_time = time.perf_counter()
            df_temp = clean_data(df_temp, fitur)
            X_scaled = scale_features(df_temp, fitur)

            if metode == "K-Means":
                labels = run_kmeans(X_scaled, k)
            elif metode == "Agglomerative Hierarchical Clustering (AHC)":
                labels = run_ahc(X_scaled, k)
            else:
                labels, k_auto, sil = run_intelligent_kmedoids_streamlit(X_scaled)
                data["k"] = k_auto  # ✅ simpan hasil cluster otomatis
                st.info(
                    f"🤖 Jumlah cluster optimal hasil Intelligent K-Medoids: **{k_auto}**"
                )

            df_temp["Cluster"] = labels
            sil, dbi = evaluate_clusters(X_scaled, labels)
            end_time = time.perf_counter()


            waktu_komputasi = end_time - start_time
            waktu_fmt = f"{waktu_komputasi*1000:.2f} ms" if waktu_komputasi < 1 else f"{waktu_komputasi:.2f} detik"

            hasil[metode] = {
                "df": df_temp,
                "labels": labels,
                "k": k,
                "sil": sil,
                "dbi": dbi,
                "time": waktu_fmt,
                "X_scaled": X_scaled
            }
            progress.progress(i / 3)

        progress.empty()
        st.session_state["triple_results"] = hasil
        st.success("✅ Ketiga metode berhasil dijalankan!")

    # =====================================================
    # 5️⃣ VISUALISASI HASIL
    # =====================================================
    if "triple_results" in st.session_state:
        hasil = st.session_state["triple_results"]
        fitur = st.session_state["fitur_tab3"]

        st.markdown("---")
        st.subheader("📊 Hasil Evaluasi Perbandingan")
        for metode, data in hasil.items():
            st.markdown(f"#### 🧩 {metode}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Silhouette Score", f"{data['sil']:.3f}")
            col2.metric("Davies–Bouldin Index", f"{data['dbi']:.3f}")
            col3.metric("Waktu Komputasi", data["time"])

        # =====================================================
        # 6️⃣ TAB VISUALISASI (URUTAN SAMA SEPERTI TAB1 & TAB2)
        # =====================================================
        tabA, tabB, tabC, tabD = st.tabs([
            "📦 Boxplot & Peta",
            "📊 Tren Tahunan",
            "📈 Silhouette Plot",
            "🎯 Scatter Plot"
        ])

        # === BOX PLOT & PETA ===
        with tabA:
            st.markdown("### 📦 Distribusi Nilai & Peta per Metode")
            from modules.interpretasi import interpretasi_untuk_legend_otomatis
            import seaborn as sns
            import matplotlib.colors as mcolors
            import leafmap.foliumap as leafmap
            import geopandas as gpd
            import folium

            custom_palette = [
                "#4C72B0", "#DD8452", "#55A868", "#C44E52",
                "#8172B3", "#937860", "#DA8BC3", "#8C8C3E",
                "#64B5CD", "#FFB300", "#009688", "#AB47BC",
                "#7E57C2", "#EF5350", "#26A69A", "#FF7043"
            ]

            for metode, data in hasil.items():
                st.markdown(f"## 🧩 {metode}")

                df_box = data["df"].copy()
                n_clusters = df_box["Cluster"].nunique()
                cluster_palette_hex = custom_palette[:n_clusters]

                cluster_colors = {
                    c: cluster_palette_hex[i % len(cluster_palette_hex)]
                    for i, c in enumerate(sorted(df_box["Cluster"].unique()))
                }

                # ======================================================
                # 📊 BOX PLOT
                # ======================================================
                st.markdown("### 📦 Distribusi Nilai Variabel per Tahun")
                n_vars = len(fitur)
                n_cols = min(n_vars, 3)
                cols = st.columns(n_cols, gap="large")

                if n_vars == 2:
                    cols = [st.empty(), *cols, st.empty()]

                for i, var in enumerate(fitur):
                    target_col = cols[i + 1] if n_vars == 2 else cols[i]
                    with target_col:
                        fig, ax = plt.subplots(figsize=(4.3, 3.5))
                        sns.boxplot(
                            x="Tahun", y=var, hue="Cluster",
                            data=df_box, palette=cluster_palette_hex, ax=ax,
                            fliersize=2, linewidth=0.8
                        )
                        ax.set_xlabel("Tahun")
                        ax.set_ylabel(var)
                        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
                        ax.get_legend().remove()
                        plt.tight_layout(rect=[0, 0.05, 1, 1], pad=1.0)
                        st.pyplot(fig)
                        plt.close(fig)

                legend_items = []
                for j, cluster_id in enumerate(sorted(df_box["Cluster"].unique())):
                    color = cluster_palette_hex[j % len(cluster_palette_hex)]
                    legend_items.append(
                        f"<span style='display:inline-flex; align-items:center; gap:5px; margin-right:10px;'>"
                        f"<span style='width:14px; height:14px; background-color:{color}; border-radius:3px; display:inline-block;'></span>"
                        f"<span style='font-size:13px; color:#333;'>Cluster {cluster_id}</span>"
                        f"</span>"
                    )

                legend_html = (
                    "<div style='text-align:center; margin-top:10px; display:flex; justify-content:center; flex-wrap:wrap; gap:10px;'>"
                    + "".join(legend_items)
                    + "</div>"
                )
                st.markdown(legend_html, unsafe_allow_html=True)
                st.markdown("---")

                # ======================================================
                # 🗺️ PETA INTERAKTIF
                # ======================================================
                st.markdown("### 🗺️ Peta Visualisasi Hasil Clustering")

                @st.cache_data
                def load_geojson():
                    geo_path = "data/Indonesia_cities.geojson"
                    gdf = gpd.read_file(geo_path)
                    gdf["NAME_2"] = gdf["NAME_2"].str.strip().str.upper()
                    return gdf

                @st.cache_data
                def merge_data_for_map(df_map, fitur):
                    gdf = load_geojson()
                    gdf["NAME_2"] = (
                        gdf["NAME_2"].str.replace("Kabupaten", "", case=False)
                        .str.replace("KOTA", "", case=False)
                        .str.strip().str.upper()
                    )
                    df_map["Kabupaten/Kota"] = (
                        df_map["Kabupaten/Kota"].str.replace(
                            "KABUPATEN", "", case=False)
                        .str.replace("KOTA", "", case=False)
                        .str.strip().str.upper()
                    )
                    cols_merge = ["Kabupaten/Kota", "Cluster"] + \
                        [c for c in fitur if c in df_map.columns]
                    gdf = gdf.merge(
                        df_map[cols_merge], left_on="NAME_2", right_on="Kabupaten/Kota", how="left")
                    return gdf

                gdf = merge_data_for_map(df_box, fitur)

                def make_tooltip(row):
                    if pd.isna(row["Kabupaten/Kota"]) or pd.isna(row["Cluster"]):
                        return "<b>Tidak Ada Data</b>"

                    nama = str(row["Kabupaten/Kota"]).title().strip()
                    if not nama.startswith("Kota"):
                        nama = f"{nama}"

                    cluster_val = row["Cluster"]
                    teks = f"<b>{nama}</b><br><b>Cluster:</b> {int(cluster_val)}<hr style='margin:3px 0;'>"

                    for f in fitur:
                        if f in row and pd.notnull(row[f]):
                            if isinstance(row[f], (int, float)):
                                teks += f"<b>{f}:</b> {row[f]:,.2f}<br>"
                            else:
                                teks += f"<b>{f}:</b> {row[f]}<br>"
                    return teks

                gdf["info"] = gdf.apply(make_tooltip, axis=1)

                m = leafmap.Map(center=[-2.5, 118], zoom=5)
                m.add_basemap("CartoDB.Positron")

                from folium import GeoJson, GeoJsonTooltip

                geo_layer = GeoJson(
                    gdf[["geometry", "Cluster", "info"]],
                    name="Peta Cluster",
                    style_function=lambda x: {
                        "fillColor": (
                            cluster_colors.get(x["properties"].get("Cluster"))
                            if pd.notnull(x["properties"].get("Cluster"))
                            else "#C8C8C8"
                        ),
                        "color": "#4d4d4d",
                        "weight": 0.4,
                        "opacity": 0.6,
                        "fillOpacity": 0.9,
                    },
                    tooltip=GeoJsonTooltip(
                        fields=["info"],
                        aliases=[""],
                        labels=False,
                        sticky=True,
                        style=(
                            "background-color: rgba(255,255,255,0.95); "
                            "color: #222; font-size: 12px; "
                            "font-family: Arial; border-radius: 4px; "
                            "padding: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
                        ),
                    ),
                )

                m.add_layer(geo_layer)
                m.to_streamlit(height=550)

                # ======================================================
                # 🧭 INTERPRETASI CLUSTER
                # ======================================================
                auto_labels = interpretasi_untuk_legend_otomatis(df_box, fitur)
                st.markdown("### Interpretasi Cluster")

                for cluster in sorted(auto_labels.keys()):
                    color = cluster_colors.get(
                        cluster, list(cluster_colors.values())[0])
                    label = auto_labels[cluster]
                    st.markdown(
                        f"""
                        <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                            <div style='width:16px; height:16px; background-color:{color};
                                        border-radius:3px; flex-shrink:0;'></div>
                            <div style='font-size:14px; color:#333; line-height:1.4;'>
                                <b>Cluster {cluster} :</b> {label}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # =======================================================
                # 📋 TABEL & 📊 GRAFIK JUMLAH KABUPATEN/KOTA PER CLUSTER (DOMINAN, FIX AKURAT)
                # =======================================================
                with st.spinner("📋 Menyiapkan Daftar Kabupaten/Kota..."):

                    # Pastikan kolom Cluster tersedia
                    if "Cluster" not in df_box.columns:
                        st.warning("⚠️ Kolom **Cluster** belum tersedia. Jalankan proses clustering terlebih dahulu.")
                    else:
                        # 1️⃣ Hitung cluster dominan (mode) tiap kabupaten/kota di seluruh tahun
                        def ambil_cluster_dominan(series_cluster):
                            mode_val = series_cluster.mode()
                            if not mode_val.empty:
                                return mode_val.iloc[0]
                            else:
                                return series_cluster.iloc[-1]  # fallback ke cluster terakhir

                        df_cluster_list = (
                            df_box.groupby("Kabupaten/Kota")["Cluster"]
                            .apply(ambil_cluster_dominan)
                            .reset_index(name="Cluster")  # hasil jadi DataFrame dengan kolom Cluster
                            .sort_values(by="Cluster")
                            .reset_index(drop=True)
                        )

                        # Format nama kabupaten/kota agar rapi
                        def format_nama(nama):
                            nama = nama.strip().upper()
                            return nama.title()

                        df_cluster_list["Kabupaten/Kota"] = df_cluster_list["Kabupaten/Kota"].apply(format_nama)

                        # =======================================================
                        # 📊 TABEL & GRAFIK BERDAMPINGAN
                        # =======================================================
                        col1, col2 = st.columns([1.0, 1.3])

                        with col1:
                            st.markdown("### 📋 Daftar Kabupaten/Kota Berdasarkan Cluster")
                            st.dataframe(
                                df_cluster_list,
                                use_container_width=False,
                                width=380,
                                hide_index=True
                            )

                        with col2:
                            # =======================================================
                            # 📊 JUMLAH KABUPATEN/KOTA PER CLUSTER
                            # =======================================================
                            cluster_counts = (
                                df_cluster_list.groupby("Cluster")["Kabupaten/Kota"]
                                .count()
                                .reset_index()
                                .rename(columns={"Kabupaten/Kota": "Jumlah_KabKota"})
                                .sort_values(by="Cluster", ascending=True)
                            )

                            # Warna batang disamakan dengan warna cluster di boxplot/peta
                            cluster_colors_list = [
                                cluster_colors.get(c, "#999999") for c in cluster_counts["Cluster"]
                            ]

                            st.markdown("### 📊 Jumlah Kabupaten/Kota pada Tiap Cluster")

                            fig, ax = plt.subplots(figsize=(4.8, 3.3))
                            sns.barplot(
                                x="Cluster",
                                y="Jumlah_KabKota",
                                data=cluster_counts,
                                palette=cluster_colors_list,
                                ax=ax
                            )

                            # Tambahkan judul langsung di atas plot
                            ax.set_title("Distribusi Jumlah Kabupaten/Kota per Cluster",
                                        fontsize=12, fontweight="bold", pad=10)

                            # Tambahkan label di atas batang
                            for container in ax.containers:
                                ax.bar_label(
                                    container, fmt="%d", label_type="edge", fontsize=9, padding=2, color="#222"
                                )

                            ymax = cluster_counts["Jumlah_KabKota"].max()
                            ax.set_ylim(0, ymax * 1.15)
                            ax.set_xlabel("Cluster", fontsize=11)
                            ax.set_ylabel("Jumlah Kab/Kota", fontsize=11)
                            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        st.markdown("---")


        # === TREN (SAMA DENGAN TAB1 LOGIKA) ===
        with tabB:
            tahun_unik = df["Tahun"].unique()
            n_tahun = len(tahun_unik)

            if n_tahun > 1:
                # =================================================
                # MODE: GRAFIK TREN (multi-year)
                # =================================================
                st.markdown("### 📊 Tren Tahunan Setiap Variabel")
                st.markdown(
                    "Analisis tren maksimum, minimum, dan rata-rata untuk tiap variabel dari tahun yang tersedia.")

                with st.spinner("🔄 Menghitung tren tahunan..."):
                    # Pastikan kolom Tahun numerik
                    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")

                    # Pilih kolom yang dipakai
                    df_trend = df[["Tahun"] + fitur].copy()

                    # Hitung statistik tahunan (max, min, mean)
                    stats = df_trend.groupby("Tahun").agg(
                        ["max", "min", "mean"]).round(2)
                    stats.columns = ["_".join(col) for col in stats.columns]
                    stats = stats.reset_index()

                    # Ubah kolom Tahun jadi string supaya rapi di sumbu X
                    stats["Tahun"] = stats["Tahun"].astype(str)

                    # Buat subplot dinamis sesuai jumlah variabel
                    n_vars = len(fitur)
                    fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 3))
                    if n_vars == 1:
                        axes = [axes]

                    for i, var in enumerate(fitur):
                        ax = axes[i]
                        tahun_str = stats["Tahun"]

                        ax.plot(
                            tahun_str, stats[f"{var}_max"], marker="^", color="blue", label="Maksimum")
                        ax.plot(
                            tahun_str, stats[f"{var}_min"], marker="v", color="red", label="Minimum")
                        ax.plot(
                            tahun_str, stats[f"{var}_mean"], marker="o", color="green", label="Rata-rata")

                        ax.set_title(f"({chr(97+i)}) Tren {var}",
                                     fontsize=12, fontweight="bold")
                        ax.set_xlabel("Tahun")
                        ax.set_ylabel(var)
                        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                        ax.set_xticks(range(len(tahun_str)))
                        ax.set_xticklabels(tahun_str)

                    # ✅ Legend di luar plot (bawah semua subplot)
                    fig.legend(
                        labels=["Maksimum", "Minimum", "Rata-rata"],
                        loc="lower center",
                        ncol=3,
                        frameon=False,
                        fontsize=9
                    )

                    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                    st.pyplot(fig)

                st.success("✅ Grafik tren tahunan selesai dibuat!")

            else:
                # =================================================
                # MODE: PROFIL SATU TAHUN (single-year)
                # =================================================
                tahun_terpilih = int(tahun_unik[0])
                st.markdown(f"### 📊 Profil Variabel Tahun {tahun_terpilih}")
                st.markdown(
                    "Menampilkan nilai minimum, rata-rata, dan maksimum untuk setiap variabel di tahun ini.")

                # Hitung statistik ringkas
                stats_single = df[fitur].describe(
                ).T[["min", "mean", "max"]].round(2)
                stats_single.columns = ["Minimum", "Rata-rata", "Maksimum"]

                st.dataframe(stats_single.style.format("{:.2f}"))
                st.success(
                    f"✅ Ringkasan variabel untuk tahun {tahun_terpilih} berhasil ditampilkan.")

            # =====================================================
            # 🏆 TAMBAHAN: DASHBOARD GAYA "TOP STATES" UNTUK TOP 10
            # =====================================================
            import matplotlib.cm as cm

            st.markdown("---")
            st.markdown(
                "### 🏆 Top 10 Kabupaten/Kota Tertinggi dan Terendah per Variabel (Visual Dashboard)")

            if "Kabupaten/Kota" in df.columns:
                for var in fitur:
                    st.markdown(f"#### 📊 {var}")
                    with st.spinner(f"Menghitung peringkat untuk {var}..."):
                        df_sorted = df[["Kabupaten/Kota", var,
                                        "Tahun"]].dropna(subset=[var])
                        df_mean = (
                            df_sorted.groupby("Kabupaten/Kota")[var]
                            .mean()
                            .reset_index()
                            .sort_values(by=var, ascending=False)
                        )

                        # ✅ Tambahkan "Kabupaten" di depan kalau belum ada kata "Kota"
                        def format_nama(nama):
                            nama = nama.strip().title()
                            if not nama.lower().startswith("kota"):
                                return f"Kabupaten {nama}"
                            return nama

                        df_mean["Kabupaten/Kota"] = df_mean["Kabupaten/Kota"].apply(
                            format_nama)

                        top10_high = df_mean.head(10).reset_index(drop=True)
                        top10_low = df_mean.tail(10).sort_values(
                            by=var, ascending=True).reset_index(drop=True)

                        col1, col2 = st.columns(2)

                        # === TERTINGGI ===
                        with col1:
                            st.markdown("**🔺 Top 10 Tertinggi**")
                            max_val = top10_high[var].max()
                            for _, row in top10_high.iterrows():
                                bar_val = row[var] / max_val
                                st.markdown(
                                    f"""
                                    <div style='margin-bottom:20px;'>  <!-- ✅ jarak antar bar -->
                                        <div style='display:flex; justify-content:space-between;'>
                                            <span style='font-weight:600;'>{row["Kabupaten/Kota"]}</span>
                                            <span style='font-weight:500;'>{row[var]:,.2f}</span>
                                        </div>
                                        <div style='background-color:#333; border-radius:5px; height:10px; margin-top:3px;'>
                                            <div style='background:linear-gradient(90deg, #ff4b5c, #ff8fa3); width:{bar_val*100:.1f}%; height:10px; border-radius:5px;'></div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        # === TERENDAH ===
                        with col2:
                            st.markdown("**🔻 Top 10 Terendah**")
                            max_val = top10_low[var].max()
                            for _, row in top10_low.iterrows():
                                bar_val = row[var] / max_val
                                st.markdown(
                                    f"""
                                    <div style='margin-bottom:20px;'>  <!-- ✅ jarak antar bar -->
                                        <div style='display:flex; justify-content:space-between;'>
                                            <span style='font-weight:600;'>{row["Kabupaten/Kota"]}</span>
                                            <span style='font-weight:500;'>{row[var]:,.2f}</span>
                                        </div>
                                        <div style='background-color:#333; border-radius:5px; height:10px; margin-top:3px;'>
                                            <div style='background:linear-gradient(90deg, #74b9ff, #a0c4ff); width:{bar_val*100:.1f}%; height:10px; border-radius:5px;'></div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                st.success(
                    "✅ Visualisasi gaya 'Top States' berhasil ditampilkan!")
            else:
                st.warning(
                    "⚠️ Kolom 'Kabupaten/Kota' tidak ditemukan dalam dataset.")

        # =====================================================
        # 📈 TAB C: SILHOUETTE PLOT (TIGA METODE)
        # =====================================================
        with tabC:
            st.markdown("### 📈 Silhouette Plot per Metode")
            with st.spinner("🔄 Membuat Silhouette Plot..."):
                n_methods = len(hasil)
                n_cols = 3 if n_methods >= 3 else 2
                cols = st.columns(n_cols)

                for i, (metode, data) in enumerate(hasil.items()):
                    with cols[i % n_cols]:
                        X_scaled = data.get("X_scaled")
                        labels = data.get("labels")
                        sil_val = data.get("sil")
                        k_val = data.get("k")

                        # --- Cegah error kalau ada data kosong ---
                        if X_scaled is None or labels is None:
                            st.warning(f"⚠️ Data untuk {metode} belum lengkap.")
                            continue

                        # --- Kalau 'k' belum terset, hitung otomatis dari label unik ---
                        if k_val is None or not isinstance(k_val, int):
                            k_val = len(np.unique(labels))
                            data["k"] = k_val

                        # --- Buat plot silhouette ---
                        fig, ax = plt.subplots(figsize=(6, 4))
                        silhouette_vals = silhouette_samples(X_scaled, labels)
                        y_lower, y_upper = 0, 0
                        yticks = []

                        # ✅ ubah 'c' jadi 'cluster_id' biar gak warning
                        for cluster_id in range(k_val):
                            cluster_sil_vals = silhouette_vals[labels == cluster_id]
                            cluster_sil_vals.sort()
                            y_upper += len(cluster_sil_vals)
                            ax.barh(range(y_lower, y_upper), cluster_sil_vals, height=1.0)
                            yticks.append((y_lower + y_upper) / 2)
                            y_lower += len(cluster_sil_vals)

                        # --- Garis rata-rata silhouette ---
                        if sil_val is not None:
                            ax.axvline(sil_val, color="red", linestyle="--", label="Rata-rata Silhouette")

                        # --- Gaya visualisasi ---
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(range(k_val))
                        ax.set_xlabel("Silhouette Coefficient")
                        ax.set_ylabel("Cluster")
                        ax.set_title(f"{metode}", fontsize=11, fontweight="bold")
                        ax.legend(loc="lower right")
                        ax.grid(True, linestyle="--", alpha=0.3)

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

            st.success("✅ Silhouette plot selesai dibuat!")


        # =====================================================
        # TAB D: SCATTER PER VARIABEL (LAYOUT HORIZONTAL PER METODE)
        # =====================================================
        with tabD:
            st.markdown("### 🎯 Scatter Plot per Variabel")
            st.caption(
                "Menampilkan hubungan antar tahun dan distribusi tiap variabel dalam satu baris per metode.")

            with st.spinner("🔄 Menyiapkan Scatter Plot..."):
                for metode, data in hasil.items():
                    st.markdown(f"## 🧩 {metode}")
                    df_plot = data["df"].copy()
                    df_plot["Tahun"] = df_plot["Tahun"].astype(int)
                    n_tahun = df_plot["Tahun"].nunique()
                    n_vars = len(fitur)

                    if n_tahun > 1:
                        # === Susun variabel horizontal ===
                        variabels = [(f"({chr(97+i)}) {var}", var)
                                     for i, var in enumerate(fitur)]
                        cols = st.columns(len(variabels))

                        for i, (judul, var) in enumerate(variabels):
                            with cols[i]:
                                df_pivot = df_plot.pivot_table(
                                    index="Kabupaten/Kota",
                                    columns="Tahun",
                                    values=var,
                                    aggfunc="mean"
                                )
                                tahun_unik = sorted(df_plot["Tahun"].unique())
                                df_pivot = df_pivot.reindex(columns=tahun_unik)
                                df_pivot["Cluster"] = (
                                    df_plot.groupby(
                                        "Kabupaten/Kota")["Cluster"]
                                    .first()
                                    .reindex(df_pivot.index)
                                )

                                # === PairGrid setup ===
                                g = sns.PairGrid(
                                    df_pivot, hue="Cluster", palette="husl", height=2.4)
                                g.map_lower(sns.scatterplot, s=35,
                                            edgecolor="k", linewidth=0.3)
                                g.map_upper(
                                    sns.scatterplot, s=35, edgecolor="k", linewidth=0.3, alpha=0.6)
                                g.map_diag(sns.kdeplot, fill=True, alpha=0.6)

                                # === Styling ===
                                g.fig.suptitle(
                                    judul, y=1.03, fontsize=11, fontweight="bold")
                                plt.tight_layout()
                                st.pyplot(g.fig)
                                plt.close(g.fig)

                        st.divider()  # garis pemisah antar metode

                    else:
                        st.info(f"""
                        ⚠️ **Scatter plot tidak ditampilkan untuk {metode}.**  
                        Dataset hanya memiliki **satu tahun data**, sehingga tidak ada hubungan antar tahun yang dapat divisualisasikan.  
                        Silakan lihat *boxplot* atau *peta* untuk distribusi cluster.
                        """)

                st.success("✅ Scatter plot semua metode selesai dibuat!")
