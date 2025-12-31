import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import leafmap.foliumap as leafmap
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# Pastikan file modules/interpretasi.py ada di folder yang sama
from modules.interpretasi import interpretasi_untuk_legend_otomatis

# =========================================================
# 🥚 KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Analisis Clustering Telur Ayam Ras",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# ℹ️ SIDEBAR INFORMASI
# =========================================================

st.sidebar.markdown("### 📘 Panduan Penggunaan")
st.sidebar.write(
    "Unduh *Manual Book* untuk panduan lengkap penggunaan aplikasi.")

# Membaca file manual book dari sidebar
try:
    with open("data/Manual_Book_Clustering_Telur_Ayam_Ras.pdf", "rb") as file:
        st.sidebar.download_button(
            label="📥 Download Manual Book (PDF)",
            data=file,
            file_name="Manual_Book_Clustering_Telur_Ayam_Ras.pdf",
            mime="application/pdf",
            use_container_width=True  # Membuat tombol full-width di sidebar
        )
except FileNotFoundError:
    st.sidebar.error(
        "File 'Manual_Book_Clustering_Telur_Ayam_Ras.pdf' tidak ditemukan di folder 'data/'.")


# =========================================================
# 🎨 CSS STYLING
# =========================================================
st.markdown("""
<style>
.minimal-title {font-size: 4rem; font-weight: 700; line-height: 1.1; color: #222; margin-bottom: 0.5rem; padding-top: 1rem;}
.minimal-subtitle {font-size: 1.75rem; color: #555; font-weight: 300; margin-bottom: 2rem;}
.hero-description {font-size: 1.1rem; color: #4A4A4A; margin-bottom: 2.5rem; max-width: 900px;} /* Sedikit dilebarkan */
.metric-card {background-color: #fff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 25px;
text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08); transition: transform 0.3s ease, box-shadow 0.3s ease;}
.metric-card:hover {transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.12);}
.metric-icon {font-size: 3.5rem; margin-bottom: 15px;}
.metric-card h3 {font-size: 1.15rem; color: #4A4A4A; margin-bottom: 5px; font-weight: 600;}
.metric-card h2 {font-size: 2.2rem; color: #004c99; margin: 0; font-weight: 700;}
.warning-box {background-color:#fff3cd; color:#664d03; border:1px solid #ffecb5; border-radius:8px;
padding:12px 16px; margin-top:1.5rem; font-size:14px;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🧭 HERO & PENJELASAN TOPIK (ALUR BARU)
# =========================================================

# 1. Judul & Subjudul
st.markdown('<h1 class="minimal-title">Analisis Klasterisasi Telur Ayam Ras di Indonesia</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="minimal-subtitle">Pemetaan Pola Harga, Konsumsi, dan Pengeluaran</p>',
            unsafe_allow_html=True)
st.divider()

# 2. Penjelasan "Kenapa Telur Ayam Ras?" + Gambar
st.markdown("### 🐔 Mengapa Telur Ayam Ras?")
# 1/3 lebar untuk gambar, 2/3 lebar untuk teks
col_img, col_text = st.columns([1, 2])

with col_img:
    # 3. Gambar Telur (Sesuai permintaanmu)
    # PENTING: Ganti 'data/telur_ayam_ras.jpg' dengan path gambar kamu
    try:
        st.image("data/telur_ayam_ras.jpg", caption="")
    except FileNotFoundError:
        st.warning(
            "File gambar 'data/telur_ayam_ras.jpg' tidak ditemukan. Silakan tambahkan file tersebut.")

with col_text:
    st.markdown("""
    Telur ayam ras dipilih sebagai komoditas utama untuk analisis ini karena perannya yang krusial bagi masyarakat Indonesia:
    - Merupakan **sumber protein hewani termurah** dan paling banyak dikonsumsi oleh seluruh lapisan rumah tangga.
    - Memiliki **fluktuasi harga tinggi** yang berpengaruh langsung terhadap daya beli masyarakat dan menjadi salah satu penyumbang inflasi bahan pangan.
""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:25px;'></div>",
            unsafe_allow_html=True)  # Spacer

# 4. Penjelasan Aplikasi (Tujuan)
st.markdown("### 🎯 Tujuan Aplikasi Ini")
st.markdown("""
<p class="hero-description">
Aplikasi ini menyajikan hasil analisis <i>clustering</i> untuk mengidentifikasi pola dan mengelompokkan wilayah di Indonesia berdasarkan karakteristik pasar telur ayam ras.
Tujuannya adalah untuk memetakan wilayah berdasarkan 3 variabel kunci:
<ul>
    <li><b>Harga:</b> Mencerminkan dinamika pasar dan biaya distribusi.</li>
    <li><b>Konsumsi:</b> Menunjukkan tingkat permintaan dan kebutuhan masyarakat.</li>
    <li><b>Pengeluaran:</b> Merepresentasikan daya beli dan beban ekonomi rumah tangga terkait telur.</li>
</ul>
Aplikasi ini membandingkan tiga metode clustering (<b>K-Means</b>, <b>AHC</b>, dan <b>Intelligent K-Medoids</b>) untuk menemukan model pengelompokan terbaik.
</p>
""", unsafe_allow_html=True)


# =========================================================
# 📊 KPI
# =========================================================
st.subheader("Sekilas Data Analisis")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="metric-card"><span class="metric-icon">📈</span><h3>Jumlah Wilayah</h3><h2>487 Kabupaten/Kota</h2></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="metric-card"><span class="metric-icon">📊</span><h3>Sumber Data</h3><h2>BPS & Bapanas<br></h2></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="metric-card"><span class="metric-icon">📅</span><h3>Periode (Tahun)</h3><h2>2022–2024<br></h2></div>""", unsafe_allow_html=True)
st.divider()

# =========================================================
# ⚙️ TAB METODE YANG DIGUNAKAN
# =========================================================
st.markdown("### ⚙️ Metode yang Digunakan")
tab1, tab2, tab3 = st.tabs(
    ["**K-Means**", "**Agglomerative (AHC)**", "**Intelligent K-Medoids**"])

with tab1:
    st.markdown("""
    **K-Means** adalah metode partisi yang membagi data ke dalam $K$ klaster berdasarkan jarak terdekat ke centroid (titik pusat rata-rata).
    - **Kelebihan:** Cepat, efisien secara komputasi, dan cocok untuk dataset besar dengan variabel numerik.
    - **Kekurangan:** Sensitif terhadap *outlier* dan penentuan jumlah $K$ di awal.
    """)

with tab2:
    st.markdown("""
    **Agglomerative Hierarchical Clustering (AHC)** adalah metode hierarki *bottom-up* yang membangun klaster secara bertahap.
    - **Kelebihan:** Menghasilkan visualisasi hierarki (dendrogram) yang baik untuk memahami kedekatan antar data. Tidak perlu menentukan jumlah klaster di awal.
    - **Kekurangan:** Lambat secara komputasi untuk dataset besar.
    """)

with tab3:
    st.markdown("""
    **Intelligent K-Medoids** (berdasarkan algoritma *FasterPAM*) adalah versi tangguh dari K-Means yang menggunakan *medoid* (titik data nyata) sebagai pusat klaster.
    - **Kelebihan:** Sangat **tahan terhadap *outlier*** dan data *noise* karena menggunakan medoid.
    - **Kekurangan:** Sedikit lebih lambat dari K-Means, namun memberikan hasil yang lebih stabil pada data dengan variansi besar.
    """)

st.markdown("<div style='margin-top:45px;'></div>", unsafe_allow_html=True)

# =========================================================
# 📂 LOAD & PROSES DATA
# =========================================================


# @st.cache_data
# def load_all_sheets(excel_path):
#     xls = pd.ExcelFile(excel_path)
#     frames = []
#     for sheet in xls.sheet_names:
#         df_temp = pd.read_excel(excel_path, sheet_name=sheet)
#         df_temp["Tahun"] = sheet
#         frames.append(df_temp)
#     return pd.concat(frames, ignore_index=True)




# # Pastikan path file data/Dataset Ready.xlsx sudah benar
# try:
#     excel_path = "data/Dataset Ready.xlsx"
#     df = load_all_sheets(excel_path)
#     df.columns = df.columns.str.strip()
#     df["Kabupaten/Kota"] = df["Kabupaten/Kota"].astype(
#         str).str.strip().str.upper()

#     fitur = [c for c in df.select_dtypes(
#         include=[np.number]).columns if c not in ["Cluster"]]
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(df[fitur])

#     # Model Clustering (Contoh menggunakan K-Means, sesuaikan jika perlu)
#     # Di aplikasi skripsi aslimu, ini mungkin akan dinamis
#     kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
#     df["Cluster"] = kmeans.fit_predict(X_scaled)
#     cluster_colors = {0: "#4C72B0", 1: "#DD8452"}  # Biru dan Oranye

    

#     # =========================================================
#     # 🗺️ PETA INTERAKTIF
#     # =========================================================
#     st.markdown("### 🗺️ Peta Visualisasi Hasil Clustering")
#     with st.spinner("🔄 Memuat peta..."):

#         @st.cache_data
#         def load_geojson():
#             geo_path = "data/Indonesia_cities.geojson"
#             gdf = gpd.read_file(geo_path)
#             gdf["NAME_2"] = gdf["NAME_2"].str.strip().str.upper()
#             return gdf

#         gdf = load_geojson()
#         gdf["NAME_2"] = gdf["NAME_2"].str.replace("KABUPATEN", "", case=False).str.replace(
#             "KOTA", "", case=False).str.strip().str.upper()
#         df["Kabupaten/Kota"] = df["Kabupaten/Kota"].str.replace(
#             "KABUPATEN", "", case=False).str.replace("KOTA", "", case=False).str.strip().str.upper()

#         # Merge data
#         gdf = gdf.merge(df[["Kabupaten/Kota", "Cluster"] + fitur],
#                         left_on="NAME_2", right_on="Kabupaten/Kota", how="left")

#         def make_tooltip(row):
#             if pd.isna(row["Kabupaten/Kota"]) or pd.isna(row["Cluster"]):
#                 return "<b>Tidak Ada Data</b>"

#             # Format judul (Title Case)
#             nama_wilayah = row['Kabupaten/Kota'].title().replace("Dki ",
#                                                                  "DKI ")
#             teks = f"<b>{nama_wilayah}</b><br><b>Cluster:</b> {int(row['Cluster'])}<hr>"

#             # Format data fitur
#             for f in fitur:
#                 if f in row and pd.notnull(row[f]):
#                     nilai = row[f]
#                     # Format Rupiah untuk Harga dan Pengeluaran
#                     if "Harga" in f or "Pengeluaran" in f:
#                         teks_nilai = f"Rp {nilai:,.0f}"
#                     else:
#                         teks_nilai = f"{nilai:,.2f}"
#                     teks += f"<b>{f}:</b> {teks_nilai}<br>"
#             return teks

#         gdf["info"] = gdf.apply(make_tooltip, axis=1)

#         m = leafmap.Map(center=[-2.5, 118], zoom=5)
#         m.add_basemap("CartoDB.Positron")

#         geo_layer = folium.GeoJson(
#             gdf[["geometry", "Cluster", "info"]],
#             name="Peta Cluster",
#             style_function=lambda x: {
#                 "fillColor": (
#                     cluster_colors.get(x["properties"].get("Cluster"))
#                     if pd.notnull(x["properties"].get("Cluster"))
#                     else "#C8C8C8"  # Warna abu-abu untuk data NaN
#                 ),
#                 "color": "#4d4d4d",  # Warna batas
#                 "weight": 0.4,
#                 "opacity": 0.6,
#                 "fillOpacity": 0.9,
#             },
#             tooltip=folium.GeoJsonTooltip(fields=["info"], aliases=[
#                                           ""], labels=False, sticky=True),
#         )
#         m.add_layer(geo_layer)
#         m.to_streamlit(height=550)

#     # =========================================================
#     # 🧭 INTERPRETASI CLUSTER
#     # =========================================================
#     st.markdown("### 🧩 Interpretasi Cluster")
#     # Memanggil fungsi interpretasi otomatis
#     try:
#         auto_labels = interpretasi_untuk_legend_otomatis(df, fitur)
#         for cluster in sorted(auto_labels.keys()):
#             color = cluster_colors.get(
#                 cluster, list(cluster_colors.values())[0])
#             label = auto_labels[cluster]
#             st.markdown(
#                 f"""
#                 <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
#                     <div style='width:16px; height:16px; background-color:{color};
#                                 border-radius:3px; flex-shrink:0;'></div>
#                     <div style='font-size:14px; color:#333; line-height:1.4;'>
#                         <b>Cluster {cluster} :</b> {label}
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#     except NameError:
#         st.error("Fungsi 'interpretasi_untuk_legend_otomatis' tidak ditemukan. Pastikan file 'modules/interpretasi.py' ada dan benar.")
#     except Exception as e:
#         st.error(f"Error saat membuat interpretasi: {e}")

#     st.markdown("<div style='margin-top: 25px;'></div>",
#                 unsafe_allow_html=True)
#     # =========================================================
#     # 📦 BOX PLOT
#     # =========================================================
#     with st.expander("📦 Lihat Distribusi Nilai Tiap Variabel per Cluster dan Tahun"):
#         with st.spinner("🔄 Membuat Boxplot..."):
#             plt.close("all")
#             n_vars = len(fitur)
#             n_cols = min(n_vars, 3)  # Maksimal 3 kolom
#             cols = st.columns(n_cols, gap="large")
#             palette = ["#4C72B0", "#DD8452"]

#             for i, var in enumerate(fitur):
#                 with cols[i % n_cols]:
#                     fig, ax = plt.subplots(figsize=(4.3, 3.5))
#                     sns.boxplot(
#                         x="Tahun", y=var, hue="Cluster",
#                         data=df, palette=palette, ax=ax,
#                         fliersize=2, linewidth=0.8
#                     )
#                     ax.set_xlabel("Tahun")
#                     ax.set_ylabel(var)
#                     ax.grid(True, axis="y", linestyle="--", alpha=0.3)

#                     # Format Y-axis (misal: Rupiah)
#                     if "Harga" in var or "Pengeluaran" in var:
#                         ax.yaxis.set_major_formatter(
#                             plt.FuncFormatter(lambda x, p: f'Rp {x:,.0f}'))

#                     ax.get_legend().remove()
#                     plt.tight_layout()
#                     st.pyplot(fig)
#                     plt.close(fig)

#     # =========================================================
#     # ⚠️ CATATAN / WARNING
#     # =========================================================
#     st.markdown("""
#     <div class="warning-box">
#     ⚠️ <b>Catatan:</b> Hasil klasterisasi yang ditampilkan di halaman ini (default) merupakan hasil terbaik yang diperoleh 
#     berdasarkan evaluasi gabungan indeks <b>Silhouette</b> (tertinggi) dan <b>Davies–Bouldin Index</b> (terendah) 
#     dari seluruh metode yang diuji pada halaman 'Eksperimen'.
#     </div>
#     """, unsafe_allow_html=True)

# # Error handling jika file utama tidak ada
# except FileNotFoundError as e:
#     if "Dataset Ready.xlsx" in str(e):
#         st.error(
#             "File 'data/Dataset Ready.xlsx' tidak ditemukan. Pastikan file data utama ada.")
#     elif "Indonesia_cities.geojson" in str(e):
#         st.error(
#             "File 'data/Indonesia_cities.geojson' tidak ditemukan. Pastikan file geojson ada.")
#     else:
#         st.error(f"File tidak ditemukan: {e}")
# except ImportError:
#     st.error("Modul 'interpretasi' tidak ditemukan. Pastikan file 'modules/interpretasi.py' ada di folder 'modules/'.")
# except Exception as e:
#     st.error(f"Terjadi error saat memuat data atau memproses halaman: {e}")


# =========================================================
# 📂 LOAD & PROSES DATA
# =========================================================

# ---------- DATASET CLUSTERING (TAHUNAN) ----------

@st.cache_data
def load_all_sheets(excel_path):
    xls = pd.ExcelFile(excel_path)
    frames = []
    for sheet in xls.sheet_names:
        df_temp = pd.read_excel(excel_path, sheet_name=sheet)
        df_temp["Tahun"] = sheet
        frames.append(df_temp)
    return pd.concat(frames, ignore_index=True)

try:
    excel_path = "data/Dataset Ready.xlsx"
    df = load_all_sheets(excel_path)

    df.columns = df.columns.str.strip()
    df["Kabupaten/Kota"] = df["Kabupaten/Kota"].astype(str).str.strip().str.upper()

    fitur = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Cluster"]]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[fitur])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_colors = {0: "#4C72B0", 1: "#DD8452"}  # Biru dan Oranye

except Exception as e:
    st.error(f"❌ Error saat memuat dataset clustering: {e}")


# ---------- DATASET HARGA HARIAN (WILAYAH KECIL) ----------

@st.cache_data
def load_harga_harian(path):
    df_raw = pd.read_excel(path)
    df_raw.columns = df_raw.columns.astype(str)

    id_col = df_raw.columns[0]  # Kabupaten/Kota

    df_long = df_raw.melt(
        id_vars=id_col,
        var_name="Tanggal",
        value_name="Harga"
    )

    df_long["Tanggal"] = pd.to_datetime(df_long["Tanggal"], errors="coerce")
    df_long[id_col] = df_long[id_col].str.strip().str.upper()
    df_long = df_long.rename(columns={id_col: "Kabupaten/Kota"})

    df_long = df_long.dropna(subset=["Harga", "Tanggal"])
    return df_long

try:
    df_harian = load_harga_harian("data/harga_harian.xlsx")
except Exception as e:
    st.warning(f"⚠️ Dataset harga harian tidak terbaca: {e}")
    df_harian = None


# =========================================================
# ✅ TAB UTAMA HOME
# =========================================================

st.markdown("## 📌 Panel Informasi")
tab_info, tab_map = st.tabs([
    "📊 Informasi Harga Harian",
    "🗺️ Pemetaan Clustering"
])

# =========================================================
# 📂 LOAD DATA HARIAN (FORMAT WIDE → LONG)
# =========================================================
@st.cache_data
def load_harga_harian():
    df = pd.read_excel("data/harga_harian.xlsx", sheet_name=0)

    df = df.rename(columns={df.columns[0]: "Kabupaten/Kota"})
    df["Kabupaten/Kota"] = df["Kabupaten/Kota"].astype(str).str.strip()

    df_long = df.melt(
        id_vars=["Kabupaten/Kota"],
        var_name="Tanggal",
        value_name="Harga"
    )

    df_long["Tanggal"] = pd.to_datetime(df_long["Tanggal"], errors="coerce")
    df_long["Harga"] = pd.to_numeric(df_long["Harga"], errors="coerce")
    df_long = df_long.dropna(subset=["Tanggal", "Harga"])

    df_long["Tahun"] = df_long["Tanggal"].dt.year.astype(int)

    return df_long


# =========================================================
# 📊 TAB — INFORMASI HARGA HARIAN (RANGE TAHUN)
# =========================================================
with tab_info:
    st.subheader("📊 Informasi Harga Telur Ayam Ras Harian")

    df_harian = load_harga_harian()

    # =====================================================
    # 🔽 RANGE TAHUN (contoh: 2022–2023, 2022–2024)
    # =====================================================
    min_year = int(df_harian["Tahun"].min())
    max_year = int(df_harian["Tahun"].max())

    start_year, end_year = st.slider(
        "Pilih Range Tahun",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )

    df_year = df_harian[
        (df_harian["Tahun"] >= start_year) &
        (df_harian["Tahun"] <= end_year)
    ].copy()

    # =====================================================
    # 🔽 DROPDOWN Kabupaten/Kota
    # =====================================================
    kota = st.selectbox(
        "Pilih Kabupaten/Kota",
        sorted(df_year["Kabupaten/Kota"].unique())
    )

    df_kota = df_year[df_year["Kabupaten/Kota"] == kota].sort_values("Tanggal")

    # =====================================================
    # 📅 RANGE TANGGAL
    # =====================================================
    min_date = df_kota["Tanggal"].min().date()
    max_date = df_kota["Tanggal"].max().date()

    start_date, end_date = st.date_input(
        "Pilih Range Tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    df_final = df_kota[
        (df_kota["Tanggal"].dt.date >= start_date) &
        (df_kota["Tanggal"].dt.date <= end_date)
    ]

    if df_final.empty:
        st.warning("Tidak ada data pada rentang ini.")
        st.stop()

    # =====================================================
    # 📊 METRIK
    # =====================================================
    df_final = df_final.sort_values("Tanggal")

    harga_today = df_final.iloc[-1]["Harga"]
    tanggal_today = df_final.iloc[-1]["Tanggal"].strftime("%d %B %Y")

    harga_min = df_final["Harga"].min()
    harga_max = df_final["Harga"].max()
    harga_avg = df_final["Harga"].mean()

    st.metric("💰 Harga Terbaru", f"Rp {harga_today:,.0f}", tanggal_today)

    c1, c2, c3 = st.columns(3)
    c1.metric("📉 Minimum", f"Rp {harga_min:,.0f}")
    c2.metric("📈 Maksimum", f"Rp {harga_max:,.0f}")
    c3.metric("📊 Rata-rata", f"Rp {harga_avg:,.0f}")

    # =====================================================
    # 📈 GRAFIK
    # =====================================================
    st.line_chart(df_final.set_index("Tanggal")["Harga"])

    st.caption("Sumber: Badan Pangan Nasional")

# =========================================================
# ✅ TAB 2 — PEMETAAN CLUSTERING ()
# =========================================================

# with tab_map:
#     st.subheader("🗺️ Peta Visualisasi Hasil Clustering")

#     if df is None:
#         st.error("Dataset clustering tidak tersedia.")
#     else:
#         with st.spinner("🔄 Memuat peta..."):

#             @st.cache_data
#             def load_geojson():
#                 geo_path = "data/Indonesia_cities.geojson"
#                 gdf = gpd.read_file(geo_path)
#                 gdf["NAME_2"] = gdf["NAME_2"].str.strip().str.upper()
#                 return gdf

#             gdf = load_geojson()

#             gdf["NAME_2"] = gdf["NAME_2"].str.replace(
#                 "KABUPATEN", "", case=False
#             ).str.replace(
#                 "KOTA", "", case=False
#             ).str.strip().str.upper()

#             df["Kabupaten/Kota"] = df["Kabupaten/Kota"].str.replace(
#                 "KABUPATEN", "", case=False
#             ).str.replace(
#                 "KOTA", "", case=False
#             ).str.strip().str.upper()

#             gdf = gdf.merge(
#                 df[["Kabupaten/Kota", "Cluster"] + fitur],
#                 left_on="NAME_2", right_on="Kabupaten/Kota", how="left"
#             )

#             def make_tooltip(row):
#                 if pd.isna(row["Kabupaten/Kota"]) or pd.isna(row["Cluster"]):
#                     return "<b>Tidak Ada Data</b>"

#                 nama_wilayah = row['Kabupaten/Kota'].title().replace("Dki ", "DKI ")
#                 teks = f"<b>{nama_wilayah}</b><br><b>Cluster:</b> {int(row['Cluster'])}<hr>"

#                 for f in fitur:
#                     if pd.notnull(row[f]):
#                         nilai = row[f]
#                         if "Harga" in f or "Pengeluaran" in f:
#                             teks += f"<b>{f}:</b> Rp {nilai:,.0f}<br>"
#                         else:
#                             teks += f"<b>{f}:</b> {nilai:,.2f}<br>"
#                 return teks

#             gdf["info"] = gdf.apply(make_tooltip, axis=1)

#             m = leafmap.Map(center=[-2.5, 118], zoom=5)
#             m.add_basemap("CartoDB.Positron")

#             geo_layer = folium.GeoJson(
#                 gdf[["geometry", "Cluster", "info"]],
#                 style_function=lambda x: {
#                     "fillColor": (
#                         cluster_colors.get(x["properties"].get("Cluster"))
#                         if pd.notnull(x["properties"].get("Cluster"))
#                         else "#C8C8C8"
#                     ),
#                     "color": "#4d4d4d",
#                     "weight": 0.4,
#                     "opacity": 0.6,
#                     "fillOpacity": 0.9,
#                 },
#                 tooltip=folium.GeoJsonTooltip(fields=["info"], labels=False, sticky=True),
#             )

#             m.add_layer(geo_layer)
#             m.to_streamlit(height=550)

with tab_map:
    st.subheader("🗺️ Visualisasi & Interpretasi Hasil Clustering")

    if df is None:
        st.error("Dataset clustering tidak tersedia.")
    else:

        # =========================================================
        # 🗺️ PETA INTERAKTIF
        # =========================================================
        with st.spinner("🔄 Memuat peta..."):

            @st.cache_data
            def load_geojson():
                geo_path = "data/Indonesia_cities.geojson"
                gdf = gpd.read_file(geo_path)
                gdf["NAME_2"] = gdf["NAME_2"].str.strip().str.upper()
                return gdf

            gdf = load_geojson()
            gdf["NAME_2"] = gdf["NAME_2"].str.replace(
                "KABUPATEN", "", case=False
            ).str.replace(
                "KOTA", "", case=False
            ).str.strip().str.upper()

            df["Kabupaten/Kota"] = df["Kabupaten/Kota"].str.replace(
                "KABUPATEN", "", case=False
            ).str.replace(
                "KOTA", "", case=False
            ).str.strip().str.upper()

            # Merge data
            gdf = gdf.merge(
                df[["Kabupaten/Kota", "Cluster"] + fitur],
                left_on="NAME_2",
                right_on="Kabupaten/Kota",
                how="left"
            )

            def make_tooltip(row):
                if pd.isna(row["Kabupaten/Kota"]) or pd.isna(row["Cluster"]):
                    return "<b>Tidak Ada Data</b>"

                nama_wilayah = row["Kabupaten/Kota"].title().replace("Dki ", "DKI ")
                teks = f"<b>{nama_wilayah}</b><br><b>Cluster:</b> {int(row['Cluster'])}<hr>"

                for f in fitur:
                    if f in row and pd.notnull(row[f]):
                        nilai = row[f]
                        if "Harga" in f or "Pengeluaran" in f:
                            teks_nilai = f"Rp {nilai:,.0f}"
                        else:
                            teks_nilai = f"{nilai:,.2f}"
                        teks += f"<b>{f}:</b> {teks_nilai}<br>"
                return teks

            gdf["info"] = gdf.apply(make_tooltip, axis=1)

            m = leafmap.Map(center=[-2.5, 118], zoom=5)
            m.add_basemap("CartoDB.Positron")

            geo_layer = folium.GeoJson(
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
                tooltip=folium.GeoJsonTooltip(
                    fields=["info"],
                    aliases=[""],
                    labels=False,
                    sticky=True
                ),
            )

            m.add_layer(geo_layer)
            m.to_streamlit(height=550)

        # =========================================================
        # 🧭 INTERPRETASI CLUSTER
        # =========================================================
        st.markdown("### 🧩 Interpretasi Cluster")

        try:
            auto_labels = interpretasi_untuk_legend_otomatis(df, fitur)
            for cluster in sorted(auto_labels.keys()):
                color = cluster_colors.get(cluster, list(cluster_colors.values())[0])
                label = auto_labels[cluster]
                st.markdown(
                    f"""
                    <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                        <div style='width:16px; height:16px; background-color:{color};
                                    border-radius:3px;'></div>
                        <div style='font-size:14px;'>
                            <b>Cluster {cluster}:</b> {label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error interpretasi cluster: {e}")

        st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)

        # =========================================================
        # 📦 BOX PLOT
        # =========================================================
        with st.expander("📦 Lihat Distribusi Nilai Tiap Variabel per Cluster dan Tahun"):
            with st.spinner("🔄 Membuat Boxplot..."):
                plt.close("all")

                n_vars = len(fitur)
                n_cols = min(n_vars, 3)
                cols = st.columns(n_cols, gap="large")
                palette = ["#4C72B0", "#DD8452"]

                for i, var in enumerate(fitur):
                    with cols[i % n_cols]:
                        fig, ax = plt.subplots(figsize=(4.3, 3.5))
                        sns.boxplot(
                            x="Tahun",
                            y=var,
                            hue="Cluster",
                            data=df,
                            palette=palette,
                            ax=ax,
                            fliersize=2,
                            linewidth=0.8
                        )
                        ax.set_xlabel("Tahun")
                        ax.set_ylabel(var)
                        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

                        if "Harga" in var or "Pengeluaran" in var:
                            ax.yaxis.set_major_formatter(
                                plt.FuncFormatter(lambda x, p: f"Rp {x:,.0f}")
                            )

                        ax.get_legend().remove()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

        # =========================================================
        # ⚠️ CATATAN
        # =========================================================
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Catatan:</b> Hasil klasterisasi yang ditampilkan di halaman ini merupakan hasil terbaik
        berdasarkan evaluasi gabungan <b>Silhouette</b> (tertinggi) dan
        <b>Davies–Bouldin Index</b> (terendah) dari seluruh metode pada halaman Eksperimen.
        </div>
        """, unsafe_allow_html=True)


st.divider()
st.caption("Dikembangkan oleh Shareen Stephanie (535220112) © 2025.")
