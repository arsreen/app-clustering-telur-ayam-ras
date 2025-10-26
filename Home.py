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
# 🎨 CSS STYLING
# =========================================================
st.markdown("""
<style>
.minimal-title {font-size: 4rem; font-weight: 700; line-height: 1.1; color: #222; margin-bottom: 0.5rem; padding-top: 1rem;}
.minimal-subtitle {font-size: 1.75rem; color: #555; font-weight: 300; margin-bottom: 2rem;}
.hero-description {font-size: 1.1rem; color: #4A4A4A; margin-bottom: 2.5rem; max-width: 850px;}
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
# 🧭 HERO SECTION
# =========================================================
st.markdown('<h1 class="minimal-title">Analisis Klasterisasi Telur Ayam Ras di Indonesia</h1>', unsafe_allow_html=True)
st.markdown('<p class="minimal-subtitle">Pemetaan Pola Harga, Konsumsi, dan Pengeluaran</p>', unsafe_allow_html=True)
st.markdown("""
<p class="hero-description">
Aplikasi ini menyajikan hasil analisis <i>clustering</i> telur ayam ras di Indonesia berdasarkan data harga, konsumsi, dan pengeluaran rumah tangga.
Tujuannya untuk mengidentifikasi pola pasar dan efisiensi distribusi antar wilayah, menggunakan tiga metode: K-Means, AHC, dan Intelligent K-Medoids.
</p>
""", unsafe_allow_html=True)

# =========================================================
# 📊 KPI
# =========================================================
st.subheader("Sekilas Analisis")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="metric-card"><span class="metric-icon">📈</span><h3>Jumlah Wilayah</h3><h2>487 Kabupaten/Kota</h2></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="metric-card"><span class="metric-icon">📊</span><h3>Sumber Data</h3><h2>BPS & Bapanas<br></h2></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="metric-card"><span class="metric-icon">📅</span><h3>Periode (Tahun)</h3><h2>2022–2024<br></h2></div>""", unsafe_allow_html=True)
st.divider()

# =========================================================
# 💡 PENJELASAN PILIHAN KOMODITAS & VARIABEL
# =========================================================
st.markdown("### 🐔 Mengapa Telur Ayam Ras dan Tiga Variabel Ini Dipilih?")
st.markdown("""
Telur ayam ras dipilih sebagai komoditas utama karena:
- Merupakan **sumber protein hewani termurah** dan paling banyak dikonsumsi oleh rumah tangga Indonesia.
- Memiliki **fluktuasi harga tinggi** yang berpengaruh langsung terhadap daya beli masyarakat dan inflasi bahan pangan.

Tiga variabel yang digunakan yaitu **Harga**, **Konsumsi**, dan **Pengeluaran**, memiliki keterkaitan langsung:
- **Harga** mencerminkan dinamika pasar dan distribusi.
- **Konsumsi** menunjukkan tingkat permintaan masyarakat.
- **Pengeluaran** merepresentasikan daya beli dan beban ekonomi rumah tangga.
Dengan menganalisis ketiganya secara bersamaan, kita dapat memahami **struktur pasar telur ayam ras** dan mengelompokkan wilayah berdasarkan karakteristik pasarnya.
""")

# =========================================================
# 📘 TOMBOL DOWNLOAD MANUAL BOOK
# =========================================================
st.markdown("### 📘 Panduan Penggunaan Aplikasi")
st.write("Untuk memahami cara penggunaan aplikasi ini secara lengkap, Anda dapat mengunduh *Manual Book* berikut:")

with open("data/Manual_Book_Clustering_TelurAyamRas.pdf", "rb") as file:
    st.download_button(
        label="📥 Download Manual Book (PDF)",
        data=file,
        file_name="Manual_Book_Clustering_TelurAyamRas.pdf",
        mime="application/pdf"
    )

st.markdown("<div style='margin-bottom:35px;'></div>", unsafe_allow_html=True)

# =========================================================
# ⚙️ TAB METODE YANG DIGUNAKAN
# =========================================================
st.markdown("### ⚙️ Metode yang Digunakan")
tab1, tab2, tab3 = st.tabs(["**K-Means**", "**Agglomerative (AHC)**", "**Intelligent K-Medoids**"])

with tab1:
    st.markdown("""
    **K-Means** adalah metode partisi yang membagi data ke dalam $K$ klaster berdasarkan jarak terdekat ke centroid.
    - Cocok untuk dataset besar dengan variabel numerik.
    - Cepat dan efisien.
    """)

with tab2:
    st.markdown("""
    **Agglomerative Hierarchical Clustering (AHC)** membangun klaster secara bertahap dari bawah ke atas.
    - Cocok untuk memahami hierarki kesamaan antar wilayah.
    """)

with tab3:
    st.markdown("""
    **Intelligent K-Medoids** adalah versi tangguh dari K-Medoids yang tahan terhadap *outlier*.
    - Menggunakan titik data nyata (medoid) sebagai pusat.
    - Memberikan stabilitas tinggi pada data dengan variansi besar.
    """)

st.markdown("<div style='margin-top:45px;'></div>", unsafe_allow_html=True)

# =========================================================
# 📂 LOAD & PROSES DATA
# =========================================================
@st.cache_data
def load_all_sheets(excel_path):
    xls = pd.ExcelFile(excel_path)
    frames = []
    for sheet in xls.sheet_names:
        df_temp = pd.read_excel(excel_path, sheet_name=sheet)
        df_temp["Tahun"] = sheet
        frames.append(df_temp)
    return pd.concat(frames, ignore_index=True)

excel_path = "data/Dataset Ready.xlsx"
df = load_all_sheets(excel_path)
df.columns = df.columns.str.strip()
df["Kabupaten/Kota"] = df["Kabupaten/Kota"].astype(str).str.strip().str.upper()

fitur = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Cluster"]]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[fitur])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
cluster_colors = {0: "#4C72B0", 1: "#DD8452"}

# =========================================================
# 🗺️ PETA INTERAKTIF
# =========================================================
st.markdown("### 🗺️ Peta Visualisasi Hasil Clustering")
with st.spinner("🔄 Memuat peta..."):

    @st.cache_data
    def load_geojson():
        geo_path = "data/Indonesia_cities.geojson"
        gdf = gpd.read_file(geo_path)
        gdf["NAME_2"] = gdf["NAME_2"].str.strip().str.upper()
        return gdf

    gdf = load_geojson()
    gdf["NAME_2"] = gdf["NAME_2"].str.replace("KABUPATEN", "", case=False).str.replace("KOTA", "", case=False).str.strip().str.upper()
    df["Kabupaten/Kota"] = df["Kabupaten/Kota"].str.replace("KABUPATEN", "", case=False).str.replace("KOTA", "", case=False).str.strip().str.upper()
    gdf = gdf.merge(df[["Kabupaten/Kota", "Cluster"] + fitur], left_on="NAME_2", right_on="Kabupaten/Kota", how="left")

    def make_tooltip(row):
        if pd.isna(row["Kabupaten/Kota"]) or pd.isna(row["Cluster"]):
            return "<b>Tidak Ada Data</b>"
        teks = f"<b>{row['Kabupaten/Kota'].title()}</b><br><b>Cluster:</b> {int(row['Cluster'])}<hr>"
        for f in fitur:
            if f in row and pd.notnull(row[f]):
                teks += f"<b>{f}:</b> {row[f]:,.2f}<br>"
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
        tooltip=folium.GeoJsonTooltip(fields=["info"], aliases=[""], labels=False, sticky=True),
    )
    m.add_layer(geo_layer)
    m.to_streamlit(height=550)

# =========================================================
# 🧭 INTERPRETASI CLUSTER
# =========================================================
st.markdown("### 🧩 Interpretasi Cluster")
auto_labels = interpretasi_untuk_legend_otomatis(df, fitur)
for cluster in sorted(auto_labels.keys()):
    color = cluster_colors.get(cluster, list(cluster_colors.values())[0])
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
        unsafe_allow_html=True,
    )

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
                    x="Tahun", y=var, hue="Cluster",
                    data=df, palette=palette, ax=ax,
                    fliersize=2, linewidth=0.8
                )
                ax.set_xlabel("Tahun")
                ax.set_ylabel(var)
                ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                ax.get_legend().remove()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# =========================================================
# ⚠️ CATATAN / WARNING
# =========================================================
st.markdown("""
<div class="warning-box">
⚠️ <b>Catatan:</b> Hasil klasterisasi yang ditampilkan merupakan hasil terbaik berdasarkan evaluasi indeks
<b>Silhouette</b> tertinggi dan <b>Davies–Bouldin Index</b> terendah dari seluruh metode yang diuji.
</div>
""", unsafe_allow_html=True)

st.divider()
st.caption("Dikembangkan oleh Shareen Stephanie (535220112) © 2025.")
