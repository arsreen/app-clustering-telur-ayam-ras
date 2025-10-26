import streamlit as st
import pandas as pd
import io # Diperlukan untuk in-memory file

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dataset Analisis", 
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Analisis Telur Ayam Ras")
st.markdown("""
Halaman ini adalah pusat data untuk analisis. Anda bisa menggunakan **dataset bawaan** atau **mengunggah data Anda sendiri** untuk dianalisis di halaman *Eksperimen*.
""")

# --- Fungsi Bantuan ---

@st.cache_data # Cache data bawaan agar tidak di-load ulang
def load_default_data():
    # Ganti path ini jika salah
    df = pd.read_excel("data/Dataset Ready.xlsx", engine="openpyxl") 
    return df

@st.cache_data # Fungsi untuk konversi dataframe ke excel di memori
def to_excel_in_memory(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Template')
    processed_data = output.getvalue()
    return processed_data

# --- Inisialisasi Session State ---
# Ini penting agar data tersimpan saat pindah halaman
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'active_data_source' not in st.session_state:
    st.session_state.active_data_source = "Bawaan"

# Muat data bawaan
default_df = load_default_data()

# ==============================
# 2. PREVIEW DATA AKTIF
# ==============================
st.subheader("Preview Data yang Aktif")

# Tentukan dataframe mana yang akan ditampilkan
if st.session_state.active_data_source == "Upload Data Sendiri" and st.session_state.uploaded_data is not None:
    st.info("Menampilkan data dari file yang Anda upload.")
    active_df = st.session_state.uploaded_data
    # Simpan data aktif untuk dibaca halaman 'Eksperimen'
    st.session_state.active_data = active_df
else:
    st.info("Menampilkan data bawaan (default).")
    active_df = default_df
    # Simpan data aktif untuk dibaca halaman 'Eksperimen'
    st.session_state.active_data = active_df

st.dataframe(active_df.head(10), use_container_width=True)
st.caption(f"Menampilkan 10 baris pertama dari {active_df.shape[0]} data total.")


# ==============================
# 3. INFO FORMAT & DOWNLOAD TEMPLATE
# ==============================
st.markdown("---")
st.subheader("Informasi Format & Template")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("##### 📋 Format Dataset")
    st.markdown("""
    Pastikan data Anda memiliki format kolom berikut:
    - 🏙️ **Kabupaten/Kota**
    - 💰 **Harga Telur Ayam Ras**
    - 🍳 **Konsumsi Per Kapita**
    - 🧾 **Pengeluaran Per Kapita**
    """)

with col2:
    st.markdown("##### ⬇️ Unduh Template")
    st.markdown("Gunakan template ini untuk memastikan format data Anda benar.")

    # Buat template
    template_df = pd.DataFrame({
        "Kabupaten/Kota": ["Contoh Kabupaten A", "Contoh Kabupaten B"],
        "Harga Telur Ayam Ras (Rp)": [0.245, 0.312], # Nama kolom disingkat agar mudah
        "Konsumsi Telur Ayam Ras Per Kapita": [0.520, 0.474],
        "Pengeluaran Telur Ayam Ras (Rp)": [0.415, 0.386]
    })
    
    # Konversi ke excel di memori
    excel_data = to_excel_in_memory(template_df)

    st.download_button(
        label="📥 Download Template Dataset (.xlsx)",
        data=excel_data, # Data dari memori (bytes)
        file_name="Template_Dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )